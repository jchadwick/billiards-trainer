#!/usr/bin/env bash
# Install Google's USB Edge TPU runtime from its checksum-pinned release asset.
# The legacy Coral apt repository is no longer available on every host.
set -euo pipefail

DEFAULT_RUNTIME_VERSION=20221024
DEFAULT_RUNTIME_SHA256=7f17d85e6f21ec861af3f9168043a7ec351b80e08c9f168013ab68e99410aee2
RUNTIME_VERSION="${CORAL_RUNTIME_VERSION:-$DEFAULT_RUNTIME_VERSION}"
RUNTIME_SHA256="${CORAL_RUNTIME_SHA256:-}"
RUNTIME_URL="${CORAL_RUNTIME_URL:-https://github.com/google-coral/libedgetpu/releases/download/release-grouper/edgetpu_runtime_${RUNTIME_VERSION}.zip}"
RUNTIME_ROOT="${CORAL_RUNTIME_ROOT:-/var/tmp/coral-edgetpu-runtime-${RUNTIME_VERSION}}"
SERVICE_USER="${CORAL_SERVICE_USER:-${SUDO_USER:-}}"
ARCHIVE="${RUNTIME_ROOT}/edgetpu_runtime_${RUNTIME_VERSION}.zip"

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

[[ "${EUID:-$(id -u)}" -eq 0 ]] || fail 'Run this installer as root (for example, sudo ./install-coral-host-runtime.sh).'
[[ "$(uname -s)" == Linux ]] || fail 'This installer supports Linux hosts only.'

case "$(uname -m)" in
    x86_64)
        EDGE_TPU_ARCH=k8
        MULTIARCH=x86_64-linux-gnu
        ;;
    aarch64)
        EDGE_TPU_ARCH=aarch64
        MULTIARCH=aarch64-linux-gnu
        ;;
    armv7l)
        EDGE_TPU_ARCH=armv7a
        MULTIARCH=arm-linux-gnueabihf
        ;;
    *)
        fail "Unsupported Linux architecture: $(uname -m)"
        ;;
esac

if [[ -z "$RUNTIME_SHA256" ]]; then
    [[ "$RUNTIME_VERSION" == "$DEFAULT_RUNTIME_VERSION" ]] \
        || fail "Set CORAL_RUNTIME_SHA256 when selecting runtime version $RUNTIME_VERSION."
    RUNTIME_SHA256=$DEFAULT_RUNTIME_SHA256
fi
[[ "$RUNTIME_SHA256" =~ ^[[:xdigit:]]{64}$ ]] || fail 'CORAL_RUNTIME_SHA256 must contain 64 hexadecimal digits.'
[[ -n "$SERVICE_USER" ]] || fail 'Set CORAL_SERVICE_USER or invoke this script with sudo so it can identify the service account.'

for command in curl getent id install ldconfig sha256sum stat tr udevadm unzip; do
    command -v "$command" >/dev/null || fail "Required command '$command' was not found."
done

LIBRARY_DIR="/usr/lib/$MULTIARCH"
RUNTIME_DIR="${RUNTIME_ROOT}/edgetpu_runtime"
LIBRARY_SOURCE="${RUNTIME_DIR}/libedgetpu/throttled/${EDGE_TPU_ARCH}/libedgetpu.so.1.0"
RULE_SOURCE="${RUNTIME_DIR}/libedgetpu/edgetpu-accelerator.rules"

getent group plugdev >/dev/null || fail 'The plugdev group is missing.'
id -nG "$SERVICE_USER" | tr ' ' '\n' | grep -Fxq plugdev \
    || fail "The service user '$SERVICE_USER' must be a member of plugdev."
ldconfig -p | grep -Fq 'libusb-1.0.so.0' \
    || fail 'libusb-1.0 runtime is missing; install the distribution package that provides it first.'

install -d -m 0755 "$RUNTIME_ROOT"
curl --fail --location --retry 3 --silent --show-error "$RUNTIME_URL" --output "$ARCHIVE"
printf '%s  %s\n' "$RUNTIME_SHA256" "$ARCHIVE" | sha256sum --check --status \
    || fail "Coral runtime archive checksum mismatch: $ARCHIVE"
unzip -oq "$ARCHIVE" -d "$RUNTIME_ROOT"
[[ -s "$LIBRARY_SOURCE" ]] || fail "Runtime library is missing from verified archive: $LIBRARY_SOURCE"
[[ -s "$RULE_SOURCE" ]] || fail "USB access rule is missing from verified archive: $RULE_SOURCE"

install -d -m 0755 "$LIBRARY_DIR" /etc/udev/rules.d
install -m 0644 "$LIBRARY_SOURCE" "${LIBRARY_DIR}/libedgetpu.so.1.0"
ln -sfn libedgetpu.so.1.0 "${LIBRARY_DIR}/libedgetpu.so.1"
install -m 0644 "$RULE_SOURCE" /etc/udev/rules.d/99-edgetpu-accelerator.rules
ldconfig
udevadm control --reload-rules

coral_count=0
for sys_device in /sys/bus/usb/devices/*; do
    [[ -r "$sys_device/idVendor" && -r "$sys_device/idProduct" ]] || continue
    vendor=$(<"$sys_device/idVendor")
    product=$(<"$sys_device/idProduct")
    [[ "$vendor:$product" == 1a6e:089a || "$vendor:$product" == 18d1:9302 ]] || continue
    udevadm trigger --action=add "$sys_device"
    coral_count=$((coral_count + 1))
done
udevadm settle
(( coral_count > 0 )) || fail 'No Coral USB device was found. Reconnect it, then rerun this installer.'
ldconfig -p | grep -Fq 'libedgetpu.so.1 (' || fail 'The Edge TPU runtime did not register with the dynamic linker.'

for sys_device in /sys/bus/usb/devices/*; do
    [[ -r "$sys_device/idVendor" && -r "$sys_device/idProduct" ]] || continue
    vendor=$(<"$sys_device/idVendor")
    product=$(<"$sys_device/idProduct")
    [[ "$vendor:$product" == 1a6e:089a || "$vendor:$product" == 18d1:9302 ]] || continue
    device_name=$(udevadm info --query=name --path="$sys_device")
    device_node="/dev/${device_name}"
    [[ -c "$device_node" ]] || fail "Coral USB node is missing: $device_node"
    [[ "$(stat -c '%G' "$device_node")" == plugdev ]] || fail "Coral USB node is not assigned to plugdev: $device_node"
    permissions=$(stat -c '%a' "$device_node")
    group_digit=${permissions: -2:1}
    [[ "$group_digit" =~ [2367] ]] || fail "Coral USB node lacks plugdev write permission: $device_node ($permissions)"
    printf 'Coral USB permissions ready: %s (%s plugdev)\n' "$device_node" "$permissions"
done

printf 'Coral runtime %s installed for %s.\n' "$RUNTIME_VERSION" "$SERVICE_USER"
