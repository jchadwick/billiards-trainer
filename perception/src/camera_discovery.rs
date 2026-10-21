//! Bounded, read-only camera inventory support for the opt-in GStreamer
//! discovery executable. DeviceMonitor metadata is inspected without creating
//! source elements or a capture pipeline.

use std::collections::BTreeMap;
use std::fmt;
use std::io::{self, Write};

use anyhow::{bail, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};

pub const CAMERA_DISCOVERY_SCHEMA_VERSION: u8 = 2;
const CAPTURE_SELECTOR_DOMAIN: &[u8] = b"billiards.capture-device-selector.v1\0";

/// A complete discovery snapshot. Missing provider metadata remains `null` so
/// clients can distinguish an unavailable field from an empty value.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CameraDiscoveryReport {
    pub schema_version: u8,
    pub platform: HostPlatform,
    pub gstreamer_version: String,
    pub provider_status: ProviderStatus,
    pub devices: Vec<CameraDevice>,
    /// True if the byte limit forced one or more fields/devices to be omitted.
    pub truncated: bool,
    pub omitted_device_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderStatus {
    /// At least one provider matched the requested Video/Source filter and
    /// DeviceMonitor started successfully.
    Started,
    /// No installed provider advertises the requested device class.
    NoMatchingProvider,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HostPlatform {
    pub os: String,
    pub architecture: String,
    /// Kernel/product version when the host exposes it through a standard
    /// platform utility. `None` is explicitly unavailable.
    pub version: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CameraDevice {
    pub display_name: Option<String>,
    pub device_class: Option<String>,
    pub provider_properties: Option<BTreeMap<String, ProviderProperty>>,
    /// The provider's advertised GStreamer caps string, not negotiated caps.
    pub advertised_caps: Option<String>,
    /// A provider identity is populated only for the documented stable forms
    /// or for an explicitly transient Linux `/dev/videoN` path.
    pub identifier_value: Option<String>,
    pub identifier_source: Option<String>,
    pub identifier_classification: IdentifierClassification,
    /// Versioned, safe selector derived from the exact persistent provider
    /// identity. It can be carried by capture-settings v3 without embedding
    /// Linux device paths in that request contract.
    pub capture_selector: Option<String>,
    /// Set when a large report had to omit this device's identity or fields.
    pub metadata_omitted: bool,
    pub omitted_property_count: usize,
    pub advertised_caps_omitted: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IdentifierClassification {
    /// AVFoundation's `avf.unique_id` value, which is persistent on that
    /// platform according to the provider contract.
    PlatformPersistent,
    /// A current Linux enumeration path. This is explicitly not persistent.
    EnumeratedTransient,
    /// The provider did not expose one of the recognized identity forms.
    Unavailable,
}

/// Derive a wire-safe selector from the exact provider key and persistent ID.
/// Domain separation and platform/key inclusion prevent collisions with
/// unrelated values or another provider's identity namespace.
pub fn capture_selector(platform_os: &str, provider_key: &str, stable_id: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(CAPTURE_SELECTOR_DOMAIN);
    for component in [
        platform_os.as_bytes(),
        provider_key.as_bytes(),
        stable_id.as_bytes(),
    ] {
        let length = u64::try_from(component.len())
            .expect("platform strings and provider identifiers fit a u64 length");
        digest.update(length.to_be_bytes());
        digest.update(component);
    }
    format!("gstreamer-device-v1:{:x}", digest.finalize())
}

/// Confirm that a ready-record provider pair is one of the stable identity
/// forms that discovery is allowed to hash.
pub fn is_persistent_provider_identity(platform_os: &str, provider_key: &str, value: &str) -> bool {
    if value.is_empty() || value.bytes().all(|byte| byte.is_ascii_digit()) {
        return false;
    }
    match platform_os {
        "macos" => {
            provider_key == "avf.unique_id"
                && !value.contains('/')
                && !value.chars().any(char::is_whitespace)
        }
        "linux" => ["/dev/v4l/by-id/", "/dev/v4l/by-path/"]
            .iter()
            .any(|prefix| {
                value
                    .strip_prefix(prefix)
                    .is_some_and(is_single_symlink_name)
            }),
        _ => false,
    }
}

/// Linux persistent V4L selectors name one symlink in by-id/by-path. Reject
/// traversal and nested paths so a device node or arbitrary subpath cannot be
/// mistaken for the stable discovery identity.
fn is_single_symlink_name(value: &str) -> bool {
    !value.is_empty()
        && value != "."
        && value != ".."
        && !value.contains('/')
        && !value.chars().any(char::is_control)
}

/// Select exactly one camera whose discovery-generated selector equals the
/// requested value. Transient `/dev/videoN` paths never receive selectors.
pub fn select_capture_selector_index(
    requested_selector: &str,
    devices: &[CameraDevice],
) -> Result<usize, StableDeviceSelectionError> {
    if !is_capture_selector(requested_selector) {
        return Err(StableDeviceSelectionError::UnstableIdentifier);
    }

    let mut matches = devices.iter().enumerate().filter(|(_, device)| {
        device.identifier_classification == IdentifierClassification::PlatformPersistent
            && device.capture_selector.as_deref() == Some(requested_selector)
    });
    let Some((index, _)) = matches.next() else {
        return Err(StableDeviceSelectionError::NoMatch);
    };
    if matches.next().is_some() {
        return Err(StableDeviceSelectionError::DuplicateMatches);
    }
    Ok(index)
}

/// Validate the safe selector syntax accepted by capture-settings v3.
pub fn is_capture_selector(value: &str) -> bool {
    value
        .strip_prefix("gstreamer-device-v1:")
        .is_some_and(|digest| {
            digest.len() == 64
                && digest
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StableDeviceSelectionError {
    UnstableIdentifier,
    NoMatch,
    DuplicateMatches,
}

impl fmt::Display for StableDeviceSelectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnstableIdentifier => formatter.write_str(
                "camera selection requires a platform-stable identifier; numeric indices and /dev/videoN are not accepted",
            ),
            Self::NoMatch => formatter.write_str(
                "no discovered camera matches the requested stable identifier",
            ),
            Self::DuplicateMatches => formatter.write_str(
                "multiple discovered cameras match the requested stable identifier",
            ),
        }
    }
}

impl std::error::Error for StableDeviceSelectionError {}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderProperty {
    pub value_type: String,
    /// GStreamer/GLib debug representation preserves type and safely handles
    /// values that are not simple strings or numbers.
    pub display_value: String,
}

/// Raw metadata kept only while converting provider properties. It is never
/// serialized separately; `string_value` is used solely for conservative ID
/// classification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawProviderProperty {
    pub value_type: String,
    pub display_value: String,
    pub string_value: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceMetadata {
    pub display_name: Option<String>,
    pub device_class: Option<String>,
    pub provider_properties: Option<BTreeMap<String, RawProviderProperty>>,
    pub advertised_caps: Option<String>,
}

impl CameraDevice {
    pub fn from_metadata(platform_os: &str, metadata: DeviceMetadata) -> Self {
        let raw_properties = metadata.provider_properties;
        let (identifier_value, identifier_source, identifier_classification) =
            classify_identifier(platform_os, raw_properties.as_ref());
        let capture_selector =
            if identifier_classification == IdentifierClassification::PlatformPersistent {
                identifier_source
                    .as_deref()
                    .zip(identifier_value.as_deref())
                    .filter(|(source, identifier)| {
                        is_persistent_provider_identity(platform_os, source, identifier)
                    })
                    .map(|(source, identifier)| capture_selector(platform_os, source, identifier))
            } else {
                None
            };
        let provider_properties = raw_properties.map(|properties| {
            properties
                .into_iter()
                .map(|(key, value)| {
                    (
                        key,
                        ProviderProperty {
                            value_type: value.value_type,
                            display_value: value.display_value,
                        },
                    )
                })
                .collect()
        });

        Self {
            display_name: metadata.display_name.filter(|value| !value.is_empty()),
            device_class: metadata.device_class.filter(|value| !value.is_empty()),
            provider_properties,
            advertised_caps: metadata.advertised_caps.filter(|value| !value.is_empty()),
            identifier_value,
            identifier_source,
            identifier_classification,
            capture_selector,
            metadata_omitted: false,
            omitted_property_count: 0,
            advertised_caps_omitted: false,
        }
    }
}

impl CameraDiscoveryReport {
    pub fn new(
        platform: HostPlatform,
        gstreamer_version: String,
        provider_status: ProviderStatus,
        devices: Vec<CameraDevice>,
    ) -> Self {
        Self {
            schema_version: CAMERA_DISCOVERY_SCHEMA_VERSION,
            platform,
            gstreamer_version,
            provider_status,
            devices,
            truncated: false,
            omitted_device_count: 0,
        }
    }

    /// Serialize one compact JSON object plus a newline without ever writing
    /// more than `max_output_bytes`. If necessary, remove provider details in
    /// a documented order and report the loss inside the bounded result.
    pub fn to_bounded_json(mut self, max_output_bytes: usize) -> Result<Vec<u8>> {
        if max_output_bytes < 2 {
            bail!("camera discovery output limit must leave room for JSON and a newline");
        }

        loop {
            let mut writer = LimitedVecWriter::new(max_output_bytes - 1);
            if serde_json::to_writer(&mut writer, &self).is_ok() {
                // The JSON writer's byte limit already reserves this final
                // newline; append it directly rather than asking the same
                // limit-enforcing writer to count it twice.
                writer.bytes.push(b'\n');
                return Ok(writer.bytes);
            }

            if !self.omit_next_largest_detail() {
                bail!("camera discovery metadata does not fit the configured output limit");
            }
            self.truncated = true;
        }
    }

    fn omit_next_largest_detail(&mut self) -> bool {
        // Remove many values per pass so even unusually large property sets
        // take logarithmically many serialization attempts.
        for device in self.devices.iter_mut().rev() {
            if let Some(properties) = device.provider_properties.as_mut() {
                if !properties.is_empty() {
                    let remove_count = properties.len().div_ceil(2);
                    let keys: Vec<String> = properties
                        .keys()
                        .rev()
                        .take(remove_count)
                        .cloned()
                        .collect();
                    for key in keys {
                        properties.remove(&key);
                        device.omitted_property_count += 1;
                    }
                    return true;
                }
            }
        }

        if let Some(device) = self
            .devices
            .iter_mut()
            .rev()
            .find(|device| device.advertised_caps.is_some())
        {
            device.advertised_caps = None;
            device.advertised_caps_omitted = true;
            return true;
        }

        if let Some(device) = self
            .devices
            .iter_mut()
            .rev()
            .find(|device| device.capture_selector.is_some())
        {
            device.capture_selector = None;
            device.metadata_omitted = true;
            return true;
        }

        if let Some(device) =
            self.devices.iter_mut().rev().find(|device| {
                device.identifier_value.is_some() || device.identifier_source.is_some()
            })
        {
            device.identifier_value = None;
            device.identifier_source = None;
            device.metadata_omitted = true;
            return true;
        }

        if let Some(device) = self
            .devices
            .iter_mut()
            .rev()
            .find(|device| device.display_name.is_some() || device.device_class.is_some())
        {
            device.display_name = None;
            device.device_class = None;
            device.metadata_omitted = true;
            return true;
        }

        if self.devices.pop().is_some() {
            self.omitted_device_count += 1;
            return true;
        }

        false
    }
}

fn classify_identifier(
    platform_os: &str,
    properties: Option<&BTreeMap<String, RawProviderProperty>>,
) -> (Option<String>, Option<String>, IdentifierClassification) {
    let Some(properties) = properties else {
        return (None, None, IdentifierClassification::Unavailable);
    };

    if platform_os == "macos" {
        if let Some(value) = properties
            .get("avf.unique_id")
            .and_then(|property| property.string_value.as_ref())
            .filter(|value| !value.is_empty())
        {
            return (
                Some(value.clone()),
                Some("avf.unique_id".to_owned()),
                IdentifierClassification::PlatformPersistent,
            );
        }
    }

    if platform_os == "linux" {
        for (key, property) in properties {
            let Some(value) = property.string_value.as_deref() else {
                continue;
            };
            if is_persistent_provider_identity(platform_os, key, value) {
                return (
                    Some(value.to_owned()),
                    Some(key.clone()),
                    IdentifierClassification::PlatformPersistent,
                );
            }
        }

        for (key, property) in properties {
            let Some(value) = property.string_value.as_deref() else {
                continue;
            };
            if is_video_node_path(value) {
                return (
                    Some(value.to_owned()),
                    Some(key.clone()),
                    IdentifierClassification::EnumeratedTransient,
                );
            }
        }
    }

    (None, None, IdentifierClassification::Unavailable)
}

fn is_video_node_path(value: &str) -> bool {
    value.strip_prefix("/dev/video").is_some_and(|suffix| {
        !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit())
    })
}

/// Convert a provider's properties to bounded JSON-friendly values. The
/// returned raw string representation is used only by the stable-ID classifier.
pub fn properties_from_gstreamer(
    properties: &gstreamer::StructureRef,
) -> BTreeMap<String, RawProviderProperty> {
    properties
        .iter()
        .map(|(name, value)| {
            (
                name.to_string(),
                RawProviderProperty {
                    value_type: value.type_().name().to_string(),
                    display_value: format!("{value:?}"),
                    string_value: value.get::<String>().ok(),
                },
            )
        })
        .collect()
}

struct LimitedVecWriter {
    bytes: Vec<u8>,
    maximum: usize,
}

impl LimitedVecWriter {
    fn new(maximum: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(maximum.min(64 * 1024)),
            maximum,
        }
    }
}

impl Write for LimitedVecWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let Some(new_len) = self.bytes.len().checked_add(bytes.len()) else {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "JSON size overflow",
            ));
        };
        if new_len > self.maximum {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "camera discovery JSON exceeded its output limit",
            ));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn property(value: &str) -> RawProviderProperty {
        RawProviderProperty {
            value_type: "gchararray".to_owned(),
            display_value: format!("(gchararray) {value:?}"),
            string_value: Some(value.to_owned()),
        }
    }

    fn metadata(properties: Option<BTreeMap<String, RawProviderProperty>>) -> DeviceMetadata {
        DeviceMetadata {
            display_name: Some("Fixture camera".to_owned()),
            device_class: Some("Video/Source".to_owned()),
            provider_properties: properties,
            advertised_caps: Some("video/x-raw,format=RGB".to_owned()),
        }
    }

    fn report(devices: Vec<CameraDevice>) -> CameraDiscoveryReport {
        CameraDiscoveryReport::new(
            HostPlatform {
                os: "linux".to_owned(),
                architecture: "x86_64".to_owned(),
                version: None,
            },
            "GStreamer 1.24.4".to_owned(),
            ProviderStatus::Started,
            devices,
        )
    }

    #[test]
    fn serializes_versioned_device_metadata_and_unavailable_values() {
        let device = CameraDevice::from_metadata("linux", metadata(None));
        let bytes = report(vec![device]).to_bounded_json(4096).unwrap();
        let json: Value = serde_json::from_slice(&bytes).unwrap();

        assert_eq!(json["schema_version"], CAMERA_DISCOVERY_SCHEMA_VERSION);
        assert_eq!(json["platform"]["version"], Value::Null);
        assert_eq!(json["devices"][0]["display_name"], "Fixture camera");
        assert_eq!(json["devices"][0]["provider_properties"], Value::Null);
        assert_eq!(
            json["devices"][0]["advertised_caps"],
            "video/x-raw,format=RGB"
        );
        assert_eq!(
            json["devices"][0]["identifier_classification"],
            "unavailable"
        );
    }

    #[test]
    fn classifies_only_documented_persistent_and_transient_forms() {
        let mac_properties = BTreeMap::from([("avf.unique_id".to_owned(), property("camera-42"))]);
        let mac = CameraDevice::from_metadata("macos", metadata(Some(mac_properties)));
        assert_eq!(mac.identifier_value.as_deref(), Some("camera-42"));
        assert_eq!(mac.identifier_source.as_deref(), Some("avf.unique_id"));
        assert_eq!(
            mac.identifier_classification,
            IdentifierClassification::PlatformPersistent
        );
        let numeric_mac = CameraDevice::from_metadata(
            "macos",
            metadata(Some(BTreeMap::from([(
                "avf.unique_id".to_owned(),
                property("42"),
            )]))),
        );
        assert_eq!(numeric_mac.capture_selector, None);

        let linux_stable = BTreeMap::from([(
            "device.path".to_owned(),
            property("/dev/v4l/by-id/usb-camera"),
        )]);
        let stable = CameraDevice::from_metadata("linux", metadata(Some(linux_stable)));
        assert_eq!(
            stable.identifier_value.as_deref(),
            Some("/dev/v4l/by-id/usb-camera")
        );
        assert_eq!(
            stable.identifier_classification,
            IdentifierClassification::PlatformPersistent
        );

        let linux_transient =
            BTreeMap::from([("device.path".to_owned(), property("/dev/video12"))]);
        let transient = CameraDevice::from_metadata("linux", metadata(Some(linux_transient)));
        assert_eq!(transient.identifier_value.as_deref(), Some("/dev/video12"));
        assert_eq!(
            transient.identifier_classification,
            IdentifierClassification::EnumeratedTransient
        );
        assert_eq!(transient.capture_selector, None);

        let unknown_key = BTreeMap::from([("device.id".to_owned(), property("camera-42"))]);
        let unknown = CameraDevice::from_metadata("linux", metadata(Some(unknown_key)));
        assert_eq!(unknown.identifier_value, None);
        assert_eq!(
            unknown.identifier_classification,
            IdentifierClassification::Unavailable
        );

        assert!(!is_video_node_path("/dev/video12-extra"));
        assert!(!is_video_node_path("/dev/video"));
    }

    #[test]
    fn linux_persistent_ids_require_one_nontraversing_symlink_name() {
        for value in [
            "/dev/v4l/by-id/usb-camera-42",
            "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:2:1.0-video-index0",
        ] {
            assert!(is_persistent_provider_identity(
                "linux",
                "device.path",
                value
            ));
        }

        for value in [
            "/dev/v4l/by-id/",
            "/dev/v4l/by-id/..",
            "/dev/v4l/by-id/../video0",
            "/dev/v4l/by-path/./camera",
            "/dev/v4l/by-id/usb-camera/video0",
            "/dev/v4l/by-id/usb-camera/",
        ] {
            assert!(
                !is_persistent_provider_identity("linux", "device.path", value),
                "accepted invalid persistent identity {value:?}"
            );
            let invalid = CameraDevice::from_metadata(
                "linux",
                metadata(Some(BTreeMap::from([(
                    "device.path".to_owned(),
                    property(value),
                )]))),
            );
            assert_eq!(invalid.capture_selector, None);
            assert_eq!(
                invalid.identifier_classification,
                IdentifierClassification::Unavailable
            );
        }
    }

    #[test]
    fn selection_requires_one_exact_discovery_selector_and_rejects_transient_ids() {
        let mac = CameraDevice::from_metadata(
            "macos",
            metadata(Some(BTreeMap::from([(
                "avf.unique_id".to_owned(),
                property("usb-camera-42"),
            )]))),
        );
        let mac_selector = mac.capture_selector.clone().unwrap();
        assert_eq!(
            select_capture_selector_index(&mac_selector, std::slice::from_ref(&mac)),
            Ok(0)
        );
        assert_eq!(
            select_capture_selector_index(&mac_selector, &[mac.clone(), mac]),
            Err(StableDeviceSelectionError::DuplicateMatches)
        );
        assert_eq!(
            select_capture_selector_index(&mac_selector, &[]),
            Err(StableDeviceSelectionError::NoMatch)
        );

        let transient = CameraDevice::from_metadata(
            "linux",
            metadata(Some(BTreeMap::from([(
                "device.path".to_owned(),
                property("/dev/video0"),
            )]))),
        );
        assert_eq!(
            select_capture_selector_index("/dev/video0", std::slice::from_ref(&transient)),
            Err(StableDeviceSelectionError::UnstableIdentifier)
        );
        assert_eq!(
            select_capture_selector_index("0", &[]),
            Err(StableDeviceSelectionError::UnstableIdentifier)
        );
        assert_eq!(transient.capture_selector, None);
        assert!(!is_capture_selector("/dev/v4l/by-id/usb-camera"));
        assert!(!is_capture_selector("gstreamer-device-v1:0123"));
        assert!(!is_capture_selector(&format!(
            "gstreamer-device-v1:{}",
            "A".repeat(64)
        )));
    }

    #[test]
    fn capture_selector_hash_is_stable_and_separates_platform_and_provider_keys() {
        let first = capture_selector("linux", "device.path", "/dev/v4l/by-id/usb-camera");
        assert_eq!(
            first,
            capture_selector("linux", "device.path", "/dev/v4l/by-id/usb-camera")
        );
        assert_ne!(
            first,
            capture_selector("macos", "device.path", "/dev/v4l/by-id/usb-camera")
        );
        assert_ne!(
            first,
            capture_selector("linux", "other.provider.key", "/dev/v4l/by-id/usb-camera")
        );
        assert!(is_capture_selector(&first));

        let linux = CameraDevice::from_metadata(
            "linux",
            metadata(Some(BTreeMap::from([(
                "device.path".to_owned(),
                property("/dev/v4l/by-id/usb-camera"),
            )]))),
        );
        let selector = linux.capture_selector.clone().unwrap();
        let once = report(vec![linux.clone()]).to_bounded_json(4096).unwrap();
        let twice = report(vec![linux]).to_bounded_json(4096).unwrap();
        assert_eq!(once, twice);
        let json: Value = serde_json::from_slice(&once).unwrap();
        assert_eq!(json["schema_version"], CAMERA_DISCOVERY_SCHEMA_VERSION);
        assert_eq!(json["devices"][0]["capture_selector"], selector);
    }

    #[test]
    fn empty_inventory_is_a_valid_versioned_json_snapshot() {
        let mut empty = report(Vec::new());
        empty.provider_status = ProviderStatus::NoMatchingProvider;
        let bytes = empty.to_bounded_json(512).unwrap();
        let json: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["devices"], Value::Array(Vec::new()));
        assert_eq!(json["provider_status"], "no_matching_provider");
        assert_eq!(json["truncated"], false);
        assert!(bytes.ends_with(b"\n"));
        assert!(bytes.len() <= 512);
    }

    #[test]
    fn output_limit_prunes_metadata_and_reports_loss_without_exceeding_bound() {
        let properties = (0..100)
            .map(|index| {
                (
                    format!("property_{index:03}"),
                    property(&"large-provider-value".repeat(20)),
                )
            })
            .collect();
        let device = CameraDevice::from_metadata("linux", metadata(Some(properties)));
        let bytes = report(vec![device]).to_bounded_json(1024).unwrap();
        let json: Value = serde_json::from_slice(&bytes).unwrap();

        assert!(bytes.len() <= 1024);
        assert_eq!(json["truncated"], true);
        assert!(
            json["devices"][0]["omitted_property_count"]
                .as_u64()
                .unwrap()
                > 0
        );
    }
}
