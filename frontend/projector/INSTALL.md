# Local Projector Installation

The native projector runs locally on Mac or Linux. The repository does not
provide a remote installer or deployment service.

## Prerequisites

- LÖVE 11.4 or newer (the project is configured for 11.5)
Install LÖVE through the operating system's package manager or application
installer. For example, on macOS with Homebrew:

```bash
brew install --cask love
```

On Linux, install the `love` package from the distribution's configured package
repository. Package installation may require the local machine's administrator
rights; the application itself has no user accounts or credential settings.

Verify the installation:

```bash
love --version
```

## Run locally

From the repository root:

```bash
cd frontend/projector
love .
```

To open a windowed development instance:

```bash
love . conf_windowed.lua
```

The legacy-named helper script runs the local projector:

```bash
./deploy.sh run
```

Despite its retained filename, `deploy.sh` only operates on the current
machine. It does not install packages, connect to another host, or publish
files.
