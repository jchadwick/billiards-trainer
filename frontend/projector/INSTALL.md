# Local Projector Installation

The native projector and its test sender run on the same Mac or Linux machine.
The repository does not provide a remote installer or deployment service.

## Prerequisites

- LÖVE 11.4 or newer (the project is configured for 11.5)
- Python 3 for the local UDP test sender
- Node.js and npm only when building the browser package

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

The existing helper script accepts local run, test, and web-build commands:

```bash
./deploy.sh run
./deploy.sh test 30 0.5
./deploy.sh web
```

Despite its retained filename, `deploy.sh` only operates on the current
machine. It does not install packages, connect to another host, or publish
files.
