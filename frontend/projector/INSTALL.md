# LÖVE2D Installation Instructions

## On Target System (Ubuntu/Debian)

Run the following commands on the target system to install LÖVE2D:

```bash
sudo apt-get update
sudo apt-get install -y love
```

## Verify Installation

```bash
love --version
```

Should output something like: `LÖVE 11.5 (Mysterious Mysteries)`

## Then Run the Projector

```bash
cd /opt/billiards-trainer/frontend/projector
love .
```

Or use the deployment script from your local machine:

```bash
./deploy.sh install  # Installs LÖVE2D (requires sudo password)
./deploy.sh run      # Runs the projector
```
