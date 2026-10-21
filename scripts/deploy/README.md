# Billiards Trainer - Local Distribution

This is a locally built distribution of the Billiards Trainer system. The
project does not provide remote deployment or user authentication.

## Quick Start

1. Copy this entire directory to the machine that will run the application
2. Run the setup script (first time only):
   ```bash
   ./setup_config.sh
   ```
3. Start the system:
   ```bash
   ./run.sh
   ```

## Features

- **Local restart**: The startup script can restart the local process when files change
- **Built-in configuration wizard**: Interactive setup for first-time deployment
- **Self-contained**: Includes all necessary components (backend, frontend, projector)
- **Production-optimized**: Pre-built frontend and optimized for performance

## Deployment Workflow

### Initial Local Install
```bash
# On the machine that will run the application
cp -R /path/to/dist /opt/billiards-trainer
cd /opt/billiards-trainer
./run.sh
```

### Local Update
```bash
# Rebuild the distribution in the source checkout
make deploy

# Copy it locally to the installation directory, then restart the app
cp -R dist/. /opt/billiards-trainer/
cd /opt/billiards-trainer
./run.sh
```

The project does not copy files over SSH, sign into a container registry, or
configure remote deployment credentials. Move the bundle using a local file
transfer method if the application runs on another machine.

## Directory Structure

```
dist/
├── backend/          # Python backend modules
├── frontend/         # Built web frontend (static files)
├── config/           # Configuration templates
├── data/            # Runtime data directory
├── logs/            # Application logs
├── run.sh           # Main startup script with auto-restart
├── setup_config.sh  # Interactive configuration wizard
├── requirements.txt # Python dependencies
└── .env.example     # Configuration template
```

## Requirements

- Python 3.8+
- OpenCV dependencies (installed automatically)
- Camera and projector hardware (for full functionality)

## Configuration

The configuration wizard will guide you through:
- Local API port and device settings (the API has no login)
- API port configuration
- Camera settings
- Projector settings
- Environment mode

Configuration is stored in `.env` file.

## Monitoring

- Application logs: `logs/` directory
- API endpoint: `http://localhost:8000/docs` (Swagger UI)
- Health check: `http://localhost:8000/health`

## Troubleshooting

### Port already in use
```bash
# Find and kill the process
lsof -ti:8000 | xargs kill -9
./run.sh
```

### Dependencies not installing
```bash
# Manually create virtual environment and install
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Auto-restart not working
Make sure watchdog is installed:
```bash
source .venv/bin/activate
pip install watchdog
```

## Support

See the main repository for documentation and support.
