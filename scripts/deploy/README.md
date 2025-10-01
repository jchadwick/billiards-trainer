# Billiards Trainer - Production Distribution

This is a production-ready distribution of the Billiards Trainer system.

## Quick Start

1. Copy this entire directory to your target system
2. Run the setup script (first time only):
   ```bash
   ./setup_config.sh
   ```
3. Start the system:
   ```bash
   ./run.sh
   ```

## Features

- **Auto-restart**: The system automatically restarts when files change (via rsync)
- **Built-in configuration wizard**: Interactive setup for first-time deployment
- **Self-contained**: Includes all necessary components (backend, frontend, projector)
- **Production-optimized**: Pre-built frontend and optimized for performance

## Deployment Workflow

### Initial Deployment
```bash
# On target system
rsync -av --delete /path/to/dist/ /opt/billiards-trainer/
cd /opt/billiards-trainer
./run.sh
```

### Update Deployment
```bash
# On source system: rebuild distribution
make deploy

# On target system: sync and the system will auto-restart
rsync -av --delete /path/to/dist/ /opt/billiards-trainer/
```

The system will automatically detect file changes and restart.

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
- Security credentials (auto-generated)
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
