# Getting Started with Billiards Trainer

Dead simple setup guide. Clone, run setup, start coding.

## Quick Start (Ubuntu)

```bash
# 1. Clone the repo
git clone https://github.com/jchadwick/billiards-trainer.git
cd billiards-trainer

# 2. Run setup script (installs everything)
./setup.sh

# 3. Activate environment and run
source venv/bin/activate
make run
```

Done. Application running at http://localhost:8000

## Prerequisites

- Ubuntu 20.04+ (or compatible Linux)
- sudo access
- Internet connection

That's it. The setup script handles everything else.

## What Gets Installed

The `setup.sh` script automatically installs:
- Python 3.9+ and pip
- Node.js 20.x and npm
- Docker and Docker Compose
- System libraries (OpenCV, OpenGL, SDL, etc.)
- Python dependencies (FastAPI, OpenCV, NumPy, etc.)
- Node.js dependencies (React, TanStack Router, etc.)
- Creates virtual environment
- Sets up camera/video permissions
- Creates required directories
- Generates .env configuration

## Running the Application

### Option 1: Direct (Development)
```bash
source venv/bin/activate
make run
```

API available at:
- Main API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- WebSocket: ws://localhost:8001

### Option 2: Docker (Production)
```bash
docker-compose up -d
```

Services:
- Frontend: http://localhost:80
- Backend: http://localhost:8000
- Redis: localhost:6379

## Configuration

Edit `.env` file for your setup:

```bash
# Required: Generate secure keys
JWT_SECRET_KEY=your-secret-here
DEFAULT_API_KEY=your-api-key-here

# Generate with:
openssl rand -hex 32

# Camera settings
CAMERA_INDEX=0           # 0=built-in, 1+=external
CAMERA_WIDTH=1920
CAMERA_HEIGHT=1080

# Projector settings
PROJECTOR_DISPLAY_INDEX=1
PROJECTOR_WIDTH=1920
PROJECTOR_HEIGHT=1080
```

## Useful Commands

```bash
# Development
make help              # Show all commands
make run               # Run application
make test              # Run tests
make lint              # Check code quality
make format            # Format code

# Logs
make logs              # View logs
make logs-error        # View error logs
make monitor           # Live log monitoring

# Docker
docker-compose up -d            # Start all services
docker-compose down             # Stop all services
docker-compose logs -f backend  # View backend logs
docker-compose restart backend  # Restart backend

# Verification
curl http://localhost:8000/api/v1/health  # Check health
make test-fast                             # Quick test
```

## Troubleshooting

### "Permission denied" on camera
```bash
sudo usermod -aG video $USER
# Log out and back in
```

### "Cannot connect to Docker daemon"
```bash
sudo systemctl start docker
sudo usermod -aG docker $USER
# Log out and back in
```

### "Module 'backend' not found"
```bash
source venv/bin/activate
pip install -e .
```

### Port already in use
Edit `.env` and change:
```bash
API_PORT=8080
WS_PORT=8081
```

### Re-run setup
```bash
./setup.sh
```

## Project Structure

```
billiards-trainer/
├── backend/              # Python backend
│   ├── api/             # FastAPI app
│   ├── vision/          # Computer vision
│   ├── projector/       # Rendering
│   ├── physics/         # Physics engine
│   └── core/            # Core utilities
├── frontend/web/        # React frontend
├── config/              # Configuration
├── setup.sh             # Setup script
├── Makefile             # Commands
└── .env                 # Environment config
```

## Next Steps

1. Edit `.env` with your settings
2. Read `PLAN.md` for implementation details
3. Run `make test` to verify everything works
4. Start developing!

## Key URLs

- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- WebSocket: ws://localhost:8001
- Frontend (Docker): http://localhost:80
