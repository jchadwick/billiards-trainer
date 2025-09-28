# Billiards Trainer

An augmented reality system that provides real-time assistance for pool/billiards players by detecting the game state through computer vision and projecting trajectory predictions directly onto the pool table. The system uses an overhead camera to capture the table, processes the imagery to detect balls, cue stick, and table boundaries, calculates shot trajectories based on physics simulation, and displays visual guidance through a calibrated projector.

### Primary Goals

1. **Enhance Player Experience**: Provide intuitive visual feedback to help players improve their aim and understand ball physics
2. **Real-time Performance**: Process and display trajectory updates with less than 50ms latency
3. **Zero Installation Access**: Enable users to access the system interface from any device without software installation
4. **Professional Accuracy**: Achieve trajectory prediction accuracy suitable for training and entertainment venues

## System Overview

### Core Functionality

The system performs continuous real-time analysis of a pool table through the following pipeline:

1. **Image Capture**: Overhead camera captures the pool table at 30+ FPS
2. **Game State Detection**: Computer vision identifies all balls, cue stick position, and table boundaries
3. **Physics Calculation**: Predicts ball trajectories based on cue angle and estimated force
4. **Augmented Display**: Projects trajectory lines and collision predictions onto the physical table
5. **User Interface**: Web-based control panel for system configuration and monitoring

### Key Features

- **Automatic Calibration**: Self-calibrating system that adapts to different table sizes and lighting conditions
- **Multi-ball Trajectory**: Calculates collision chains showing how balls will interact
- **Difficulty Levels**: Adjustable assistance levels from beginner to expert
- **Remote Access**: Web-based interface accessible from any device
- **Spectator Mode**: Stream augmented view to external displays or devices
- **Shot History**: Record and replay shot sequences for training analysis

## Technical Requirements

### Hardware Requirements

- **Camera**: Minimum 1080p resolution, 30 FPS, wide-angle lens for full table coverage
- **Computer**: Modern CPU (Intel i5/AMD Ryzen 5 or better), 8GB RAM minimum
- **Projector**: 1080p resolution minimum, 3000+ lumens for daylight visibility (optional for AR display)
- **Network**: Ethernet or Wi-Fi for multi-device connectivity

### Software Stack

- **Backend**: Python 3.12+ with FastAPI framework
- **Computer Vision**: OpenCV 4.8+ for image processing
- **Frontend**: Modern web application (React) with WebSocket support
- **Deployment**: Can be run natively or in Docker containers for easy installation

## Project Structure

```
/
├── backend/
│   ├── api/              # FastAPI routes and WebSocket handlers
│   ├── core/             # Detection, tracking, physics calculations
│   ├── vision/           # OpenCV processing modules
│   ├── projector/        # Projection client module
│   └── config/           # Configuration management
└── frontend/             # Web application
```

## Architecture

See @ARCHITECTURE.md for detailed architecture diagrams and explanations.

## Project Scope

### In Scope

- Complete pool table game state detection
- Real-time trajectory calculation and display
- Web-based configuration and monitoring interface
- Projector calibration and geometric correction
- Multi-client support for remote viewing
- Configuration persistence and management
- Docker-based deployment solution

### Out of Scope

- Automatic shot execution (robotic cue)
- Score keeping and game rule enforcement
- Player identification and tracking
- 3D visualization (2D projection only)
