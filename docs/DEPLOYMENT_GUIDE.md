# Billiards Trainer Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying the Billiards Trainer system in production environments. It covers everything from system requirements to ongoing maintenance procedures.

## System Requirements

### Hardware Requirements

#### Minimum Requirements
- **CPU**: Quad-core processor (Intel i5 or AMD equivalent)
- **RAM**: 4GB (8GB recommended)
- **Storage**: 50GB available disk space
- **Camera**: USB 3.0 compatible camera with 1080p capability
- **Network**: Ethernet connection recommended

#### Recommended Production Setup
- **CPU**: 8-core processor (Intel i7/i9 or AMD Ryzen 7/9)
- **RAM**: 16GB or higher
- **Storage**: SSD with 100GB+ available space
- **Camera**: High-quality USB 3.0 or GigE camera
- **GPU**: NVIDIA GPU with CUDA support (optional, for acceleration)
- **Network**: Gigabit Ethernet

#### Camera Requirements
- Resolution: Minimum 1920x1080 (Full HD)
- Frame Rate: 30 FPS or higher
- Interface: USB 3.0, USB-C, or GigE Vision
- Lens: Wide-angle lens to capture full table
- Mount: Stable overhead mounting system

### Software Requirements

#### Operating System
- **Linux**: Ubuntu 20.04 LTS or 22.04 LTS (recommended)
- **Alternative**: CentOS 8, RHEL 8, or Debian 11

#### System Dependencies
```bash
# Essential system packages
sudo apt-get update
sudo apt-get install -y \
  python3.8+ \
  python3-pip \
  python3-venv \
  python3-dev \
  build-essential \
  cmake \
  git \
  nginx \
  redis-server \
  curl \
  wget \
  htop \
  systemd

# Computer vision dependencies
sudo apt-get install -y \
  libopencv-dev \
  python3-opencv \
  libv4l-dev \
  v4l-utils \
  guvcview

# Additional libraries
sudo apt-get install -y \
  libblas-dev \
  liblapack-dev \
  libatlas-base-dev \
  gfortran \
  libhdf5-dev \
  pkg-config
```

## Pre-Deployment Setup

### 1. Create System User
```bash
# Create dedicated user for the application
sudo useradd -r -m -s /bin/bash billiards
sudo usermod -aG video billiards  # Camera access
sudo usermod -aG dialout billiards  # Serial port access (if needed)
```

### 2. Create Directory Structure
```bash
# Create application directories
sudo mkdir -p /opt/billiards-trainer
sudo mkdir -p /opt/billiards-trainer/{logs,data,config,backups}
sudo chown -R billiards:billiards /opt/billiards-trainer
```

### 3. Configure Firewall
```bash
# Allow HTTP and HTTPS traffic
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Allow SSH (if not already configured)
sudo ufw allow 22/tcp

# Enable firewall
sudo ufw enable
```

## Deployment Process

### Option 1: Automated Deployment (Recommended)

#### 1. Clone Repository
```bash
git clone https://github.com/your-org/billiards-trainer.git
cd billiards-trainer
```

#### 2. Run Deployment Script
```bash
# Deploy to production
sudo python3 deployment/deploy.py /opt/billiards-trainer --environment production

# Or deploy without backup (first-time deployment)
sudo python3 deployment/deploy.py /opt/billiards-trainer --skip-backup
```

The automated deployment script will:
- Validate system requirements
- Create backups (if existing deployment)
- Deploy application code
- Install dependencies
- Configure services
- Set up nginx
- Start services
- Verify deployment

### Option 2: Manual Deployment

#### 1. Deploy Application Code
```bash
# Switch to application user
sudo su - billiards

# Clone repository
git clone https://github.com/your-org/billiards-trainer.git /tmp/billiards-trainer

# Copy to production directory
cp -r /tmp/billiards-trainer/backend /opt/billiards-trainer/
cp -r /tmp/billiards-trainer/deployment /opt/billiards-trainer/
```

#### 2. Create Virtual Environment
```bash
# Create Python virtual environment
cd /opt/billiards-trainer
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install production packages
pip install uvicorn[standard] gunicorn psutil redis prometheus-client
```

#### 3. Configure Application
```bash
# Create production configuration
mkdir -p /opt/billiards-trainer/config

cat > /opt/billiards-trainer/config/production.json << 'EOF'
{
  "environment": "production",
  "debug": false,
  "api": {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 4
  },
  "logging": {
    "level": "INFO",
    "file": "/opt/billiards-trainer/logs/billiards-trainer.log"
  },
  "monitoring": {
    "enabled": true,
    "prometheus_port": 9090
  },
  "security": {
    "cors_origins": ["http://your-domain.com"],
    "rate_limiting": true
  }
}
EOF
```

#### 4. Set Up Systemd Service
```bash
# Copy service file
sudo cp /opt/billiards-trainer/deployment/systemd/billiards-trainer.service \
  /etc/systemd/system/

# Reload systemd and enable service
sudo systemctl daemon-reload
sudo systemctl enable billiards-trainer
```

#### 5. Configure Nginx
```bash
# Copy nginx configuration
sudo cp /opt/billiards-trainer/deployment/nginx/billiards-trainer.conf \
  /etc/nginx/sites-available/

# Enable site
sudo ln -sf /etc/nginx/sites-available/billiards-trainer.conf \
  /etc/nginx/sites-enabled/

# Test nginx configuration
sudo nginx -t

# Reload nginx
sudo systemctl reload nginx
```

#### 6. Set Up Log Rotation
```bash
# Copy logrotate configuration
sudo cp /opt/billiards-trainer/deployment/logrotate.conf \
  /etc/logrotate.d/billiards-trainer
```

## Service Management

### Starting Services
```bash
# Start all services
sudo systemctl start billiards-trainer
sudo systemctl start nginx

# Check service status
sudo systemctl status billiards-trainer
sudo systemctl status nginx
```

### Stopping Services
```bash
# Stop services
sudo systemctl stop billiards-trainer
sudo systemctl stop nginx
```

### Service Logs
```bash
# View live logs
sudo journalctl -u billiards-trainer -f

# View application logs
tail -f /opt/billiards-trainer/logs/billiards-trainer.log

# View nginx logs
tail -f /var/log/nginx/billiards-trainer-access.log
tail -f /var/log/nginx/billiards-trainer-error.log
```

## Post-Deployment Configuration

### 1. Camera Setup
```bash
# List available cameras
v4l2-ctl --list-devices

# Test camera capture
v4l2-ctl --device=/dev/video0 --all

# Set camera permissions
sudo usermod -aG video billiards
```

### 2. Performance Tuning
```bash
# Optimize system for real-time processing
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
echo 'kernel.sched_rt_runtime_us=-1' | sudo tee -a /etc/sysctl.conf

# Apply changes
sudo sysctl -p
```

### 3. Security Hardening
```bash
# Set proper file permissions
sudo chown -R billiards:billiards /opt/billiards-trainer
sudo chmod -R 755 /opt/billiards-trainer
sudo chmod 644 /opt/billiards-trainer/config/*.json

# Secure log files
sudo chmod 640 /opt/billiards-trainer/logs/*.log
```

## SSL/TLS Setup (Optional)

### Using Let's Encrypt
```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d your-domain.com

# Test automatic renewal
sudo certbot renew --dry-run
```

### Manual Certificate Setup
```bash
# Copy certificates to nginx directory
sudo mkdir -p /etc/nginx/ssl
sudo cp your-certificate.crt /etc/nginx/ssl/
sudo cp your-private-key.key /etc/nginx/ssl/

# Update nginx configuration for SSL
# (Uncomment SSL sections in nginx config)
```

## Monitoring Setup

### 1. System Monitoring
```bash
# Install monitoring tools
sudo apt-get install -y htop iotop nethogs

# Set up system monitoring (optional)
# Configure Prometheus, Grafana, or other monitoring tools
```

### 2. Application Monitoring
The system includes built-in monitoring accessible via:
- Health endpoint: `http://your-domain/health`
- Metrics endpoint: `http://your-domain/api/v1/system/metrics`
- WebSocket monitoring: `ws://your-domain/ws`

## Backup Configuration

### 1. Automated Backups
```bash
# Create backup script
cat > /opt/billiards-trainer/scripts/backup.sh << 'EOF'
#!/bin/bash
cd /opt/billiards-trainer
python3 deployment/backup.py backup --data-only
python3 deployment/backup.py cleanup --keep-count 7
EOF

chmod +x /opt/billiards-trainer/scripts/backup.sh

# Set up cron job for daily backups
echo "0 2 * * * billiards /opt/billiards-trainer/scripts/backup.sh" | sudo crontab -u billiards -
```

### 2. Manual Backup
```bash
# Create full system backup
python3 /opt/billiards-trainer/deployment/backup.py backup

# Create data-only backup
python3 /opt/billiards-trainer/deployment/backup.py backup --data-only

# List available backups
python3 /opt/billiards-trainer/deployment/backup.py list
```

## Troubleshooting

### Common Issues

#### 1. Service Won't Start
```bash
# Check service status
sudo systemctl status billiards-trainer

# View detailed logs
sudo journalctl -u billiards-trainer -n 50

# Check configuration
python3 /opt/billiards-trainer/backend/system_launcher.py --help
```

#### 2. Camera Not Detected
```bash
# List cameras
lsusb | grep -i camera
v4l2-ctl --list-devices

# Check permissions
groups billiards | grep video

# Test camera
guvcview
```

#### 3. High CPU Usage
```bash
# Monitor processes
htop

# Check system resources
free -h
df -h

# Review configuration
less /opt/billiards-trainer/config/production.json
```

#### 4. Network Issues
```bash
# Check port binding
netstat -tlnp | grep 8000

# Test nginx configuration
sudo nginx -t

# Check firewall
sudo ufw status
```

### Performance Optimization

#### 1. System Optimization
```bash
# Disable unnecessary services
sudo systemctl disable bluetooth
sudo systemctl disable cups

# Optimize CPU governor
echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

#### 2. Application Optimization
```bash
# Adjust worker processes in production.json
{
  "api": {
    "workers": 4  // Adjust based on CPU cores
  }
}

# Optimize vision processing
{
  "vision": {
    "target_fps": 30,  // Reduce if needed
    "enable_gpu": true  // If GPU available
  }
}
```

## Maintenance Procedures

### Regular Maintenance Tasks

#### Daily
- Monitor system logs for errors
- Check disk space usage
- Verify service status

#### Weekly
- Review performance metrics
- Check for system updates
- Validate backup integrity

#### Monthly
- Apply security updates
- Clean up old log files
- Review configuration settings

### Update Procedures

#### 1. Application Updates
```bash
# Create backup before update
python3 /opt/billiards-trainer/deployment/backup.py backup

# Stop services
sudo systemctl stop billiards-trainer

# Deploy new version
python3 /opt/billiards-trainer/deployment/deploy.py /opt/billiards-trainer

# Verify deployment
curl http://localhost/health
```

#### 2. System Updates
```bash
# Update system packages
sudo apt-get update
sudo apt-get upgrade

# Reboot if kernel updated
sudo reboot
```

## Disaster Recovery

### 1. System Restore
```bash
# List available backups
python3 /opt/billiards-trainer/deployment/backup.py list

# Restore from backup
python3 /opt/billiards-trainer/deployment/backup.py restore \
  --backup-file /path/to/backup.tar.gz

# Verify restore
sudo systemctl status billiards-trainer
curl http://localhost/health
```

### 2. Database Recovery
```bash
# Restore data only
python3 /opt/billiards-trainer/deployment/backup.py restore \
  --backup-file /path/to/backup.tar.gz --data-only
```

## Security Considerations

### 1. System Security
- Keep system packages updated
- Use strong passwords and SSH keys
- Configure fail2ban for SSH protection
- Regular security audits

### 2. Application Security
- Configure CORS properly
- Implement rate limiting
- Use HTTPS in production
- Monitor access logs

### 3. Network Security
- Configure firewall rules
- Use VPN for remote access
- Monitor network traffic
- Implement intrusion detection

## Support and Documentation

### Log Locations
- Application logs: `/opt/billiards-trainer/logs/`
- System logs: `journalctl -u billiards-trainer`
- Nginx logs: `/var/log/nginx/billiards-trainer*.log`

### Configuration Files
- Application config: `/opt/billiards-trainer/config/production.json`
- Service config: `/etc/systemd/system/billiards-trainer.service`
- Nginx config: `/etc/nginx/sites-available/billiards-trainer.conf`

### Useful Commands
```bash
# System status
sudo systemctl status billiards-trainer
curl http://localhost/health

# Resource monitoring
htop
iotop
df -h

# Log monitoring
tail -f /opt/billiards-trainer/logs/billiards-trainer.log
sudo journalctl -u billiards-trainer -f

# Service management
sudo systemctl restart billiards-trainer
sudo systemctl reload nginx
```

This deployment guide provides comprehensive instructions for successfully deploying and maintaining the Billiards Trainer system in production environments.
