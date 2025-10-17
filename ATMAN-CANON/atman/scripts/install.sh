#!/bin/bash
# ATMAN-CANON v5 Installation Script
# Installs ATMAN-CANON Fractal Consciousness Framework

set -e

echo "Installing ATMAN-CANON v5 - Fractal Consciousness Framework"
echo "============================================================"

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "This script should not be run as root for security reasons."
   echo "Please run as a regular user with sudo privileges."
   exit 1
fi

# Update system packages
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
echo "Installing system dependencies..."
sudo apt install -y \
    python3-pip python3-venv python3-dev \
    build-essential libssl-dev libffi-dev \
    postgresql postgresql-contrib redis-server \
    nginx supervisor fail2ban ufw \
    git jq curl wget htop \
    libpq-dev python3-psycopg2

# Create ATMAN user and directories
echo "Setting up ATMAN user and directories..."
sudo useradd -r -m -d /opt/atman -s /bin/bash atman 2>/dev/null || echo "User atman already exists"

# Create directory structure
sudo mkdir -p /opt/atman/{atman,atman_live,logs,keystore,vector_store/chronicle,backup,scripts,tests,systemd,data/historical}
sudo mkdir -p /opt/atman/atman/{core,ledger,governance,epic,murmur,adapters/exec,adapters/treasury,wallet,llm,monitoring,security}

# Set ownership
sudo chown -R atman:atman /opt/atman

# Create Python virtual environment
echo "Creating Python virtual environment..."
sudo python3 -m venv /opt/atman/venv
sudo chown -R atman:atman /opt/atman/venv

# Install Python dependencies
echo "Installing Python dependencies..."
sudo -u atman /opt/atman/venv/bin/pip install --upgrade pip setuptools wheel

# Core dependencies
sudo -u atman /opt/atman/venv/bin/pip install \
    flask==3.0.0 \
    gunicorn==21.2.0 \
    psutil==5.9.6 \
    numpy==1.26.4 \
    scipy==1.11.4 \
    pandas==2.2.2 \
    requests==2.32.3 \
    psycopg2-binary==2.9.9 \
    redis==5.0.1 \
    python-dotenv==1.0.1 \
    colorlog==6.8.0 \
    jsonschema==4.21.1

# Configure PostgreSQL
echo "Configuring PostgreSQL..."
sudo systemctl enable postgresql
sudo systemctl start postgresql

# Create ATMAN database and user
sudo -u postgres psql -c "CREATE DATABASE atman_db;" 2>/dev/null || echo "Database already exists"
sudo -u postgres psql -c "CREATE USER atman WITH ENCRYPTED PASSWORD 'AtmanSecure2024!';" 2>/dev/null || echo "User already exists"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE atman_db TO atman;"

# Optimize PostgreSQL for production
sudo -u postgres psql -c "ALTER SYSTEM SET shared_buffers = '2GB';"
sudo -u postgres psql -c "ALTER SYSTEM SET effective_cache_size = '6GB';"
sudo -u postgres psql -c "ALTER SYSTEM SET work_mem = '16MB';"
sudo -u postgres psql -c "ALTER SYSTEM SET maintenance_work_mem = '256MB';"
sudo systemctl restart postgresql

# Configure Redis
echo "Configuring Redis..."
sudo systemctl enable redis-server
sudo systemctl start redis-server

# Optimize Redis
sudo sed -i 's/^# maxmemory .*/maxmemory 1gb/' /etc/redis/redis.conf
sudo sed -i 's/^# maxmemory-policy .*/maxmemory-policy allkeys-lru/' /etc/redis/redis.conf
sudo systemctl restart redis-server

# Copy ATMAN-CANON files
echo "Installing ATMAN-CANON files..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATMAN_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

sudo cp -r "$ATMAN_ROOT/atman/core" /opt/atman/atman/
sudo cp -r "$ATMAN_ROOT/atman/configs" /opt/atman/atman/
sudo cp -r "$ATMAN_ROOT/atman/scripts" /opt/atman/atman/
sudo cp "$ATMAN_ROOT/atman/app.py" /opt/atman/atman/
sudo chown -R atman:atman /opt/atman/atman

# Create configuration files
echo "Creating configuration files..."
sudo -u atman tee /opt/atman/config.json << 'EOF'
{
  "version": "5.0.0",
  "environment": "production",
  "database": {
    "host": "localhost",
    "port": 5432,
    "name": "atman_db",
    "user": "atman",
    "password": "AtmanSecure2024!"
  },
  "redis": {
    "host": "localhost",
    "port": 6379,
    "db": 0
  },
  "server": {
    "host": "0.0.0.0",
    "port": 8080,
    "debug": false
  },
  "monitoring": {
    "metrics_port": 9090,
    "enabled": true
  }
}
EOF

# Create environment file
sudo -u atman tee /opt/atman/.env << 'EOF'
ATMAN_VERSION=5.0.0
ATMAN_ENV=production
DATABASE_URL=postgresql://atman:AtmanSecure2024!@localhost:5432/atman_db
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=atman_secret_key_2024
EOF

# Create systemd service
echo "Creating systemd service..."
sudo tee /etc/systemd/system/atman.service << 'EOF'
[Unit]
Description=ATMAN-CANON v5 Production Server
After=network.target postgresql.service redis-server.service
Requires=postgresql.service redis-server.service

[Service]
Type=notify
User=atman
Group=atman
WorkingDirectory=/opt/atman/atman
Environment=PATH=/opt/atman/venv/bin
EnvironmentFile=/opt/atman/.env
ExecStart=/opt/atman/venv/bin/gunicorn --bind 0.0.0.0:8080 --workers 2 --timeout 30 --keep-alive 2 --max-requests 1000 --max-requests-jitter 100 app:app
ExecReload=/bin/kill -s HUP $MAINPID
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable atman.service
sudo systemctl start atman.service

# Configure firewall
echo "Configuring firewall..."
sudo ufw allow 8080/tcp comment "ATMAN-CANON v5 API"
sudo ufw allow 9090/tcp comment "ATMAN-CANON v5 Metrics"

echo ""
echo "ATMAN-CANON v5 Installation Complete!"
echo "====================================="
echo "✅ Database: PostgreSQL configured and optimized"
echo "✅ Cache: Redis configured and optimized"
echo "✅ Application: ATMAN-CANON v5 installed and running"
echo "✅ Service: Enabled for automatic startup"
echo ""
echo "Access your ATMAN-CANON system at:"
echo "  Main API: http://localhost:8080/"
echo "  Dashboard: http://localhost:8080/dashboard"
echo "  Health Check: http://localhost:8080/health"
echo ""
echo "Service Management:"
echo "  Status: sudo systemctl status atman.service"
echo "  Logs: sudo journalctl -u atman.service -f"
echo "  Restart: sudo systemctl restart atman.service"
echo ""
echo "🚀 ATMAN-CANON v5 is ready for production use!"
