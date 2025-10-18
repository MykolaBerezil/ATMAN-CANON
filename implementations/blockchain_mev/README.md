# ATMAN-CANON Blockchain MEV Implementation

This implementation demonstrates the ATMAN-CANON framework applied to blockchain Maximum Extractable Value (MEV) analysis and trading systems.

## Overview

This implementation uses the core ATMAN-CANON framework to:
- Analyze blockchain transaction patterns
- Detect MEV opportunities
- Implement safe trading strategies
- Monitor system performance and safety

## Components

### Core Application
- `app.py` - Main Flask web application with monitoring dashboard
- Production-ready server with real-time metrics

### Legacy Core Modules (Deprecated)
- `core/` - Legacy core modules (use `atman_core` package instead)
  - `evidence_calculator.py` - Evidence quality assessment
  - `hypothesis_generator.py` - Hypothesis generation system
  - `rbmk.py` - Recursive Bayesian Meta-Knowledge framework
  - `transfer_learning.py` - Cross-domain knowledge transfer

### System Configuration
- `scripts/install.sh` - Installation script for production deployment
- `systemd/atman.service` - Systemd service configuration

## Migration Notice

⚠️ **Important**: This implementation is being migrated to use the new `atman_core` framework package.

### Old Import Pattern (Deprecated)
```python
from core.rbmk import RBMKReasoner
from core.evidence_calculator import EvidenceCalculator
```

### New Import Pattern (Recommended)
```python
from atman_core import RBMKReasoner, EvidenceCalculator
from atman_core.utils import RenormalizationSafety, KappaBlockLogic
```

## Installation

### Using the Framework Package
```bash
# Install the core framework
pip install -e ../../

# Install blockchain-specific dependencies
pip install web3 eth-account
```

### Legacy Installation
```bash
# Run the installation script
./scripts/install.sh
```

## Usage

### Starting the Server
```bash
# Using systemd (recommended for production)
sudo systemctl start atman.service

# Or run directly
python app.py
```

### Accessing the Dashboard
- Main dashboard: http://localhost:8080/
- Health check: http://localhost:8080/health
- API status: http://localhost:8080/api/v5/status

## Configuration

The application uses environment variables from `/opt/atman/.env`:
- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_URL` - Redis connection string
- `ATMAN_MASTER_PASSPHRASE` - Encryption passphrase for sensitive data

## Safety Features

This implementation includes:
- **Renormalization Safety**: Prevents runaway trading algorithms
- **κ-Block Logic**: Ensures logical consistency in trading decisions
- **Invariant Detection**: Monitors for anomalous market conditions
- **Emergency Stops**: Automatic shutdown on dangerous conditions

## Monitoring

Real-time monitoring includes:
- System resource usage (CPU, memory, disk)
- Trading algorithm performance
- Safety mechanism status
- Database and cache health

## Development

### Running Tests
```bash
# Test the core framework
python -m pytest ../../tests/

# Test blockchain-specific functionality
python -m pytest tests/ # (if tests directory exists)
```

### Code Style
```bash
black app.py core/
flake8 app.py core/
```

## Production Deployment

This implementation is designed for production deployment on:
- Ubuntu 22.04 LTS
- PostgreSQL database
- Redis cache
- Nginx reverse proxy (optional)

See the main repository documentation for complete deployment instructions.

## License

This implementation is part of the ATMAN-CANON framework and is licensed under:
- **Academic/Research Use**: GNU Affero General Public License v3.0
- **Commercial Use**: Separate commercial license required

## Support

For issues specific to this blockchain implementation, please:
1. Check the main framework documentation
2. Review the safety mechanisms documentation
3. Open an issue in the main repository with the `blockchain-mev` label
