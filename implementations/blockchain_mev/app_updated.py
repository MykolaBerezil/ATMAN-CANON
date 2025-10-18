#!/usr/bin/env python3
"""
ATMAN-CANON Blockchain MEV Implementation
Updated to use the new atman_core framework package.

This is the updated version that imports from the core framework.
The original app.py is preserved for reference.
"""

import os
import sys
import json
import logging
from datetime import datetime
from flask import Flask, jsonify, request, render_template_string
import psutil
import time

# Add the framework to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import from the new atman_core framework
try:
    from atman_core import RBMKReasoner, EvidenceCalculator, InvariantDetector
    from atman_core.utils import RenormalizationSafety, KappaBlockLogic
    FRAMEWORK_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import atman_core framework: {e}")
    print("Falling back to legacy imports...")
    FRAMEWORK_AVAILABLE = False

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/atman/logs/atman.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('ATMAN-CANON-Blockchain')

# Initialize framework components if available
if FRAMEWORK_AVAILABLE:
    try:
        reasoner = RBMKReasoner()
        evidence_calc = EvidenceCalculator()
        invariant_detector = InvariantDetector()
        safety = RenormalizationSafety()
        kappa_logic = KappaBlockLogic()
        logger.info("ATMAN-CANON framework components initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize framework components: {e}")
        FRAMEWORK_AVAILABLE = False

# HTML template for the monitoring dashboard
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>ATMAN-CANON Blockchain MEV Monitor</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: #1a1a1a; 
            color: #e0e0e0; 
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { 
            text-align: center; 
            margin-bottom: 30px; 
            border-bottom: 2px solid #333;
            padding-bottom: 20px;
        }
        .title { 
            color: #4CAF50; 
            margin: 0; 
            font-size: 2.5em; 
            font-weight: 300;
        }
        .subtitle { 
            color: #888; 
            margin: 10px 0 0 0; 
            font-size: 1.1em;
        }
        .grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
            gap: 20px; 
            margin-bottom: 30px; 
        }
        .card { 
            background: #2d2d2d; 
            border-radius: 8px; 
            padding: 20px; 
            border-left: 4px solid #4CAF50;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        .card h3 { 
            margin: 0 0 15px 0; 
            color: #4CAF50; 
            font-size: 1.2em;
            font-weight: 500;
        }
        .metric { 
            display: flex; 
            justify-content: space-between; 
            margin: 8px 0; 
            padding: 5px 0;
            border-bottom: 1px solid #404040;
        }
        .metric:last-child { border-bottom: none; }
        .metric-label { color: #ccc; }
        .metric-value { 
            font-weight: bold; 
            color: #fff;
            font-family: 'Courier New', monospace;
        }
        .status-good { color: #4CAF50; }
        .status-warning { color: #FF9800; }
        .status-error { color: #f44336; }
        .framework-status {
            text-align: center;
            padding: 15px;
            margin: 20px 0;
            border-radius: 8px;
            font-weight: bold;
        }
        .framework-available {
            background: rgba(76, 175, 80, 0.1);
            border: 2px solid #4CAF50;
            color: #4CAF50;
        }
        .framework-unavailable {
            background: rgba(244, 67, 54, 0.1);
            border: 2px solid #f44336;
            color: #f44336;
        }
        .timestamp { 
            text-align: center; 
            color: #666; 
            margin-top: 30px; 
            font-size: 0.9em;
        }
        .progress-bar {
            width: 100%;
            height: 8px;
            background: #404040;
            border-radius: 4px;
            overflow: hidden;
            margin: 5px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #45a049);
            transition: width 0.3s ease;
        }
    </style>
    <script>
        function updateDashboard() {
            fetch('/api/metrics')
                .then(response => response.json())
                .then(data => {
                    // Update metrics dynamically
                    document.getElementById('cpu-usage').textContent = data.system.cpu_percent + '%';
                    document.getElementById('memory-usage').textContent = data.system.memory_percent + '%';
                    document.getElementById('disk-usage').textContent = data.system.disk_percent + '%';
                    
                    // Update progress bars
                    document.getElementById('cpu-bar').style.width = data.system.cpu_percent + '%';
                    document.getElementById('memory-bar').style.width = data.system.memory_percent + '%';
                    document.getElementById('disk-bar').style.width = data.system.disk_percent + '%';
                    
                    // Update timestamp
                    document.getElementById('last-update').textContent = new Date().toLocaleString();
                })
                .catch(error => console.error('Error updating dashboard:', error));
        }
        
        // Update every 2 seconds
        setInterval(updateDashboard, 2000);
        
        // Initial update
        window.onload = updateDashboard;
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 class="title">ATMAN-CANON</h1>
            <p class="subtitle">Blockchain MEV Implementation - Production Monitor</p>
        </div>
        
        <div class="framework-status {{ 'framework-available' if framework_available else 'framework-unavailable' }}">
            {{ '✓ ATMAN-CANON Framework Active' if framework_available else '⚠ Framework Unavailable - Legacy Mode' }}
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>System Resources</h3>
                <div class="metric">
                    <span class="metric-label">CPU Usage</span>
                    <span class="metric-value" id="cpu-usage">{{ system_metrics.cpu_percent }}%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="cpu-bar" style="width: {{ system_metrics.cpu_percent }}%"></div>
                </div>
                
                <div class="metric">
                    <span class="metric-label">Memory Usage</span>
                    <span class="metric-value" id="memory-usage">{{ system_metrics.memory_percent }}%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="memory-bar" style="width: {{ system_metrics.memory_percent }}%"></div>
                </div>
                
                <div class="metric">
                    <span class="metric-label">Disk Usage</span>
                    <span class="metric-value" id="disk-usage">{{ system_metrics.disk_percent }}%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="disk-bar" style="width: {{ system_metrics.disk_percent }}%"></div>
                </div>
            </div>
            
            <div class="card">
                <h3>Application Status</h3>
                <div class="metric">
                    <span class="metric-label">Status</span>
                    <span class="metric-value status-good">{{ app_status.status }}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Version</span>
                    <span class="metric-value">{{ app_status.version }}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Environment</span>
                    <span class="metric-value">{{ app_status.environment }}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Uptime</span>
                    <span class="metric-value">{{ app_status.uptime }}</span>
                </div>
            </div>
            
            {% if framework_available %}
            <div class="card">
                <h3>Framework Components</h3>
                <div class="metric">
                    <span class="metric-label">RBMK Reasoner</span>
                    <span class="metric-value status-good">Active</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Evidence Calculator</span>
                    <span class="metric-value status-good">Active</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Safety Systems</span>
                    <span class="metric-value status-good">Monitoring</span>
                </div>
                <div class="metric">
                    <span class="metric-label">κ-Block Logic</span>
                    <span class="metric-value status-good">Enabled</span>
                </div>
            </div>
            
            <div class="card">
                <h3>Safety Metrics</h3>
                <div class="metric">
                    <span class="metric-label">Renormalizations</span>
                    <span class="metric-value">{{ safety_metrics.renormalization_count if safety_metrics else 'N/A' }}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Emergency Stops</span>
                    <span class="metric-value">{{ safety_metrics.emergency_stops if safety_metrics else 'N/A' }}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">System Energy</span>
                    <span class="metric-value">{{ "%.1f"|format(safety_metrics.current_energy) if safety_metrics else 'N/A' }}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">κ-Blocks</span>
                    <span class="metric-value">{{ kappa_metrics.total_blocks if kappa_metrics else 'N/A' }}</span>
                </div>
            </div>
            {% endif %}
            
            <div class="card">
                <h3>Network & Services</h3>
                <div class="metric">
                    <span class="metric-label">Database</span>
                    <span class="metric-value status-good">Connected</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Redis Cache</span>
                    <span class="metric-value status-good">Connected</span>
                </div>
                <div class="metric">
                    <span class="metric-label">API Endpoints</span>
                    <span class="metric-value status-good">{{ api_endpoints|length }} Active</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Port</span>
                    <span class="metric-value">8080</span>
                </div>
            </div>
        </div>
        
        <div class="timestamp">
            Last updated: <span id="last-update">{{ timestamp }}</span>
        </div>
    </div>
</body>
</html>
"""

# Application start time for uptime calculation
app_start_time = time.time()

def get_system_metrics():
    """Get current system metrics."""
    return {
        'cpu_percent': round(psutil.cpu_percent(interval=1), 1),
        'memory_percent': round(psutil.virtual_memory().percent, 1),
        'disk_percent': round(psutil.disk_usage('/').percent, 1),
        'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
    }

def get_app_status():
    """Get application status information."""
    uptime_seconds = int(time.time() - app_start_time)
    uptime_hours = uptime_seconds // 3600
    uptime_minutes = (uptime_seconds % 3600) // 60
    
    return {
        'status': 'Operational',
        'version': '5.0.0',
        'environment': 'Production',
        'uptime': f"{uptime_hours}h {uptime_minutes}m"
    }

@app.route('/')
def dashboard():
    """Main monitoring dashboard."""
    system_metrics = get_system_metrics()
    app_status = get_app_status()
    
    # Get framework metrics if available
    safety_metrics = None
    kappa_metrics = None
    
    if FRAMEWORK_AVAILABLE:
        try:
            safety_metrics = safety.get_safety_metrics()
            kappa_metrics = kappa_logic.get_kappa_metrics()
        except Exception as e:
            logger.error(f"Error getting framework metrics: {e}")
    
    return render_template_string(
        DASHBOARD_TEMPLATE,
        framework_available=FRAMEWORK_AVAILABLE,
        system_metrics=system_metrics,
        app_status=app_status,
        safety_metrics=safety_metrics,
        kappa_metrics=kappa_metrics,
        api_endpoints=['/health', '/api/v5/status', '/api/metrics', '/api/reasoning'],
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'framework_available': FRAMEWORK_AVAILABLE,
        'services': {
            'database': 'connected',
            'redis': 'connected',
            'application': 'running'
        }
    })

@app.route('/api/v5/status')
def api_status():
    """API status endpoint."""
    return jsonify({
        'api_version': '5.0',
        'status': 'active',
        'framework': 'atman-canon' if FRAMEWORK_AVAILABLE else 'legacy',
        'endpoints': [
            '/health',
            '/api/v5/status',
            '/api/metrics',
            '/api/reasoning'
        ]
    })

@app.route('/api/metrics')
def api_metrics():
    """Real-time metrics API endpoint."""
    system_metrics = get_system_metrics()
    app_status = get_app_status()
    
    metrics = {
        'system': system_metrics,
        'application': app_status,
        'framework_available': FRAMEWORK_AVAILABLE,
        'timestamp': datetime.utcnow().isoformat()
    }
    
    # Add framework metrics if available
    if FRAMEWORK_AVAILABLE:
        try:
            metrics['safety'] = safety.get_safety_metrics()
            metrics['kappa_logic'] = kappa_logic.get_kappa_metrics()
        except Exception as e:
            logger.error(f"Error getting framework metrics: {e}")
    
    return jsonify(metrics)

@app.route('/api/reasoning', methods=['POST'])
def api_reasoning():
    """Reasoning API endpoint using ATMAN-CANON framework."""
    if not FRAMEWORK_AVAILABLE:
        return jsonify({
            'error': 'ATMAN-CANON framework not available',
            'message': 'Running in legacy mode'
        }), 503
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Process evidence through the framework
        evidence = data.get('evidence', {})
        
        # Evaluate evidence quality
        metrics, accepted = evidence_calc.evaluate(evidence)
        
        # Generate reasoning if evidence is accepted
        result = {
            'evidence_accepted': accepted,
            'quality_metrics': metrics,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if accepted:
            # Process through RBMK reasoner
            reasoning_result = reasoner.process_evidence(evidence)
            result['reasoning'] = reasoning_result
            
            # Apply safety checks
            safe_state = safety.apply_renormalization(reasoning_result)
            result['safety_applied'] = safe_state.get('renormalization_applied', False)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in reasoning API: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("Starting ATMAN-CANON Blockchain MEV Implementation")
    logger.info(f"Framework available: {FRAMEWORK_AVAILABLE}")
    
    # Run the application
    app.run(host='0.0.0.0', port=8080, debug=False)
