#!/usr/bin/env python3
"""
ATMAN-CANON v5 Production Server
Main application integrating all ATMAN-CANON components
"""

import os
import sys
import json
import logging
import psutil
from datetime import datetime
from flask import Flask, jsonify, request, render_template_string
from typing import Dict, Any, List

# Add ATMAN to Python path
sys.path.insert(0, '/opt/atman')
sys.path.insert(0, '/opt/atman/atman')

# Import ATMAN-CANON core components
try:
    from core.evidence_calculator import EvidenceCalculator
    from core.hypothesis_generator import HypothesisGenerator
    from core.transfer_learning import TransferLearningEngine
    from core.rbmk import RBMKFramework
except ImportError as e:
    print(f"Warning: Could not import ATMAN core components: {e}")
    # Create placeholder classes for development
    class EvidenceCalculator:
        def evaluate_evidence(self, evidence): return {'overall_confidence': 0.8, 'accepted': True}
        def get_evidence_summary(self): return {'total_evidence': 0, 'accepted_count': 0}
    
    class HypothesisGenerator:
        def generate_hypotheses(self, evidence_set, domain_context=None): return []
        def get_generation_summary(self): return {'total_generations': 0, 'total_hypotheses': 0}
    
    class TransferLearningEngine:
        def learn_from_domain(self, domain_name, knowledge_items): return {'domain': domain_name}
        def transfer_knowledge(self, source, target, context): return {'transferred_items': []}
        def get_transfer_summary(self): return {'total_transfers': 0, 'domains_involved': 0}
    
    class RBMKFramework:
        def process_knowledge_item(self, knowledge_item, context=None): return {'final_confidence': 0.7}
        def get_rbmk_summary(self): return {'total_processed': 0, 'average_confidence': 0.0}

# Initialize Flask app
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

logger = logging.getLogger('ATMAN-CANON-v5')

# Initialize ATMAN-CANON components
evidence_calculator = EvidenceCalculator(confidence_threshold=0.7)
hypothesis_generator = HypothesisGenerator(creativity_factor=0.7, max_hypotheses=10)
transfer_engine = TransferLearningEngine(similarity_threshold=0.6, max_transfer_depth=3)
rbmk_framework = RBMKFramework(max_recursion_depth=5, uncertainty_threshold=0.1)

# System metrics storage
system_metrics_history = []
MAX_METRICS_HISTORY = 100

@app.route('/')
def home():
    """Main status endpoint"""
    return jsonify({
        'name': 'ATMAN-CANON v5',
        'description': 'Fractal Consciousness Framework for Emergent Intelligence',
        'version': '5.0.0',
        'status': 'operational',
        'timestamp': datetime.utcnow().isoformat(),
        'environment': 'production',
        'components': {
            'evidence_calculator': 'active',
            'hypothesis_generator': 'active', 
            'transfer_learning': 'active',
            'rbmk_framework': 'active'
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    try:
        # Check system resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Check component health
        evidence_summary = evidence_calculator.get_evidence_summary()
        hypothesis_summary = hypothesis_generator.get_generation_summary()
        transfer_summary = transfer_engine.get_transfer_summary()
        rbmk_summary = rbmk_framework.get_rbmk_summary()
        
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'system_resources': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': (disk.used / disk.total) * 100,
                'available_memory_gb': memory.available / (1024**3)
            },
            'component_status': {
                'evidence_calculator': {
                    'status': 'healthy',
                    'total_evidence': evidence_summary.get('total_evidence', 0),
                    'acceptance_rate': evidence_summary.get('acceptance_rate', 0.0)
                },
                'hypothesis_generator': {
                    'status': 'healthy',
                    'total_hypotheses': hypothesis_summary.get('total_hypotheses', 0),
                    'average_quality': hypothesis_summary.get('average_quality', 0.0)
                },
                'transfer_learning': {
                    'status': 'healthy',
                    'total_transfers': transfer_summary.get('total_transfers', 0),
                    'domains_involved': transfer_summary.get('domains_involved', 0)
                },
                'rbmk_framework': {
                    'status': 'healthy',
                    'total_processed': rbmk_summary.get('total_processed', 0),
                    'average_confidence': rbmk_summary.get('average_confidence', 0.0)
                }
            }
        }
        
        # Determine overall health
        if cpu_percent > 90 or memory.percent > 90:
            health_status['status'] = 'degraded'
            health_status['warnings'] = ['High resource utilization']
        
        return jsonify(health_status)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/api/v5/status')
def api_status():
    """API status and capabilities"""
    return jsonify({
        'api_version': '5.0',
        'status': 'active',
        'framework': 'ATMAN-CANON',
        'capabilities': [
            'evidence_evaluation',
            'hypothesis_generation', 
            'transfer_learning',
            'recursive_bayesian_reasoning',
            'invariant_detection',
            'uncertainty_quantification'
        ],
        'endpoints': {
            'core': [
                '/api/v5/evidence/evaluate',
                '/api/v5/hypotheses/generate',
                '/api/v5/transfer/learn',
                '/api/v5/transfer/apply',
                '/api/v5/rbmk/process'
            ],
            'monitoring': [
                '/api/v5/metrics',
                '/api/v5/metrics/system',
                '/dashboard'
            ],
            'system': [
                '/health',
                '/api/v5/status'
            ]
        }
    })

@app.route('/api/v5/metrics')
def get_metrics():
    """Get comprehensive system metrics"""
    try:
        # Collect current metrics
        current_metrics = collect_system_metrics()
        
        # Store in history
        system_metrics_history.append(current_metrics)
        if len(system_metrics_history) > MAX_METRICS_HISTORY:
            system_metrics_history.pop(0)
        
        return jsonify({
            'current_metrics': current_metrics,
            'metrics_history': system_metrics_history[-10:],  # Last 10 entries
            'collection_timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v5/metrics/system')
def get_system_metrics():
    """Get detailed system metrics"""
    try:
        metrics = collect_detailed_system_metrics()
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"System metrics collection failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/dashboard')
def dashboard():
    """Real-time monitoring dashboard"""
    dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ATMAN-CANON v5 Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0a0a0a; color: #e0e0e0; line-height: 1.6;
        }
        .header { 
            background: #1a1a1a; padding: 1rem 2rem; border-bottom: 1px solid #333;
            display: flex; justify-content: space-between; align-items: center;
        }
        .header h1 { color: #4a9eff; font-size: 1.5rem; }
        .status { color: #00ff88; font-weight: bold; }
        .container { padding: 2rem; max-width: 1400px; margin: 0 auto; }
        .grid { 
            display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem; margin-bottom: 2rem;
        }
        .card { 
            background: #1a1a1a; border: 1px solid #333; border-radius: 8px;
            padding: 1.5rem; transition: border-color 0.3s;
        }
        .card:hover { border-color: #4a9eff; }
        .card h3 { color: #4a9eff; margin-bottom: 1rem; font-size: 1.1rem; }
        .metric { 
            display: flex; justify-content: space-between; align-items: center;
            margin: 0.5rem 0; padding: 0.5rem; background: #0f0f0f; border-radius: 4px;
        }
        .metric-label { color: #b0b0b0; }
        .metric-value { color: #e0e0e0; font-weight: bold; }
        .progress-bar { 
            width: 100%; height: 8px; background: #333; border-radius: 4px;
            overflow: hidden; margin: 0.5rem 0;
        }
        .progress-fill { 
            height: 100%; background: linear-gradient(90deg, #00ff88, #4a9eff);
            transition: width 0.3s ease;
        }
        .chart-container { 
            background: #1a1a1a; border: 1px solid #333; border-radius: 8px;
            padding: 1.5rem; margin-bottom: 2rem;
        }
        .chart { 
            width: 100%; height: 200px; background: #0f0f0f; border-radius: 4px;
            display: flex; align-items: end; justify-content: space-around; padding: 1rem;
        }
        .chart-bar { 
            width: 20px; background: linear-gradient(180deg, #4a9eff, #00ff88);
            border-radius: 2px; transition: height 0.3s ease;
        }
        .timestamp { color: #888; font-size: 0.85rem; }
        .error { color: #ff4444; }
        .warning { color: #ffaa00; }
        .success { color: #00ff88; }
    </style>
</head>
<body>
    <div class="header">
        <h1>ATMAN-CANON v5 Production Dashboard</h1>
        <div class="status" id="status">● OPERATIONAL</div>
    </div>
    
    <div class="container">
        <div class="grid">
            <div class="card">
                <h3>System Resources</h3>
                <div class="metric">
                    <span class="metric-label">CPU Usage</span>
                    <span class="metric-value" id="cpu-usage">---%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="cpu-progress" style="width: 0%"></div>
                </div>
                
                <div class="metric">
                    <span class="metric-label">Memory Usage</span>
                    <span class="metric-value" id="memory-usage">---%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="memory-progress" style="width: 0%"></div>
                </div>
                
                <div class="metric">
                    <span class="metric-label">Disk Usage</span>
                    <span class="metric-value" id="disk-usage">---%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="disk-progress" style="width: 0%"></div>
                </div>
            </div>
            
            <div class="card">
                <h3>Evidence Calculator</h3>
                <div class="metric">
                    <span class="metric-label">Total Evidence</span>
                    <span class="metric-value" id="evidence-total">---</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Acceptance Rate</span>
                    <span class="metric-value" id="evidence-rate">---%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Avg Confidence</span>
                    <span class="metric-value" id="evidence-confidence">---</span>
                </div>
            </div>
            
            <div class="card">
                <h3>Hypothesis Generator</h3>
                <div class="metric">
                    <span class="metric-label">Total Hypotheses</span>
                    <span class="metric-value" id="hypothesis-total">---</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Avg Quality</span>
                    <span class="metric-value" id="hypothesis-quality">---</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Generations</span>
                    <span class="metric-value" id="hypothesis-generations">---</span>
                </div>
            </div>
            
            <div class="card">
                <h3>Transfer Learning</h3>
                <div class="metric">
                    <span class="metric-label">Total Transfers</span>
                    <span class="metric-value" id="transfer-total">---</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Domains</span>
                    <span class="metric-value" id="transfer-domains">---</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Avg Quality</span>
                    <span class="metric-value" id="transfer-quality">---</span>
                </div>
            </div>
            
            <div class="card">
                <h3>RBMK Framework</h3>
                <div class="metric">
                    <span class="metric-label">Items Processed</span>
                    <span class="metric-value" id="rbmk-processed">---</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Avg Confidence</span>
                    <span class="metric-value" id="rbmk-confidence">---</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Recursion Rate</span>
                    <span class="metric-value" id="rbmk-recursion">---%</span>
                </div>
            </div>
            
            <div class="card">
                <h3>Network Activity</h3>
                <div class="metric">
                    <span class="metric-label">Bytes Sent</span>
                    <span class="metric-value" id="network-sent">--- MB</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Bytes Received</span>
                    <span class="metric-value" id="network-recv">--- MB</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Connections</span>
                    <span class="metric-value" id="network-connections">---</span>
                </div>
            </div>
        </div>
        
        <div class="chart-container">
            <h3 style="color: #4a9eff; margin-bottom: 1rem;">CPU Usage History</h3>
            <div class="chart" id="cpu-chart">
                <!-- Chart bars will be populated by JavaScript -->
            </div>
        </div>
        
        <div class="timestamp" id="last-update">Last updated: ---</div>
    </div>
    
    <script>
        let metricsHistory = [];
        
        async function fetchMetrics() {
            try {
                const response = await fetch('/api/v5/metrics');
                const data = await response.json();
                updateDashboard(data.current_metrics);
                updateChart(data.metrics_history || []);
                document.getElementById('last-update').textContent = 
                    'Last updated: ' + new Date().toLocaleTimeString();
            } catch (error) {
                console.error('Failed to fetch metrics:', error);
                document.getElementById('status').textContent = '● ERROR';
                document.getElementById('status').className = 'error';
            }
        }
        
        function updateDashboard(metrics) {
            // System resources
            document.getElementById('cpu-usage').textContent = metrics.cpu_percent.toFixed(1) + '%';
            document.getElementById('cpu-progress').style.width = metrics.cpu_percent + '%';
            
            document.getElementById('memory-usage').textContent = metrics.memory_percent.toFixed(1) + '%';
            document.getElementById('memory-progress').style.width = metrics.memory_percent + '%';
            
            document.getElementById('disk-usage').textContent = metrics.disk_percent.toFixed(1) + '%';
            document.getElementById('disk-progress').style.width = metrics.disk_percent + '%';
            
            // Component metrics
            document.getElementById('evidence-total').textContent = metrics.evidence_total || 0;
            document.getElementById('evidence-rate').textContent = 
                ((metrics.evidence_acceptance_rate || 0) * 100).toFixed(1) + '%';
            document.getElementById('evidence-confidence').textContent = 
                (metrics.evidence_avg_confidence || 0).toFixed(3);
            
            document.getElementById('hypothesis-total').textContent = metrics.hypothesis_total || 0;
            document.getElementById('hypothesis-quality').textContent = 
                (metrics.hypothesis_avg_quality || 0).toFixed(3);
            document.getElementById('hypothesis-generations').textContent = 
                metrics.hypothesis_generations || 0;
            
            document.getElementById('transfer-total').textContent = metrics.transfer_total || 0;
            document.getElementById('transfer-domains').textContent = metrics.transfer_domains || 0;
            document.getElementById('transfer-quality').textContent = 
                (metrics.transfer_avg_quality || 0).toFixed(3);
            
            document.getElementById('rbmk-processed').textContent = metrics.rbmk_processed || 0;
            document.getElementById('rbmk-confidence').textContent = 
                (metrics.rbmk_avg_confidence || 0).toFixed(3);
            document.getElementById('rbmk-recursion').textContent = 
                ((metrics.rbmk_recursion_rate || 0) * 100).toFixed(1) + '%';
            
            // Network metrics
            document.getElementById('network-sent').textContent = 
                (metrics.network_bytes_sent / 1024 / 1024).toFixed(2) + ' MB';
            document.getElementById('network-recv').textContent = 
                (metrics.network_bytes_recv / 1024 / 1024).toFixed(2) + ' MB';
            document.getElementById('network-connections').textContent = 
                metrics.network_connections || 0;
        }
        
        function updateChart(history) {
            const chart = document.getElementById('cpu-chart');
            chart.innerHTML = '';
            
            const maxHistory = 20;
            const recentHistory = history.slice(-maxHistory);
            
            recentHistory.forEach(entry => {
                const bar = document.createElement('div');
                bar.className = 'chart-bar';
                bar.style.height = (entry.cpu_percent || 0) * 2 + 'px';
                bar.title = `CPU: ${(entry.cpu_percent || 0).toFixed(1)}%`;
                chart.appendChild(bar);
            });
        }
        
        // Auto-refresh every 2 seconds
        setInterval(fetchMetrics, 2000);
        
        // Initial load
        fetchMetrics();
    </script>
</body>
</html>
    """
    return dashboard_html

# API Endpoints for ATMAN-CANON components

@app.route('/api/v5/evidence/evaluate', methods=['POST'])
def evaluate_evidence():
    """Evaluate evidence using Evidence Calculator"""
    try:
        evidence_data = request.get_json()
        if not evidence_data:
            return jsonify({'error': 'No evidence data provided'}), 400
        
        result = evidence_calculator.evaluate_evidence(evidence_data)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Evidence evaluation failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v5/hypotheses/generate', methods=['POST'])
def generate_hypotheses():
    """Generate hypotheses using Hypothesis Generator"""
    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify({'error': 'No request data provided'}), 400
        
        evidence_set = request_data.get('evidence_set', [])
        domain_context = request_data.get('domain_context')
        
        hypotheses = hypothesis_generator.generate_hypotheses(evidence_set, domain_context)
        return jsonify({'hypotheses': hypotheses})
        
    except Exception as e:
        logger.error(f"Hypothesis generation failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v5/transfer/learn', methods=['POST'])
def learn_domain():
    """Learn from domain using Transfer Learning Engine"""
    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify({'error': 'No request data provided'}), 400
        
        domain_name = request_data.get('domain_name')
        knowledge_items = request_data.get('knowledge_items', [])
        
        if not domain_name:
            return jsonify({'error': 'Domain name required'}), 400
        
        result = transfer_engine.learn_from_domain(domain_name, knowledge_items)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Domain learning failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v5/transfer/apply', methods=['POST'])
def apply_transfer():
    """Apply transfer learning"""
    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify({'error': 'No request data provided'}), 400
        
        source_domain = request_data.get('source_domain')
        target_domain = request_data.get('target_domain')
        target_context = request_data.get('target_context', {})
        
        if not source_domain or not target_domain:
            return jsonify({'error': 'Source and target domains required'}), 400
        
        result = transfer_engine.transfer_knowledge(source_domain, target_domain, target_context)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Transfer application failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v5/rbmk/process', methods=['POST'])
def process_rbmk():
    """Process knowledge item through RBMK framework"""
    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify({'error': 'No request data provided'}), 400
        
        knowledge_item = request_data.get('knowledge_item')
        context = request_data.get('context')
        
        if not knowledge_item:
            return jsonify({'error': 'Knowledge item required'}), 400
        
        result = rbmk_framework.process_knowledge_item(knowledge_item, context)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"RBMK processing failed: {e}")
        return jsonify({'error': str(e)}), 500

# Helper functions

def collect_system_metrics() -> Dict[str, Any]:
    """Collect current system metrics"""
    
    # System resources
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    network = psutil.net_io_counters()
    
    # Component summaries
    evidence_summary = evidence_calculator.get_evidence_summary()
    hypothesis_summary = hypothesis_generator.get_generation_summary()
    transfer_summary = transfer_engine.get_transfer_summary()
    rbmk_summary = rbmk_framework.get_rbmk_summary()
    
    return {
        'timestamp': datetime.utcnow().isoformat(),
        'cpu_percent': cpu_percent,
        'memory_percent': memory.percent,
        'memory_available_gb': memory.available / (1024**3),
        'disk_percent': (disk.used / disk.total) * 100,
        'network_bytes_sent': network.bytes_sent,
        'network_bytes_recv': network.bytes_recv,
        'network_connections': len(psutil.net_connections()),
        
        # Component metrics
        'evidence_total': evidence_summary.get('total_evidence', 0),
        'evidence_acceptance_rate': evidence_summary.get('acceptance_rate', 0.0),
        'evidence_avg_confidence': evidence_summary.get('average_confidence', 0.0),
        
        'hypothesis_total': hypothesis_summary.get('total_hypotheses', 0),
        'hypothesis_generations': hypothesis_summary.get('total_generations', 0),
        'hypothesis_avg_quality': hypothesis_summary.get('average_quality', 0.0),
        
        'transfer_total': transfer_summary.get('total_transfers', 0),
        'transfer_domains': transfer_summary.get('domains_involved', 0),
        'transfer_avg_quality': transfer_summary.get('average_quality', 0.0),
        
        'rbmk_processed': rbmk_summary.get('total_processed', 0),
        'rbmk_avg_confidence': rbmk_summary.get('average_confidence', 0.0),
        'rbmk_recursion_rate': rbmk_summary.get('recursive_reasoning_rate', 0.0)
    }

def collect_detailed_system_metrics() -> Dict[str, Any]:
    """Collect detailed system metrics"""
    
    # Process information
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
        try:
            processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    # Top processes by CPU
    top_cpu = sorted(processes, key=lambda x: x['cpu_percent'] or 0, reverse=True)[:5]
    
    # Top processes by memory
    top_memory = sorted(processes, key=lambda x: x['memory_percent'] or 0, reverse=True)[:5]
    
    return {
        'timestamp': datetime.utcnow().isoformat(),
        'system_info': {
            'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat(),
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
        },
        'top_processes': {
            'by_cpu': top_cpu,
            'by_memory': top_memory
        },
        'disk_io': psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {},
        'network_io': psutil.net_io_counters()._asdict()
    }

if __name__ == '__main__':
    logger.info("Starting ATMAN-CANON v5 Production Server")
    logger.info("Fractal Consciousness Framework for Emergent Intelligence")
    
    # Run the application
    app.run(host='0.0.0.0', port=8080, debug=False)
