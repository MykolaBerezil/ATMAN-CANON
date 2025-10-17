#!/usr/bin/env python3
"""
ATMAN-CANON v5 Basic Usage Example
Demonstrates how to use the ATMAN-CANON Fractal Consciousness Framework
"""

import requests
import json
import time
from typing import Dict, Any, List

class ATMANCANONClient:
    """Client for interacting with ATMAN-CANON v5 API"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def get_status(self) -> Dict[str, Any]:
        """Get ATMAN-CANON system status"""
        response = self.session.get(f"{self.base_url}/")
        response.raise_for_status()
        return response.json()
    
    def get_health(self) -> Dict[str, Any]:
        """Get detailed health information"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def evaluate_evidence(self, evidence_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate evidence using Evidence Calculator"""
        response = self.session.post(
            f"{self.base_url}/api/v5/evidence/evaluate",
            json=evidence_data
        )
        response.raise_for_status()
        return response.json()
    
    def generate_hypotheses(self, evidence_set: List[Dict], domain_context: Dict = None) -> Dict[str, Any]:
        """Generate hypotheses using Hypothesis Generator"""
        data = {
            'evidence_set': evidence_set,
            'domain_context': domain_context
        }
        response = self.session.post(
            f"{self.base_url}/api/v5/hypotheses/generate",
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def learn_domain(self, domain_name: str, knowledge_items: List[Dict]) -> Dict[str, Any]:
        """Learn from domain using Transfer Learning Engine"""
        data = {
            'domain_name': domain_name,
            'knowledge_items': knowledge_items
        }
        response = self.session.post(
            f"{self.base_url}/api/v5/transfer/learn",
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def apply_transfer(self, source_domain: str, target_domain: str, target_context: Dict = None) -> Dict[str, Any]:
        """Apply transfer learning between domains"""
        data = {
            'source_domain': source_domain,
            'target_domain': target_domain,
            'target_context': target_context or {}
        }
        response = self.session.post(
            f"{self.base_url}/api/v5/transfer/apply",
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def process_rbmk(self, knowledge_item: Dict[str, Any], context: Dict = None) -> Dict[str, Any]:
        """Process knowledge item through RBMK framework"""
        data = {
            'knowledge_item': knowledge_item,
            'context': context
        }
        response = self.session.post(
            f"{self.base_url}/api/v5/rbmk/process",
            json=data
        )
        response.raise_for_status()
        return response.json()

def demonstrate_evidence_evaluation():
    """Demonstrate evidence evaluation capabilities"""
    print("\n=== Evidence Evaluation Demo ===")
    
    client = ATMANCANONClient()
    
    # Sample evidence data
    evidence_data = {
        'id': 'evidence_001',
        'statement': 'Temperature increases correlate with pressure increases in closed systems',
        'confidence': 0.85,
        'source_reliability': 0.9,
        'supporting_data': {
            'experiment_count': 50,
            'correlation_coefficient': 0.92,
            'p_value': 0.001
        },
        'domain': 'thermodynamics'
    }
    
    try:
        result = client.evaluate_evidence(evidence_data)
        print(f"Evidence Evaluation Result:")
        print(f"  Overall Confidence: {result.get('overall_confidence', 'N/A')}")
        print(f"  Accepted: {result.get('accepted', 'N/A')}")
        print(f"  Quality Score: {result.get('quality_score', 'N/A')}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error evaluating evidence: {e}")

def demonstrate_hypothesis_generation():
    """Demonstrate hypothesis generation capabilities"""
    print("\n=== Hypothesis Generation Demo ===")
    
    client = ATMANCANONClient()
    
    # Sample evidence set
    evidence_set = [
        {
            'id': 'obs_001',
            'observation': 'System pressure increases when temperature rises',
            'confidence': 0.9
        },
        {
            'id': 'obs_002', 
            'observation': 'Volume remains constant during heating',
            'confidence': 0.8
        }
    ]
    
    domain_context = {
        'domain': 'physics',
        'subdomain': 'thermodynamics',
        'constraints': ['closed_system', 'ideal_gas']
    }
    
    try:
        result = client.generate_hypotheses(evidence_set, domain_context)
        print(f"Generated Hypotheses:")
        for i, hypothesis in enumerate(result.get('hypotheses', []), 1):
            print(f"  {i}. {hypothesis.get('statement', 'N/A')}")
            print(f"     Confidence: {hypothesis.get('confidence', 'N/A')}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error generating hypotheses: {e}")

def demonstrate_rbmk_processing():
    """Demonstrate RBMK framework processing"""
    print("\n=== RBMK Framework Demo ===")
    
    client = ATMANCANONClient()
    
    # Sample knowledge item for RBMK processing
    knowledge_item = {
        'id': 'knowledge_001',
        'evidence': {
            'confidence': 0.8,
            'completeness': 0.7,
            'source_reliability': 0.9,
            'supports_hypothesis': 0.75,
            'uncertainty': 0.2
        },
        'hypotheses': [
            {
                'id': 'hyp_1',
                'statement': 'Increasing temperature causes pressure increase in closed systems',
                'confidence': 0.8
            },
            {
                'id': 'hyp_2',
                'statement': 'Gas molecules move faster at higher temperatures',
                'confidence': 0.9
            }
        ],
        'features': {
            'temperature': 300,
            'pressure': 2.0,
            'volume': 10.0,
            'molecule_count': 1000000
        },
        'confidence': 0.75
    }
    
    context = {
        'domain': 'thermodynamics',
        'processing_mode': 'detailed_analysis'
    }
    
    try:
        result = client.process_rbmk(knowledge_item, context)
        print(f"RBMK Processing Result:")
        print(f"  Final Confidence: {result.get('final_confidence', 'N/A'):.3f}")
        print(f"  RBMK Quality Score: {result.get('rbmk_quality_score', 'N/A'):.3f}")
        
        uncertainty_analysis = result.get('uncertainty_analysis', {})
        print(f"  Total Uncertainty: {uncertainty_analysis.get('total_uncertainty', 'N/A'):.3f}")
        
        if result.get('recursive_analysis'):
            print(f"  Recursive Depth: {result['recursive_analysis'].get('recursive_depth', 'N/A')}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error processing RBMK: {e}")

def demonstrate_transfer_learning():
    """Demonstrate transfer learning capabilities"""
    print("\n=== Transfer Learning Demo ===")
    
    client = ATMANCANONClient()
    
    # Learn from physics domain
    physics_knowledge = [
        {
            'concept': 'conservation_of_energy',
            'description': 'Energy cannot be created or destroyed, only transformed',
            'confidence': 0.95,
            'applications': ['mechanics', 'thermodynamics', 'electromagnetism']
        },
        {
            'concept': 'force_acceleration_relationship',
            'description': 'Force equals mass times acceleration (F=ma)',
            'confidence': 0.98,
            'applications': ['classical_mechanics', 'engineering']
        }
    ]
    
    try:
        # Learn physics domain
        learn_result = client.learn_domain('physics', physics_knowledge)
        print(f"Domain Learning Result:")
        print(f"  Domain: {learn_result.get('domain', 'N/A')}")
        print(f"  Concepts Learned: {len(physics_knowledge)}")
        
        # Apply transfer to engineering domain
        transfer_result = client.apply_transfer(
            'physics', 
            'engineering',
            {'target_application': 'structural_analysis'}
        )
        print(f"\nTransfer Learning Result:")
        transferred_items = transfer_result.get('transferred_items', [])
        print(f"  Transferred Items: {len(transferred_items)}")
        
        for item in transferred_items[:2]:  # Show first 2
            print(f"    - {item.get('concept', 'N/A')}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error in transfer learning: {e}")

def monitor_system_health():
    """Monitor ATMAN-CANON system health"""
    print("\n=== System Health Monitoring ===")
    
    client = ATMANCANONClient()
    
    try:
        # Get system status
        status = client.get_status()
        print(f"System Status:")
        print(f"  Name: {status.get('name', 'N/A')}")
        print(f"  Version: {status.get('version', 'N/A')}")
        print(f"  Status: {status.get('status', 'N/A')}")
        
        # Get detailed health information
        health = client.get_health()
        print(f"\nSystem Health:")
        print(f"  Overall Status: {health.get('status', 'N/A')}")
        
        resources = health.get('system_resources', {})
        print(f"  CPU Usage: {resources.get('cpu_percent', 'N/A'):.1f}%")
        print(f"  Memory Usage: {resources.get('memory_percent', 'N/A'):.1f}%")
        print(f"  Available Memory: {resources.get('available_memory_gb', 'N/A'):.2f} GB")
        
        # Component status
        components = health.get('component_status', {})
        print(f"\nComponent Status:")
        for component, status in components.items():
            print(f"  {component}: {status.get('status', 'N/A')}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error monitoring system health: {e}")

def main():
    """Main demonstration function"""
    print("ATMAN-CANON v5 Basic Usage Demonstration")
    print("=========================================")
    print("Fractal Consciousness Framework for Emergent Intelligence")
    
    # Check if ATMAN-CANON is running
    client = ATMANCANONClient()
    try:
        status = client.get_status()
        print(f"\n✅ Connected to {status['name']} v{status['version']}")
    except requests.exceptions.RequestException:
        print("\n❌ Cannot connect to ATMAN-CANON server.")
        print("Please ensure ATMAN-CANON v5 is running on http://localhost:8080")
        return
    
    # Run demonstrations
    monitor_system_health()
    demonstrate_evidence_evaluation()
    demonstrate_hypothesis_generation()
    demonstrate_rbmk_processing()
    demonstrate_transfer_learning()
    
    print("\n=== Demonstration Complete ===")
    print("Visit http://localhost:8080/dashboard for real-time monitoring")

if __name__ == "__main__":
    main()
