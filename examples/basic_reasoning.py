#!/usr/bin/env python3
"""
Basic Reasoning Example for ATMAN-CANON Framework
Demonstrates core reasoning capabilities and evidence evaluation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from atman_core import RBMKReasoner, EvidenceCalculator, InvariantDetector
from atman_core.utils import RenormalizationSafety, KappaBlockLogic
import numpy as np

def create_sample_evidence():
    """Create sample evidence for demonstration."""
    evidence_set = [
        {
            'id': 'E001',
            'domain': 'medical',
            'features': {
                'temperature': 38.5,
                'symptoms': ['fever', 'cough'],
                'duration_days': 3
            },
            'confidence': 0.85,
            'source': 'clinical_observation',
            'timestamp': '2024-01-15T10:30:00',
            'conclusion': 'patient shows signs of respiratory infection'
        },
        {
            'id': 'E002', 
            'domain': 'medical',
            'features': {
                'temperature': 38.2,
                'symptoms': ['fever', 'fatigue'],
                'duration_days': 2
            },
            'confidence': 0.78,
            'source': 'patient_report',
            'timestamp': '2024-01-15T11:00:00',
            'conclusion': 'symptoms consistent with viral infection'
        },
        {
            'id': 'E003',
            'domain': 'medical', 
            'features': {
                'temperature': 37.1,
                'symptoms': ['mild_cough'],
                'duration_days': 1
            },
            'confidence': 0.65,
            'source': 'follow_up',
            'timestamp': '2024-01-16T09:00:00',
            'conclusion': 'symptoms improving with treatment'
        }
    ]
    
    return evidence_set

def demonstrate_evidence_evaluation():
    """Demonstrate evidence evaluation capabilities."""
    print("=" * 60)
    print("ATMAN-CANON Framework - Basic Reasoning Demonstration")
    print("=" * 60)
    
    # Initialize components
    evidence_calc = EvidenceCalculator()
    invariant_detector = InvariantDetector()
    
    print("\n1. EVIDENCE EVALUATION")
    print("-" * 30)
    
    # Create sample evidence
    evidence_set = create_sample_evidence()
    
    # Evaluate each piece of evidence
    for evidence in evidence_set:
        print(f"\nEvaluating Evidence {evidence['id']}:")
        print(f"  Domain: {evidence['domain']}")
        print(f"  Confidence: {evidence['confidence']:.2f}")
        print(f"  Conclusion: {evidence['conclusion']}")
        
        # Calculate evidence metrics
        metrics = evidence_calc.evaluate_evidence(evidence)
        accepted = metrics.get('overall_confidence', 0) > evidence_calc.confidence_threshold
        
        print(f"  Overall Confidence: {metrics.get('overall_confidence', 0):.3f}")
        print(f"  Source Reliability: {metrics.get('source_reliability', 0):.3f}")
        print(f"  Data Completeness: {metrics.get('data_completeness', 0):.3f}")
        print(f"  Accepted: {'✓' if accepted else '✗'}")
    
    return evidence_set

def demonstrate_invariant_detection(evidence_set):
    """Demonstrate invariant detection."""
    print("\n\n2. INVARIANT DETECTION")
    print("-" * 30)
    
    invariant_detector = InvariantDetector()
    
    # Detect invariants in evidence set
    invariants = invariant_detector.detect_invariants(evidence_set)
    
    print("\nDetected Invariants:")
    for invariant_type, data in invariants.items():
        print(f"\n  {invariant_type.replace('_', ' ').title()}:")
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.3f}")
                else:
                    print(f"    {key}: {value}")
    
    # Test new evidence against invariants
    print("\n\nTesting New Evidence Against Invariants:")
    new_evidence = {
        'id': 'E004',
        'domain': 'medical',
        'features': {
            'temperature': 42.0,  # Extreme value
            'symptoms': ['severe_fever'],
            'duration_days': 7
        },
        'confidence': 0.95,
        'conclusion': 'critical condition detected'
    }
    
    is_valid, violations = invariant_detector.validate_against_invariants(new_evidence)
    print(f"  New evidence valid: {'✓' if is_valid else '✗'}")
    if violations:
        print("  Violations:")
        for violation in violations:
            print(f"    - {violation}")

def demonstrate_rbmk_reasoning(evidence_set):
    """Demonstrate RBMK recursive reasoning."""
    print("\n\n3. RBMK RECURSIVE REASONING")
    print("-" * 30)
    
    reasoner = RBMKReasoner()
    
    # Process evidence through RBMK
    print("\nProcessing evidence through RBMK framework...")
    
    for evidence in evidence_set:
        result = reasoner.process_knowledge_item(evidence)
        
        print(f"\nEvidence {evidence['id']} → RBMK Processing:")
        print(f"  Generated {len(result.get('hypotheses', []))} hypotheses")
        print(f"  Meta-knowledge updated: {result.get('meta_knowledge_updated', False)}")
        print(f"  Recursion depth: {result.get('recursion_depth', 0)}")
        
        # Show top hypothesis if available
        hypotheses = result.get('hypotheses', [])
        if hypotheses:
            top_hypothesis = max(hypotheses, key=lambda h: h.get('confidence', 0))
            print(f"  Top hypothesis: {top_hypothesis.get('description', 'N/A')}")
            print(f"  Confidence: {top_hypothesis.get('confidence', 0):.3f}")

def demonstrate_safety_mechanisms():
    """Demonstrate safety mechanisms."""
    print("\n\n4. SAFETY MECHANISMS")
    print("-" * 30)
    
    # Renormalization Safety
    safety = RenormalizationSafety()
    kappa_logic = KappaBlockLogic()
    
    print("\nRenormalization Safety Test:")
    
    # Create a high-energy state that should trigger renormalization
    dangerous_state = {
        'confidence': 0.999,  # Near boundary
        'features': {'extreme_value': 1000.0},
        'hypotheses': [{'id': i, 'confidence': 0.9} for i in range(20)],  # Too many
        'recursion_depth': 15  # Too deep
    }
    
    print("  Original state energy: HIGH")
    print(f"  Confidence: {dangerous_state['confidence']}")
    print(f"  Hypotheses count: {len(dangerous_state['hypotheses'])}")
    print(f"  Recursion depth: {dangerous_state['recursion_depth']}")
    
    # Apply renormalization
    safe_state = safety.apply_renormalization(dangerous_state)
    
    print("\n  After renormalization:")
    print(f"  Confidence: {safe_state['confidence']:.3f}")
    print(f"  Hypotheses count: {len(safe_state.get('hypotheses', []))}")
    print(f"  Recursion depth: {safe_state.get('recursion_depth', 0)}")
    print(f"  Renormalization applied: {safe_state.get('renormalization_applied', False)}")
    
    # κ-Block Logic Test
    print("\n\nκ-Block Logic Test:")
    
    premises = [
        {'text': 'All observed patients have fever', 'confidence': 0.9},
        {'text': 'Patient X is observed', 'confidence': 0.95}
    ]
    
    conclusion = {'text': 'Patient X has fever', 'confidence': 0.85}
    
    block = kappa_logic.create_logical_block(premises, conclusion)
    
    print(f"  κ-block created: {block['valid']}")
    print(f"  Consistency score: {block['consistency_score']:.3f}")
    print(f"  κ threshold: {kappa_logic.kappa}")

def demonstrate_cross_domain_transfer():
    """Demonstrate cross-domain knowledge transfer."""
    print("\n\n5. CROSS-DOMAIN TRANSFER")
    print("-" * 30)
    
    from atman_core import TransferLearningEngine
    
    transfer_engine = TransferLearningEngine()
    
    # Medical domain knowledge
    medical_knowledge = {
        'domain': 'medical',
        'patterns': {
            'fever_pattern': {'temperature': '>38.0', 'symptoms': ['fever']},
            'recovery_pattern': {'temperature': '<37.5', 'symptoms': ['improving']}
        },
        'relationships': [
            {'cause': 'infection', 'effect': 'fever', 'strength': 0.8},
            {'cause': 'treatment', 'effect': 'recovery', 'strength': 0.7}
        ]
    }
    
    # Attempt transfer to financial domain
    print("Transferring medical patterns to financial domain...")
    
    financial_context = {
        'domain': 'finance',
        'features': ['market_temperature', 'volatility_symptoms', 'trend_duration'],
        'target_patterns': ['market_stress', 'recovery_signals']
    }
    
    transfer_result = transfer_engine.transfer_knowledge(
        source_knowledge=medical_knowledge,
        target_context=financial_context
    )
    
    print(f"  Transfer successful: {transfer_result.get('success', False)}")
    print(f"  Confidence: {transfer_result.get('confidence', 0):.3f}")
    print(f"  Analogies found: {len(transfer_result.get('analogies', []))}")
    
    if transfer_result.get('analogies'):
        print("  Key analogies:")
        for analogy in transfer_result['analogies'][:2]:
            print(f"    {analogy.get('source', 'N/A')} → {analogy.get('target', 'N/A')}")

def main():
    """Main demonstration function."""
    try:
        # Run all demonstrations
        evidence_set = demonstrate_evidence_evaluation()
        demonstrate_invariant_detection(evidence_set)
        demonstrate_rbmk_reasoning(evidence_set)
        demonstrate_safety_mechanisms()
        demonstrate_cross_domain_transfer()
        
        print("\n\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("\nThe ATMAN-CANON framework has successfully demonstrated:")
        print("✓ Evidence evaluation and quality assessment")
        print("✓ Invariant detection and validation")
        print("✓ Recursive Bayesian meta-knowledge reasoning")
        print("✓ Safety mechanisms and renormalization")
        print("✓ Cross-domain knowledge transfer")
        print("\nFramework is ready for domain-specific applications.")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        print("This may indicate missing dependencies or import issues.")
        print("Ensure the atman_core package is properly installed.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
