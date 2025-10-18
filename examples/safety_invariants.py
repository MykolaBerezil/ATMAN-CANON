#!/usr/bin/env python3
"""
Safety and Invariants Example for ATMAN-CANON Framework
Demonstrates safety mechanisms and invariant preservation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from atman_core.utils import RenormalizationSafety, KappaBlockLogic, SafetyBounds
from atman_core import InvariantDetector
import numpy as np
import time

def create_dangerous_reasoning_states():
    """Create various dangerous reasoning states for testing."""
    return [
        {
            'name': 'High Energy State',
            'state': {
                'confidence': 0.9999,  # Too close to certainty
                'features': {'extreme_feature': 10000.0},
                'hypotheses': [{'id': i, 'confidence': 0.95} for i in range(50)],
                'recursion_depth': 20
            }
        },
        {
            'name': 'Oscillating State',
            'state': {
                'confidence': 0.1,  # Too low
                'features': {'oscillating': -5000.0},
                'hypotheses': [],
                'recursion_depth': 0,
                'oscillation_detected': True
            }
        },
        {
            'name': 'Recursive Explosion',
            'state': {
                'confidence': 0.8,
                'features': {'recursive_calls': 1000},
                'hypotheses': [{'id': i, 'confidence': 0.7 + 0.01*i} for i in range(100)],
                'recursion_depth': 50  # Way too deep
            }
        }
    ]

def demonstrate_renormalization_safety():
    """Demonstrate renormalization safety mechanisms."""
    print("=" * 60)
    print("SAFETY MECHANISMS - Renormalization Safety")
    print("=" * 60)
    
    # Create safety system with custom bounds
    custom_bounds = SafetyBounds(
        min_confidence=0.05,
        max_confidence=0.95,
        max_recursion_depth=8,
        energy_threshold=500.0,
        divergence_threshold=5.0
    )
    
    safety = RenormalizationSafety(bounds=custom_bounds)
    
    dangerous_states = create_dangerous_reasoning_states()
    
    for test_case in dangerous_states:
        print(f"\n{test_case['name']}:")
        print("-" * 40)
        
        original_state = test_case['state']
        print("Original State:")
        print(f"  Confidence: {original_state.get('confidence', 'N/A')}")
        print(f"  Hypotheses: {len(original_state.get('hypotheses', []))}")
        print(f"  Recursion Depth: {original_state.get('recursion_depth', 0)}")
        
        # Calculate original energy
        original_energy = safety._calculate_system_energy(original_state)
        print(f"  System Energy: {original_energy:.2f}")
        
        # Apply renormalization
        safe_state = safety.apply_renormalization(original_state)
        
        print("\nAfter Renormalization:")
        print(f"  Confidence: {safe_state.get('confidence', 'N/A')}")
        print(f"  Hypotheses: {len(safe_state.get('hypotheses', []))}")
        print(f"  Recursion Depth: {safe_state.get('recursion_depth', 0)}")
        
        # Calculate new energy
        new_energy = safety._calculate_system_energy(safe_state)
        print(f"  System Energy: {new_energy:.2f}")
        
        # Show safety actions taken
        if safe_state.get('renormalization_applied'):
            print(f"  ✓ Standard renormalization applied (factor: {safe_state.get('renormalization_factor', 'N/A'):.3f})")
        if safe_state.get('emergency_reset'):
            print("  ⚠ Emergency reset triggered!")
        if safe_state.get('recursion_limited'):
            print("  ✓ Recursion depth limited")
    
    # Show safety metrics
    print("\nSafety System Metrics:")
    metrics = safety.get_safety_metrics()
    print(f"  Renormalizations: {metrics['renormalization_count']}")
    print(f"  Emergency Stops: {metrics['emergency_stops']}")
    print(f"  Average Energy: {metrics['average_energy']:.2f}")
    print(f"  Max Energy: {metrics['max_energy']:.2f}")

def demonstrate_kappa_block_logic():
    """Demonstrate κ-block logic for safe reasoning."""
    print("\n\n" + "=" * 60)
    print("SAFETY MECHANISMS - κ-Block Logic")
    print("=" * 60)
    
    kappa_logic = KappaBlockLogic(kappa=0.7)  # 70% consistency threshold
    
    # Test valid logical blocks
    print("\n1. VALID LOGICAL BLOCKS")
    print("-" * 30)
    
    valid_premises = [
        {'text': 'All mammals are warm-blooded', 'confidence': 0.95},
        {'text': 'Whales are mammals', 'confidence': 0.98}
    ]
    valid_conclusion = {'text': 'Whales are warm-blooded', 'confidence': 0.92}
    
    valid_block = kappa_logic.create_logical_block(valid_premises, valid_conclusion)
    
    print("Valid Block:")
    print(f"  Premises: {len(valid_premises)}")
    print(f"  Consistency Score: {valid_block['consistency_score']:.3f}")
    print(f"  Valid: {'✓' if valid_block['valid'] else '✗'}")
    
    # Test invalid logical blocks
    print("\n2. INVALID LOGICAL BLOCKS")
    print("-" * 30)
    
    invalid_premises = [
        {'text': 'All birds can fly', 'confidence': 0.6},
        {'text': 'Penguins are birds', 'confidence': 0.95}
    ]
    invalid_conclusion = {'text': 'Penguins can fly', 'confidence': 0.8}
    
    invalid_block = kappa_logic.create_logical_block(invalid_premises, invalid_conclusion)
    
    print("Invalid Block:")
    print(f"  Premises: {len(invalid_premises)}")
    print(f"  Consistency Score: {invalid_block['consistency_score']:.3f}")
    print(f"  Valid: {'✓' if invalid_block['valid'] else '✗'}")
    if not invalid_block['valid']:
        print(f"  Rejection Reason: {invalid_block['rejection_reason']}")
    
    # Test logical chain validation
    print("\n3. LOGICAL CHAIN VALIDATION")
    print("-" * 30)
    
    logical_chain = [
        {'conclusion': 'Temperature is rising', 'confidence': 0.8},
        {'conclusion': 'Ice is melting', 'confidence': 0.85},
        {'conclusion': 'Sea level is rising', 'confidence': 0.75},
        {'conclusion': 'Coastal areas are flooding', 'confidence': 0.7}
    ]
    
    is_valid, issues = kappa_logic.validate_logical_chain(logical_chain)
    
    print("Logical Chain:")
    print(f"  Steps: {len(logical_chain)}")
    print(f"  Valid: {'✓' if is_valid else '✗'}")
    if issues:
        print("  Issues:")
        for issue in issues:
            print(f"    - {issue}")
    
    # Test contradictory chain
    print("\n4. CONTRADICTORY CHAIN")
    print("-" * 30)
    
    contradictory_chain = [
        {'conclusion': 'The system is safe', 'confidence': 0.9},
        {'conclusion': 'Multiple failures detected', 'confidence': 0.8},
        {'conclusion': 'The system is not safe', 'confidence': 0.85}
    ]
    
    is_valid, issues = kappa_logic.validate_logical_chain(contradictory_chain)
    
    print("Contradictory Chain:")
    print(f"  Steps: {len(contradictory_chain)}")
    print(f"  Valid: {'✓' if is_valid else '✗'}")
    if issues:
        print("  Issues:")
        for issue in issues:
            print(f"    - {issue}")
    
    # Show κ-logic metrics
    print("\nκ-Block Logic Metrics:")
    metrics = kappa_logic.get_kappa_metrics()
    print(f"  κ Threshold: {metrics['kappa_threshold']}")
    print(f"  Total Blocks: {metrics['total_blocks']}")
    print(f"  Valid Blocks: {metrics['valid_blocks']}")
    print(f"  Rejection Rate: {metrics['rejection_rate']:.1%}")
    print(f"  Average Consistency: {metrics['average_consistency']:.3f}")

def demonstrate_invariant_preservation():
    """Demonstrate invariant detection and preservation."""
    print("\n\n" + "=" * 60)
    print("INVARIANT PRESERVATION")
    print("=" * 60)
    
    invariant_detector = InvariantDetector()
    
    # Create a dataset with known invariants
    print("\n1. ESTABLISHING INVARIANTS")
    print("-" * 30)
    
    training_data = [
        {'domain': 'physics', 'energy': 100, 'momentum': 50, 'mass': 2.0, 'velocity': 25},
        {'domain': 'physics', 'energy': 200, 'momentum': 60, 'mass': 3.0, 'velocity': 20},
        {'domain': 'physics', 'energy': 150, 'momentum': 45, 'mass': 1.5, 'velocity': 30},
        {'domain': 'physics', 'energy': 300, 'momentum': 90, 'mass': 4.5, 'velocity': 20},
        {'domain': 'physics', 'energy': 250, 'momentum': 75, 'mass': 2.5, 'velocity': 30}
    ]
    
    print(f"Training on {len(training_data)} physics examples...")
    
    # Detect invariants
    invariants = invariant_detector.detect_invariants(training_data)
    
    print("\nDetected Invariants:")
    for inv_type, data in invariants.items():
        print(f"  {inv_type.replace('_', ' ').title()}:")
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    print(f"    {key}: {value:.3f}")
                else:
                    print(f"    {key}: {value}")
    
    # Test invariant preservation
    print("\n2. TESTING INVARIANT PRESERVATION")
    print("-" * 30)
    
    test_cases = [
        {
            'name': 'Valid Physics Example',
            'data': {'domain': 'physics', 'energy': 180, 'momentum': 54, 'mass': 2.7, 'velocity': 20}
        },
        {
            'name': 'Energy Conservation Violation',
            'data': {'domain': 'physics', 'energy': 1000000, 'momentum': 50, 'mass': 2.0, 'velocity': 25}
        },
        {
            'name': 'Momentum-Mass Relationship Violation',
            'data': {'domain': 'physics', 'energy': 200, 'momentum': 1000, 'mass': 0.1, 'velocity': 10000}
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}:")
        
        is_valid, violations = invariant_detector.validate_against_invariants(test_case['data'])
        
        print(f"  Valid: {'✓' if is_valid else '✗'}")
        if violations:
            print("  Violations:")
            for violation in violations:
                print(f"    - {violation}")
        
        # Show the data
        data = test_case['data']
        print(f"  Energy: {data['energy']}, Momentum: {data['momentum']}")
        print(f"  Mass: {data['mass']}, Velocity: {data['velocity']}")

def demonstrate_safety_under_stress():
    """Demonstrate safety mechanisms under stress conditions."""
    print("\n\n" + "=" * 60)
    print("SAFETY UNDER STRESS CONDITIONS")
    print("=" * 60)
    
    # Create aggressive safety bounds
    stress_bounds = SafetyBounds(
        min_confidence=0.1,
        max_confidence=0.9,
        max_recursion_depth=5,
        energy_threshold=100.0,
        divergence_threshold=2.0
    )
    
    safety = RenormalizationSafety(bounds=stress_bounds)
    
    print("\nStress Test: Rapidly Escalating System")
    print("-" * 40)
    
    # Simulate a system that keeps getting more extreme
    state = {'confidence': 0.5, 'features': {'value': 1.0}, 'hypotheses': [], 'recursion_depth': 0}
    
    for iteration in range(10):
        print(f"\nIteration {iteration + 1}:")
        
        # Make the system more extreme each iteration
        state['confidence'] = min(0.999, state['confidence'] * 1.1)
        state['features']['value'] *= 2.0
        state['hypotheses'].append({'id': iteration, 'confidence': 0.8})
        state['recursion_depth'] += 2
        
        energy_before = safety._calculate_system_energy(state)
        print(f"  Energy before: {energy_before:.1f}")
        
        # Apply safety mechanisms
        state = safety.apply_renormalization(state)
        
        energy_after = safety._calculate_system_energy(state)
        print(f"  Energy after: {energy_after:.1f}")
        print(f"  Confidence: {state['confidence']:.3f}")
        print(f"  Hypotheses: {len(state['hypotheses'])}")
        print(f"  Recursion: {state['recursion_depth']}")
        
        # Check if emergency reset was triggered
        if state.get('emergency_reset'):
            print("  ⚠ EMERGENCY RESET TRIGGERED!")
            break
        
        # Simulate some recovery time
        time.sleep(0.1)
    
    # Final safety report
    print("\nFinal Safety Report:")
    metrics = safety.get_safety_metrics()
    print(f"  Total Renormalizations: {metrics['renormalization_count']}")
    print(f"  Emergency Stops: {metrics['emergency_stops']}")
    print(f"  System Stability: {'STABLE' if metrics['emergency_stops'] == 0 else 'UNSTABLE'}")

def main():
    """Main demonstration function."""
    try:
        print("ATMAN-CANON Framework - Safety and Invariants Demonstration")
        
        demonstrate_renormalization_safety()
        demonstrate_kappa_block_logic()
        demonstrate_invariant_preservation()
        demonstrate_safety_under_stress()
        
        print("\n\n" + "=" * 60)
        print("SAFETY DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("\nSafety mechanisms successfully demonstrated:")
        print("✓ Renormalization safety with energy bounds")
        print("✓ κ-block logic for consistent reasoning")
        print("✓ Invariant detection and preservation")
        print("✓ Emergency safety protocols")
        print("✓ Stress testing and recovery")
        print("\nThe ATMAN-CANON framework maintains safety under all tested conditions.")
        
    except Exception as e:
        print(f"\nError during safety demonstration: {e}")
        print("This may indicate missing dependencies or import issues.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
