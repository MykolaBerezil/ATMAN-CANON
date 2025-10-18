"""
Invariant Detection and Preservation System
Maintains logical consistency across reasoning scales and domains.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging

class InvariantDetector:
    """
    Detects and preserves logical invariants across reasoning operations.
    Ensures consistency at multiple scales and prevents logical contradictions.
    """
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        self.detected_invariants = {}
        self.violation_history = []
        self.logger = logging.getLogger(__name__)
        
    def detect_invariants(self, evidence_set: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detect logical invariants in a set of evidence.
        
        Args:
            evidence_set: List of evidence items to analyze
            
        Returns:
            Dictionary of detected invariants with confidence scores
        """
        invariants = {
            'logical_consistency': self._check_logical_consistency(evidence_set),
            'causal_relationships': self._detect_causal_invariants(evidence_set),
            'domain_constraints': self._identify_domain_constraints(evidence_set),
            'temporal_consistency': self._check_temporal_invariants(evidence_set)
        }
        
        # Store detected invariants
        self.detected_invariants.update(invariants)
        
        return invariants
    
    def validate_against_invariants(self, new_evidence: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate new evidence against existing invariants.
        
        Args:
            new_evidence: New evidence to validate
            
        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []
        
        # Check against each type of invariant
        for invariant_type, invariant_data in self.detected_invariants.items():
            violation = self._check_invariant_violation(new_evidence, invariant_type, invariant_data)
            if violation:
                violations.append(violation)
        
        is_valid = len(violations) == 0
        
        if not is_valid:
            self.violation_history.append({
                'evidence': new_evidence,
                'violations': violations,
                'timestamp': np.datetime64('now')
            })
        
        return is_valid, violations
    
    def _check_logical_consistency(self, evidence_set: List[Dict[str, Any]]) -> Dict[str, float]:
        """Check for logical consistency patterns in evidence."""
        consistency_score = 0.0
        contradiction_count = 0
        
        for i, evidence1 in enumerate(evidence_set):
            for j, evidence2 in enumerate(evidence_set[i+1:], i+1):
                # Check for direct contradictions
                if self._are_contradictory(evidence1, evidence2):
                    contradiction_count += 1
                else:
                    consistency_score += self._calculate_consistency_score(evidence1, evidence2)
        
        total_pairs = len(evidence_set) * (len(evidence_set) - 1) // 2
        if total_pairs > 0:
            consistency_score /= total_pairs
        
        return {
            'consistency_score': consistency_score,
            'contradiction_ratio': contradiction_count / max(total_pairs, 1),
            'confidence': max(0.0, 1.0 - contradiction_count / max(total_pairs, 1))
        }
    
    def _detect_causal_invariants(self, evidence_set: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect causal relationship patterns."""
        causal_chains = []
        causal_strength = {}
        
        for evidence in evidence_set:
            if 'causal_relationships' in evidence:
                for relationship in evidence['causal_relationships']:
                    cause = relationship.get('cause')
                    effect = relationship.get('effect')
                    strength = relationship.get('strength', 0.5)
                    
                    if cause and effect:
                        causal_key = f"{cause} -> {effect}"
                        if causal_key not in causal_strength:
                            causal_strength[causal_key] = []
                        causal_strength[causal_key].append(strength)
        
        # Calculate average causal strengths
        stable_causals = {}
        for causal_key, strengths in causal_strength.items():
            avg_strength = np.mean(strengths)
            variance = np.var(strengths)
            if variance < self.tolerance:  # Stable causal relationship
                stable_causals[causal_key] = {
                    'strength': avg_strength,
                    'stability': 1.0 - variance,
                    'evidence_count': len(strengths)
                }
        
        return {
            'stable_causal_relationships': stable_causals,
            'total_relationships': len(causal_strength),
            'stable_count': len(stable_causals)
        }
    
    def _identify_domain_constraints(self, evidence_set: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify domain-specific constraints and bounds."""
        constraints = {}
        
        # Collect domain-specific bounds
        for evidence in evidence_set:
            domain = evidence.get('domain', 'general')
            if domain not in constraints:
                constraints[domain] = {
                    'value_bounds': {},
                    'categorical_constraints': set(),
                    'relationship_constraints': []
                }
            
            # Extract numerical bounds
            if 'features' in evidence:
                for feature, value in evidence['features'].items():
                    if isinstance(value, (int, float)):
                        if feature not in constraints[domain]['value_bounds']:
                            constraints[domain]['value_bounds'][feature] = {'min': value, 'max': value}
                        else:
                            constraints[domain]['value_bounds'][feature]['min'] = min(
                                constraints[domain]['value_bounds'][feature]['min'], value
                            )
                            constraints[domain]['value_bounds'][feature]['max'] = max(
                                constraints[domain]['value_bounds'][feature]['max'], value
                            )
            
            # Extract categorical constraints
            if 'categories' in evidence:
                constraints[domain]['categorical_constraints'].update(evidence['categories'])
        
        # Convert sets to lists for JSON serialization
        for domain_constraints in constraints.values():
            domain_constraints['categorical_constraints'] = list(domain_constraints['categorical_constraints'])
        
        return constraints
    
    def _check_temporal_invariants(self, evidence_set: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check for temporal consistency and ordering invariants."""
        temporal_violations = []
        temporal_patterns = {}
        
        # Sort evidence by timestamp if available
        timestamped_evidence = [e for e in evidence_set if 'timestamp' in e]
        timestamped_evidence.sort(key=lambda x: x['timestamp'])
        
        # Check for temporal consistency
        for i in range(len(timestamped_evidence) - 1):
            current = timestamped_evidence[i]
            next_evidence = timestamped_evidence[i + 1]
            
            # Check if later evidence contradicts earlier evidence
            if self._are_temporally_inconsistent(current, next_evidence):
                temporal_violations.append({
                    'earlier': current,
                    'later': next_evidence,
                    'violation_type': 'temporal_contradiction'
                })
        
        return {
            'temporal_violations': temporal_violations,
            'violation_count': len(temporal_violations),
            'temporal_consistency_score': 1.0 - len(temporal_violations) / max(len(timestamped_evidence), 1)
        }
    
    def _are_contradictory(self, evidence1: Dict[str, Any], evidence2: Dict[str, Any]) -> bool:
        """Check if two pieces of evidence are contradictory."""
        # Simple contradiction check based on opposing conclusions
        if 'conclusion' in evidence1 and 'conclusion' in evidence2:
            conclusion1 = evidence1['conclusion'].lower()
            conclusion2 = evidence2['conclusion'].lower()
            
            # Check for explicit negations
            if ('not' in conclusion1 and conclusion1.replace('not ', '') == conclusion2) or \
               ('not' in conclusion2 and conclusion2.replace('not ', '') == conclusion1):
                return True
        
        # Check for contradictory feature values
        if 'features' in evidence1 and 'features' in evidence2:
            for feature in evidence1['features']:
                if feature in evidence2['features']:
                    val1 = evidence1['features'][feature]
                    val2 = evidence2['features'][feature]
                    
                    # For boolean features, check direct contradiction
                    if isinstance(val1, bool) and isinstance(val2, bool) and val1 != val2:
                        return True
                    
                    # For numerical features, check if they're impossibly different
                    if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                        if abs(val1 - val2) > 10 * self.tolerance:  # Significant difference
                            return True
        
        return False
    
    def _calculate_consistency_score(self, evidence1: Dict[str, Any], evidence2: Dict[str, Any]) -> float:
        """Calculate consistency score between two pieces of evidence."""
        score = 0.0
        comparisons = 0
        
        # Compare confidence levels
        if 'confidence' in evidence1 and 'confidence' in evidence2:
            conf_diff = abs(evidence1['confidence'] - evidence2['confidence'])
            score += 1.0 - conf_diff
            comparisons += 1
        
        # Compare feature similarity
        if 'features' in evidence1 and 'features' in evidence2:
            common_features = set(evidence1['features'].keys()) & set(evidence2['features'].keys())
            for feature in common_features:
                val1 = evidence1['features'][feature]
                val2 = evidence2['features'][feature]
                
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # Normalized similarity for numerical values
                    max_val = max(abs(val1), abs(val2), 1.0)
                    similarity = 1.0 - abs(val1 - val2) / max_val
                    score += similarity
                    comparisons += 1
                elif val1 == val2:
                    score += 1.0
                    comparisons += 1
        
        return score / max(comparisons, 1)
    
    def _check_invariant_violation(self, evidence: Dict[str, Any], invariant_type: str, 
                                 invariant_data: Dict[str, Any]) -> Optional[str]:
        """Check if evidence violates a specific invariant."""
        if invariant_type == 'domain_constraints':
            domain = evidence.get('domain', 'general')
            if domain in invariant_data:
                constraints = invariant_data[domain]
                
                # Check value bounds
                if 'features' in evidence:
                    for feature, value in evidence['features'].items():
                        if feature in constraints['value_bounds']:
                            bounds = constraints['value_bounds'][feature]
                            if isinstance(value, (int, float)):
                                if value < bounds['min'] or value > bounds['max']:
                                    return f"Feature '{feature}' value {value} outside bounds [{bounds['min']}, {bounds['max']}]"
                
                # Check categorical constraints
                if 'categories' in evidence:
                    for category in evidence['categories']:
                        if category not in constraints['categorical_constraints']:
                            return f"Unknown category '{category}' for domain '{domain}'"
        
        elif invariant_type == 'logical_consistency':
            # Check if evidence maintains logical consistency threshold
            consistency_threshold = invariant_data.get('confidence', 0.8)
            evidence_confidence = evidence.get('confidence', 0.0)
            if evidence_confidence < consistency_threshold * 0.5:  # Significantly below threshold
                return f"Evidence confidence {evidence_confidence} significantly below consistency threshold"
        
        return None
    
    def _are_temporally_inconsistent(self, earlier: Dict[str, Any], later: Dict[str, Any]) -> bool:
        """Check if later evidence is temporally inconsistent with earlier evidence."""
        # Simple check: if later evidence directly contradicts earlier with lower confidence
        if self._are_contradictory(earlier, later):
            earlier_conf = earlier.get('confidence', 0.5)
            later_conf = later.get('confidence', 0.5)
            
            # If later evidence has much lower confidence but contradicts, it's suspicious
            if later_conf < earlier_conf * 0.7:
                return True
        
        return False
    
    def get_invariant_summary(self) -> Dict[str, Any]:
        """Get summary of all detected invariants."""
        return {
            'total_invariants': len(self.detected_invariants),
            'invariant_types': list(self.detected_invariants.keys()),
            'violation_count': len(self.violation_history),
            'last_update': str(np.datetime64('now'))
        }
