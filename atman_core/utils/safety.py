"""
Safety Mechanisms for ATMAN-CANON Framework
Implements renormalization bounds and κ-block logic for safe AI reasoning.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Callable
import logging
from dataclasses import dataclass

@dataclass
class SafetyBounds:
    """Data class for safety boundary parameters."""
    min_confidence: float = 0.1
    max_confidence: float = 0.99
    max_recursion_depth: int = 10
    energy_threshold: float = 1000.0
    divergence_threshold: float = 10.0

class RenormalizationSafety:
    """
    Implements renormalization group safety bounds for AI reasoning.
    Prevents runaway feedback loops and maintains bounded operation.
    """
    
    def __init__(self, bounds: Optional[SafetyBounds] = None):
        self.bounds = bounds or SafetyBounds()
        self.energy_history = []
        self.renormalization_count = 0
        self.emergency_stops = 0
        self.logger = logging.getLogger(__name__)
        
    def apply_renormalization(self, reasoning_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply renormalization to keep reasoning within safe bounds.
        
        Args:
            reasoning_state: Current state of reasoning system
            
        Returns:
            Renormalized reasoning state
        """
        renormalized_state = reasoning_state.copy()
        
        # Calculate current "energy" of the system
        current_energy = self._calculate_system_energy(reasoning_state)
        self.energy_history.append(current_energy)
        
        # Check for divergence
        if self._detect_divergence():
            self.logger.warning("Divergence detected, applying emergency renormalization")
            renormalized_state = self._emergency_renormalization(reasoning_state)
            self.emergency_stops += 1
            return renormalized_state
        
        # Apply standard renormalization if energy exceeds threshold
        if current_energy > self.bounds.energy_threshold:
            renormalized_state = self._standard_renormalization(reasoning_state, current_energy)
            self.renormalization_count += 1
        
        # Bound confidence values
        renormalized_state = self._bound_confidence_values(renormalized_state)
        
        # Limit recursion depth
        renormalized_state = self._limit_recursion_depth(renormalized_state)
        
        return renormalized_state
    
    def _calculate_system_energy(self, state: Dict[str, Any]) -> float:
        """Calculate the total 'energy' of the reasoning system."""
        energy = 0.0
        
        # Confidence energy
        if 'confidence' in state:
            conf = state['confidence']
            # Energy increases near boundaries (0 and 1)
            energy += -np.log(conf + 1e-10) - np.log(1 - conf + 1e-10)
        
        # Feature energy
        if 'features' in state:
            for feature_value in state['features'].values():
                if isinstance(feature_value, (int, float)):
                    energy += feature_value ** 2
        
        # Hypothesis energy
        if 'hypotheses' in state:
            energy += len(state['hypotheses']) * 10.0  # Penalize too many hypotheses
        
        # Recursion energy
        recursion_depth = state.get('recursion_depth', 0)
        energy += recursion_depth ** 2 * 50.0  # Quadratic penalty for deep recursion
        
        return energy
    
    def _detect_divergence(self) -> bool:
        """Detect if the system is diverging based on energy history."""
        if len(self.energy_history) < 3:
            return False
        
        # Check for rapid energy increase
        recent_energies = self.energy_history[-3:]
        energy_gradient = np.gradient(recent_energies)
        
        # Divergence if energy is increasing rapidly
        if np.all(energy_gradient > 0) and energy_gradient[-1] > self.bounds.divergence_threshold:
            return True
        
        # Divergence if energy exceeds critical threshold
        if recent_energies[-1] > self.bounds.energy_threshold * 5:
            return True
        
        return False
    
    def _emergency_renormalization(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply emergency renormalization to prevent system failure."""
        emergency_state = {
            'confidence': 0.5,  # Reset to neutral confidence
            'features': {},
            'hypotheses': [],
            'recursion_depth': 0,
            'emergency_reset': True,
            'original_energy': self.energy_history[-1] if self.energy_history else 0
        }
        
        # Preserve essential information if possible
        if 'domain' in state:
            emergency_state['domain'] = state['domain']
        
        if 'timestamp' in state:
            emergency_state['timestamp'] = state['timestamp']
        
        self.logger.critical(f"Emergency renormalization applied. Original energy: {emergency_state.get('original_energy', 0)}")
        
        return emergency_state
    
    def _standard_renormalization(self, state: Dict[str, Any], current_energy: float) -> Dict[str, Any]:
        """Apply standard renormalization scaling."""
        renormalized_state = state.copy()
        
        # Calculate renormalization factor
        target_energy = self.bounds.energy_threshold * 0.8
        renorm_factor = np.sqrt(target_energy / current_energy)
        
        # Scale confidence towards 0.5
        if 'confidence' in renormalized_state:
            conf = renormalized_state['confidence']
            renormalized_state['confidence'] = 0.5 + (conf - 0.5) * renorm_factor
        
        # Scale feature values
        if 'features' in renormalized_state:
            for feature, value in renormalized_state['features'].items():
                if isinstance(value, (int, float)):
                    renormalized_state['features'][feature] = value * renorm_factor
        
        # Reduce number of hypotheses if too many
        if 'hypotheses' in renormalized_state and len(renormalized_state['hypotheses']) > 10:
            # Keep only the highest confidence hypotheses
            hypotheses = renormalized_state['hypotheses']
            if all('confidence' in h for h in hypotheses):
                hypotheses.sort(key=lambda x: x.get('confidence', 0), reverse=True)
                renormalized_state['hypotheses'] = hypotheses[:10]
        
        renormalized_state['renormalization_applied'] = True
        renormalized_state['renormalization_factor'] = renorm_factor
        
        return renormalized_state
    
    def _bound_confidence_values(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all confidence values are within safe bounds."""
        bounded_state = state.copy()
        
        # Bound main confidence
        if 'confidence' in bounded_state:
            conf = bounded_state['confidence']
            bounded_state['confidence'] = np.clip(conf, self.bounds.min_confidence, self.bounds.max_confidence)
        
        # Bound hypothesis confidences
        if 'hypotheses' in bounded_state:
            for hypothesis in bounded_state['hypotheses']:
                if 'confidence' in hypothesis:
                    hypothesis['confidence'] = np.clip(
                        hypothesis['confidence'], 
                        self.bounds.min_confidence, 
                        self.bounds.max_confidence
                    )
        
        return bounded_state
    
    def _limit_recursion_depth(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Limit recursion depth to prevent infinite loops."""
        limited_state = state.copy()
        
        recursion_depth = limited_state.get('recursion_depth', 0)
        if recursion_depth > self.bounds.max_recursion_depth:
            limited_state['recursion_depth'] = self.bounds.max_recursion_depth
            limited_state['recursion_limited'] = True
            self.logger.warning(f"Recursion depth limited to {self.bounds.max_recursion_depth}")
        
        return limited_state
    
    def get_safety_metrics(self) -> Dict[str, Any]:
        """Get current safety metrics and statistics."""
        return {
            'current_energy': self.energy_history[-1] if self.energy_history else 0,
            'average_energy': np.mean(self.energy_history) if self.energy_history else 0,
            'max_energy': np.max(self.energy_history) if self.energy_history else 0,
            'renormalization_count': self.renormalization_count,
            'emergency_stops': self.emergency_stops,
            'energy_history_length': len(self.energy_history),
            'bounds': {
                'min_confidence': self.bounds.min_confidence,
                'max_confidence': self.bounds.max_confidence,
                'max_recursion_depth': self.bounds.max_recursion_depth,
                'energy_threshold': self.bounds.energy_threshold,
                'divergence_threshold': self.bounds.divergence_threshold
            }
        }

class KappaBlockLogic:
    """
    Implements κ-block logic for safe logical reasoning.
    Prevents logical contradictions and maintains consistency.
    """
    
    def __init__(self, kappa: float = 0.8):
        self.kappa = kappa  # Consistency threshold
        self.logical_blocks = []
        self.contradiction_count = 0
        self.consistency_history = []
        self.logger = logging.getLogger(__name__)
    
    def create_logical_block(self, premises: List[Dict[str, Any]], 
                           conclusion: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a κ-block containing premises and conclusion.
        
        Args:
            premises: List of premise statements
            conclusion: Conclusion statement
            
        Returns:
            κ-block with consistency guarantees
        """
        block = {
            'id': len(self.logical_blocks),
            'premises': premises,
            'conclusion': conclusion,
            'consistency_score': self._calculate_block_consistency(premises, conclusion),
            'timestamp': np.datetime64('now'),
            'valid': True
        }
        
        # Check if block meets κ-consistency threshold
        if block['consistency_score'] < self.kappa:
            block['valid'] = False
            block['rejection_reason'] = f"Consistency score {block['consistency_score']:.3f} below κ threshold {self.kappa}"
            self.logger.warning(f"κ-block rejected: {block['rejection_reason']}")
        
        self.logical_blocks.append(block)
        self.consistency_history.append(block['consistency_score'])
        
        return block
    
    def validate_logical_chain(self, logical_chain: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Validate a chain of logical reasoning for consistency.
        
        Args:
            logical_chain: List of logical statements in sequence
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if len(logical_chain) < 2:
            return True, issues
        
        # Check pairwise consistency
        for i in range(len(logical_chain) - 1):
            current = logical_chain[i]
            next_statement = logical_chain[i + 1]
            
            consistency = self._check_pairwise_consistency(current, next_statement)
            if consistency < self.kappa:
                issues.append(f"Low consistency ({consistency:.3f}) between statements {i} and {i+1}")
        
        # Check for circular reasoning
        if self._detect_circular_reasoning(logical_chain):
            issues.append("Circular reasoning detected in logical chain")
        
        # Check for contradictions
        contradictions = self._find_contradictions(logical_chain)
        if contradictions:
            issues.extend([f"Contradiction found: {c}" for c in contradictions])
            self.contradiction_count += len(contradictions)
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def _calculate_block_consistency(self, premises: List[Dict[str, Any]], 
                                   conclusion: Dict[str, Any]) -> float:
        """Calculate consistency score for a logical block."""
        if not premises:
            return 0.5  # Neutral consistency for empty premises
        
        consistency_scores = []
        
        # Check consistency between each premise and conclusion
        for premise in premises:
            score = self._check_statement_consistency(premise, conclusion)
            consistency_scores.append(score)
        
        # Check internal consistency of premises
        for i, premise1 in enumerate(premises):
            for premise2 in premises[i+1:]:
                score = self._check_statement_consistency(premise1, premise2)
                consistency_scores.append(score)
        
        return np.mean(consistency_scores) if consistency_scores else 0.5
    
    def _check_statement_consistency(self, statement1: Dict[str, Any], 
                                   statement2: Dict[str, Any]) -> float:
        """Check consistency between two logical statements."""
        consistency = 0.5  # Default neutral consistency
        
        # Check confidence compatibility
        if 'confidence' in statement1 and 'confidence' in statement2:
            conf1, conf2 = statement1['confidence'], statement2['confidence']
            # High consistency if confidences are similar
            conf_similarity = 1.0 - abs(conf1 - conf2)
            consistency += 0.3 * conf_similarity
        
        # Check feature compatibility
        if 'features' in statement1 and 'features' in statement2:
            common_features = set(statement1['features'].keys()) & set(statement2['features'].keys())
            if common_features:
                feature_consistency = 0
                for feature in common_features:
                    val1 = statement1['features'][feature]
                    val2 = statement2['features'][feature]
                    
                    if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                        # Numerical consistency
                        max_val = max(abs(val1), abs(val2), 1.0)
                        similarity = 1.0 - abs(val1 - val2) / max_val
                        feature_consistency += similarity
                    elif val1 == val2:
                        feature_consistency += 1.0
                
                consistency += 0.4 * (feature_consistency / len(common_features))
        
        # Check logical relationship
        if 'conclusion' in statement1 and 'conclusion' in statement2:
            conclusion1 = statement1['conclusion'].lower()
            conclusion2 = statement2['conclusion'].lower()
            
            # Check for direct contradiction
            if self._are_contradictory_conclusions(conclusion1, conclusion2):
                consistency = 0.0
            elif conclusion1 == conclusion2:
                consistency += 0.3
        
        return np.clip(consistency, 0.0, 1.0)
    
    def _check_pairwise_consistency(self, statement1: Dict[str, Any], 
                                  statement2: Dict[str, Any]) -> float:
        """Check consistency between two adjacent statements in a chain."""
        return self._check_statement_consistency(statement1, statement2)
    
    def _detect_circular_reasoning(self, logical_chain: List[Dict[str, Any]]) -> bool:
        """Detect circular reasoning in a logical chain."""
        if len(logical_chain) < 3:
            return False
        
        # Simple circular reasoning detection
        # Check if any conclusion appears as a premise later in the chain
        conclusions = set()
        
        for i, statement in enumerate(logical_chain):
            if 'conclusion' in statement:
                conclusion = statement['conclusion'].lower().strip()
                
                # Check if this conclusion was seen before
                if conclusion in conclusions:
                    return True
                
                conclusions.add(conclusion)
                
                # Check if this conclusion matches any previous premise
                for j in range(i):
                    prev_statement = logical_chain[j]
                    if 'premises' in prev_statement:
                        for premise in prev_statement['premises']:
                            if isinstance(premise, dict) and 'text' in premise:
                                premise_text = premise['text'].lower().strip()
                                if premise_text == conclusion:
                                    return True
        
        return False
    
    def _find_contradictions(self, logical_chain: List[Dict[str, Any]]) -> List[str]:
        """Find contradictions in a logical chain."""
        contradictions = []
        
        for i, statement1 in enumerate(logical_chain):
            for j, statement2 in enumerate(logical_chain[i+1:], i+1):
                if self._are_contradictory_statements(statement1, statement2):
                    contradictions.append(f"Statements {i} and {j} are contradictory")
        
        return contradictions
    
    def _are_contradictory_statements(self, statement1: Dict[str, Any], 
                                    statement2: Dict[str, Any]) -> bool:
        """Check if two statements are contradictory."""
        # Check conclusions
        if 'conclusion' in statement1 and 'conclusion' in statement2:
            conclusion1 = statement1['conclusion'].lower()
            conclusion2 = statement2['conclusion'].lower()
            
            if self._are_contradictory_conclusions(conclusion1, conclusion2):
                return True
        
        # Check features for contradictory values
        if 'features' in statement1 and 'features' in statement2:
            common_features = set(statement1['features'].keys()) & set(statement2['features'].keys())
            for feature in common_features:
                val1 = statement1['features'][feature]
                val2 = statement2['features'][feature]
                
                # Boolean contradiction
                if isinstance(val1, bool) and isinstance(val2, bool) and val1 != val2:
                    return True
                
                # Extreme numerical contradiction (values differ by orders of magnitude)
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    if val1 != 0 and val2 != 0:
                        ratio = abs(val1 / val2)
                        if ratio > 1000 or ratio < 0.001:  # Orders of magnitude difference
                            return True
        
        return False
    
    def _are_contradictory_conclusions(self, conclusion1: str, conclusion2: str) -> bool:
        """Check if two conclusion strings are contradictory."""
        # Simple negation detection
        if 'not' in conclusion1 and conclusion1.replace('not ', '').strip() == conclusion2:
            return True
        if 'not' in conclusion2 and conclusion2.replace('not ', '').strip() == conclusion1:
            return True
        
        # Antonym detection (basic)
        antonym_pairs = [
            ('true', 'false'), ('valid', 'invalid'), ('correct', 'incorrect'),
            ('possible', 'impossible'), ('likely', 'unlikely'), ('safe', 'dangerous')
        ]
        
        for word1, word2 in antonym_pairs:
            if word1 in conclusion1 and word2 in conclusion2:
                return True
            if word2 in conclusion1 and word1 in conclusion2:
                return True
        
        return False
    
    def get_kappa_metrics(self) -> Dict[str, Any]:
        """Get κ-block logic metrics and statistics."""
        valid_blocks = [b for b in self.logical_blocks if b['valid']]
        
        return {
            'kappa_threshold': self.kappa,
            'total_blocks': len(self.logical_blocks),
            'valid_blocks': len(valid_blocks),
            'rejection_rate': 1.0 - len(valid_blocks) / max(len(self.logical_blocks), 1),
            'average_consistency': np.mean(self.consistency_history) if self.consistency_history else 0,
            'contradiction_count': self.contradiction_count,
            'consistency_trend': np.polyfit(range(len(self.consistency_history)), 
                                          self.consistency_history, 1)[0] if len(self.consistency_history) > 1 else 0
        }
