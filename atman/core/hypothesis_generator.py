#!/usr/bin/env python3
"""
ATMAN-CANON Hypothesis Generator
AI-driven hypothesis generation engine for creating testable theories
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import json
import logging
from datetime import datetime
import itertools
from collections import defaultdict

class HypothesisGenerator:
    """
    Advanced hypothesis generation system for ATMAN-CANON framework
    Creates testable hypotheses based on evidence patterns and domain knowledge
    """
    
    def __init__(self, creativity_factor: float = 0.7, max_hypotheses: int = 10):
        """
        Initialize the Hypothesis Generator
        
        Args:
            creativity_factor: Balance between conservative and creative hypothesis generation (0-1)
            max_hypotheses: Maximum number of hypotheses to generate per request
        """
        self.creativity_factor = creativity_factor
        self.max_hypotheses = max_hypotheses
        self.hypothesis_history = []
        self.domain_knowledge = {}
        self.pattern_library = {}
        self.logger = logging.getLogger(__name__)
        
        # Hypothesis generation strategies
        self.generation_strategies = {
            'causal_inference': 0.25,
            'pattern_extrapolation': 0.20,
            'analogical_reasoning': 0.20,
            'contradiction_resolution': 0.15,
            'emergent_property_detection': 0.20
        }
        
    def generate_hypotheses(self, evidence_set: List[Dict[str, Any]], 
                          domain_context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Generate hypotheses based on provided evidence and domain context
        
        Args:
            evidence_set: List of evidence items with evaluations
            domain_context: Optional domain-specific context and constraints
            
        Returns:
            List of generated hypotheses with confidence scores and testability metrics
        """
        try:
            # Preprocess evidence for pattern detection
            processed_evidence = self._preprocess_evidence(evidence_set)
            
            # Update domain knowledge if context provided
            if domain_context:
                self._update_domain_knowledge(domain_context)
            
            # Generate hypotheses using multiple strategies
            hypotheses = []
            
            # Causal inference hypotheses
            causal_hypotheses = self._generate_causal_hypotheses(processed_evidence)
            hypotheses.extend(causal_hypotheses)
            
            # Pattern extrapolation hypotheses
            pattern_hypotheses = self._generate_pattern_hypotheses(processed_evidence)
            hypotheses.extend(pattern_hypotheses)
            
            # Analogical reasoning hypotheses
            analogical_hypotheses = self._generate_analogical_hypotheses(processed_evidence)
            hypotheses.extend(analogical_hypotheses)
            
            # Contradiction resolution hypotheses
            contradiction_hypotheses = self._generate_contradiction_hypotheses(processed_evidence)
            hypotheses.extend(contradiction_hypotheses)
            
            # Emergent property hypotheses
            emergent_hypotheses = self._generate_emergent_hypotheses(processed_evidence)
            hypotheses.extend(emergent_hypotheses)
            
            # Rank and filter hypotheses
            ranked_hypotheses = self._rank_hypotheses(hypotheses, processed_evidence)
            final_hypotheses = ranked_hypotheses[:self.max_hypotheses]
            
            # Store in history
            generation_record = {
                'timestamp': datetime.utcnow().isoformat(),
                'evidence_count': len(evidence_set),
                'hypotheses_generated': len(hypotheses),
                'final_hypotheses': final_hypotheses,
                'domain_context': domain_context
            }
            self.hypothesis_history.append(generation_record)
            
            self.logger.info(f"Generated {len(final_hypotheses)} hypotheses from {len(evidence_set)} evidence items")
            
            return final_hypotheses
            
        except Exception as e:
            self.logger.error(f"Hypothesis generation failed: {e}")
            return []
    
    def _preprocess_evidence(self, evidence_set: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Preprocess evidence for pattern detection and analysis"""
        processed = {
            'variables': set(),
            'relationships': [],
            'temporal_patterns': [],
            'statistical_patterns': [],
            'contradictions': [],
            'confidence_levels': []
        }
        
        for evidence in evidence_set:
            # Extract variables and relationships
            data = evidence.get('data', {})
            evaluation = evidence.get('evaluation', {})
            
            # Collect variables
            if 'variables' in data:
                processed['variables'].update(data['variables'])
            
            # Detect relationships
            if 'correlations' in data:
                for correlation in data['correlations']:
                    processed['relationships'].append({
                        'type': 'correlation',
                        'variables': correlation.get('variables', []),
                        'strength': correlation.get('strength', 0.0),
                        'confidence': evaluation.get('overall_confidence', 0.0)
                    })
            
            # Temporal patterns
            if 'temporal_data' in data:
                processed['temporal_patterns'].append({
                    'pattern': data['temporal_data'],
                    'confidence': evaluation.get('overall_confidence', 0.0)
                })
            
            # Statistical patterns
            if 'statistics' in data:
                processed['statistical_patterns'].append({
                    'stats': data['statistics'],
                    'confidence': evaluation.get('overall_confidence', 0.0)
                })
            
            # Track confidence levels
            processed['confidence_levels'].append(evaluation.get('overall_confidence', 0.0))
        
        # Convert variables set to list
        processed['variables'] = list(processed['variables'])
        
        return processed
    
    def _generate_causal_hypotheses(self, processed_evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate hypotheses based on causal inference"""
        hypotheses = []
        relationships = processed_evidence['relationships']
        variables = processed_evidence['variables']
        
        # Generate causal hypotheses from correlations
        for rel in relationships:
            if rel['type'] == 'correlation' and rel['strength'] > 0.5:
                vars_involved = rel['variables']
                if len(vars_involved) >= 2:
                    # Generate bidirectional causal hypotheses
                    for i, cause in enumerate(vars_involved):
                        for j, effect in enumerate(vars_involved):
                            if i != j:
                                hypothesis = {
                                    'id': f"causal_{len(hypotheses)}",
                                    'type': 'causal_inference',
                                    'statement': f"{cause} causally influences {effect}",
                                    'variables': {'cause': cause, 'effect': effect},
                                    'confidence': rel['confidence'] * 0.8,  # Reduce confidence for causal claim
                                    'testability': self._calculate_testability(cause, effect, 'causal'),
                                    'novelty': self._calculate_novelty('causal', [cause, effect]),
                                    'evidence_support': rel['strength']
                                }
                                hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _generate_pattern_hypotheses(self, processed_evidence: Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate hypotheses based on pattern extrapolation"""
        hypotheses = []
        temporal_patterns = processed_evidence['temporal_patterns']
        
        for pattern_data in temporal_patterns:
            pattern = pattern_data['pattern']
            confidence = pattern_data['confidence']
            
            # Detect trend patterns
            if 'trend' in pattern:
                trend = pattern['trend']
                hypothesis = {
                    'id': f"pattern_{len(hypotheses)}",
                    'type': 'pattern_extrapolation',
                    'statement': f"The observed {trend} trend will continue in the future",
                    'variables': {'trend_type': trend, 'temporal_scope': 'future'},
                    'confidence': confidence * 0.7,  # Reduce for extrapolation uncertainty
                    'testability': self._calculate_testability(trend, 'future_observation', 'temporal'),
                    'novelty': self._calculate_novelty('pattern', [trend]),
                    'evidence_support': confidence
                }
                hypotheses.append(hypothesis)
            
            # Detect cyclical patterns
            if 'cycle' in pattern:
                cycle = pattern['cycle']
                hypothesis = {
                    'id': f"pattern_{len(hypotheses)}",
                    'type': 'pattern_extrapolation',
                    'statement': f"A cyclical pattern with period {cycle.get('period', 'unknown')} exists",
                    'variables': {'cycle_period': cycle.get('period'), 'cycle_amplitude': cycle.get('amplitude')},
                    'confidence': confidence * 0.75,
                    'testability': self._calculate_testability('cycle', 'periodic_observation', 'temporal'),
                    'novelty': self._calculate_novelty('pattern', [cycle]),
                    'evidence_support': confidence
                }
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _generate_analogical_hypotheses(self, processed_evidence: Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate hypotheses based on analogical reasoning"""
        hypotheses = []
        variables = processed_evidence['variables']
        
        # Use domain knowledge for analogical reasoning
        for domain, knowledge in self.domain_knowledge.items():
            analogies = knowledge.get('analogies', [])
            
            for analogy in analogies:
                source_domain = analogy.get('source')
                target_domain = analogy.get('target')
                mapping = analogy.get('mapping', {})
                
                # Check if current variables match analogy pattern
                if any(var in str(source_domain) for var in variables):
                    hypothesis = {
                        'id': f"analogical_{len(hypotheses)}",
                        'type': 'analogical_reasoning',
                        'statement': f"By analogy with {source_domain}, {target_domain} should exhibit similar patterns",
                        'variables': {'source': source_domain, 'target': target_domain, 'mapping': mapping},
                        'confidence': 0.6 * self.creativity_factor,  # Moderate confidence for analogies
                        'testability': self._calculate_testability(source_domain, target_domain, 'analogical'),
                        'novelty': self._calculate_novelty('analogical', [source_domain, target_domain]),
                        'evidence_support': 0.5
                    }
                    hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _generate_contradiction_hypotheses(self, processed_evidence: Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate hypotheses to resolve contradictions in evidence"""
        hypotheses = []
        relationships = processed_evidence['relationships']
        
        # Find contradictory relationships
        contradictions = self._detect_contradictions(relationships)
        
        for contradiction in contradictions:
            rel1, rel2 = contradiction['relationships']
            
            # Generate resolution hypotheses
            resolution_hypothesis = {
                'id': f"contradiction_{len(hypotheses)}",
                'type': 'contradiction_resolution',
                'statement': f"The apparent contradiction between {rel1['variables']} and {rel2['variables']} can be resolved by considering mediating factors",
                'variables': {
                    'contradictory_evidence': [rel1, rel2],
                    'proposed_mediators': self._suggest_mediators(rel1, rel2)
                },
                'confidence': 0.5,  # Moderate confidence for resolution attempts
                'testability': self._calculate_testability(rel1, rel2, 'contradiction'),
                'novelty': self._calculate_novelty('contradiction', [rel1, rel2]),
                'evidence_support': min(rel1['confidence'], rel2['confidence'])
            }
            hypotheses.append(resolution_hypothesis)
        
        return hypotheses
    
    def _generate_emergent_hypotheses(self, processed_evidence: Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate hypotheses about emergent properties"""
        hypotheses = []
        variables = processed_evidence['variables']
        relationships = processed_evidence['relationships']
        
        # Look for potential emergent properties
        if len(variables) >= 3 and len(relationships) >= 2:
            # Generate emergent property hypothesis
            hypothesis = {
                'id': f"emergent_{len(hypotheses)}",
                'type': 'emergent_property_detection',
                'statement': f"The interaction between {variables[:3]} may produce emergent properties not present in individual components",
                'variables': {
                    'interacting_components': variables[:3],
                    'potential_emergent_properties': self._predict_emergent_properties(variables[:3])
                },
                'confidence': 0.4 + (0.3 * self.creativity_factor),  # Creative confidence
                'testability': self._calculate_testability(variables[:3], 'emergent_behavior', 'emergent'),
                'novelty': self._calculate_novelty('emergent', variables[:3]),
                'evidence_support': np.mean(processed_evidence['confidence_levels'])
            }
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _detect_contradictions(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect contradictory relationships in evidence"""
        contradictions = []
        
        for i, rel1 in enumerate(relationships):
            for j, rel2 in enumerate(relationships[i+1:], i+1):
                # Check for contradictory relationships
                if (set(rel1['variables']) & set(rel2['variables']) and
                    abs(rel1['strength'] - rel2['strength']) > 0.7):
                    contradictions.append({
                        'relationships': [rel1, rel2],
                        'contradiction_strength': abs(rel1['strength'] - rel2['strength'])
                    })
        
        return contradictions
    
    def _suggest_mediators(self, rel1: Dict[str, Any], rel2: Dict[str, Any]) -> List[str]:
        """Suggest potential mediating variables for contradiction resolution"""
        # Simple heuristic for mediator suggestion
        common_vars = set(rel1['variables']) & set(rel2['variables'])
        mediators = [f"mediator_for_{var}" for var in common_vars]
        mediators.extend(['temporal_factor', 'contextual_factor', 'measurement_error'])
        return mediators[:3]  # Limit to top 3 suggestions
    
    def _predict_emergent_properties(self, components: List[str]) -> List[str]:
        """Predict potential emergent properties from component interactions"""
        emergent_properties = [
            f"synergistic_effect_of_{components[0]}_and_{components[1]}",
            f"nonlinear_interaction_between_{components}",
            f"collective_behavior_of_{len(components)}_components",
            "phase_transition_phenomenon",
            "self_organization_pattern"
        ]
        return emergent_properties[:3]  # Return top 3 predictions
    
    def _calculate_testability(self, var1: Any, var2: Any, hypothesis_type: str) -> float:
        """Calculate testability score for a hypothesis"""
        base_testability = {
            'causal': 0.8,
            'temporal': 0.7,
            'analogical': 0.5,
            'contradiction': 0.6,
            'emergent': 0.4
        }
        
        # Adjust based on variable complexity
        complexity_penalty = 0.1 * (len(str(var1)) + len(str(var2))) / 50.0
        testability = base_testability.get(hypothesis_type, 0.5) - complexity_penalty
        
        return max(0.1, min(1.0, testability))
    
    def _calculate_novelty(self, hypothesis_type: str, variables: List[Any]) -> float:
        """Calculate novelty score for a hypothesis"""
        # Check against historical hypotheses
        historical_patterns = [h['final_hypotheses'] for h in self.hypothesis_history]
        
        novelty_score = 0.8  # Base novelty
        
        # Reduce novelty if similar hypotheses exist
        for history in historical_patterns:
            for past_hypothesis in history:
                if (past_hypothesis.get('type') == hypothesis_type and
                    set(str(v) for v in variables) & set(str(v) for v in past_hypothesis.get('variables', {}).values())):
                    novelty_score *= 0.8
        
        return max(0.1, novelty_score)
    
    def _rank_hypotheses(self, hypotheses: List[Dict[str, Any]], 
                        processed_evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank hypotheses by overall quality score"""
        for hypothesis in hypotheses:
            # Calculate composite quality score
            quality_score = (
                hypothesis['confidence'] * 0.3 +
                hypothesis['testability'] * 0.3 +
                hypothesis['novelty'] * 0.2 +
                hypothesis['evidence_support'] * 0.2
            )
            hypothesis['quality_score'] = quality_score
        
        # Sort by quality score (descending)
        return sorted(hypotheses, key=lambda h: h['quality_score'], reverse=True)
    
    def _update_domain_knowledge(self, domain_context: Dict[str, Any]) -> None:
        """Update internal domain knowledge base"""
        domain_name = domain_context.get('domain', 'general')
        self.domain_knowledge[domain_name] = domain_context
    
    def get_generation_summary(self) -> Dict[str, Any]:
        """Get summary statistics of hypothesis generation history"""
        if not self.hypothesis_history:
            return {'total_generations': 0, 'total_hypotheses': 0, 'average_quality': 0.0}
        
        total_hypotheses = sum(len(record['final_hypotheses']) for record in self.hypothesis_history)
        
        # Calculate average quality across all hypotheses
        all_qualities = []
        for record in self.hypothesis_history:
            all_qualities.extend([h.get('quality_score', 0.0) for h in record['final_hypotheses']])
        
        avg_quality = np.mean(all_qualities) if all_qualities else 0.0
        
        return {
            'total_generations': len(self.hypothesis_history),
            'total_hypotheses': total_hypotheses,
            'average_hypotheses_per_generation': total_hypotheses / len(self.hypothesis_history),
            'average_quality': avg_quality,
            'creativity_factor': self.creativity_factor
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize hypothesis generator
    generator = HypothesisGenerator(creativity_factor=0.7, max_hypotheses=5)
    
    # Example evidence set
    sample_evidence = [
        {
            'data': {
                'variables': ['temperature', 'pressure', 'volume'],
                'correlations': [
                    {'variables': ['temperature', 'pressure'], 'strength': 0.8},
                    {'variables': ['pressure', 'volume'], 'strength': -0.7}
                ],
                'temporal_data': {
                    'trend': 'increasing_temperature',
                    'cycle': {'period': 24, 'amplitude': 0.5}
                }
            },
            'evaluation': {'overall_confidence': 0.85}
        },
        {
            'data': {
                'variables': ['humidity', 'temperature'],
                'correlations': [
                    {'variables': ['humidity', 'temperature'], 'strength': 0.6}
                ]
            },
            'evaluation': {'overall_confidence': 0.75}
        }
    ]
    
    # Generate hypotheses
    hypotheses = generator.generate_hypotheses(sample_evidence)
    
    print("Generated Hypotheses:")
    for i, hypothesis in enumerate(hypotheses, 1):
        print(f"\n{i}. {hypothesis['statement']}")
        print(f"   Type: {hypothesis['type']}")
        print(f"   Confidence: {hypothesis['confidence']:.3f}")
        print(f"   Testability: {hypothesis['testability']:.3f}")
        print(f"   Quality Score: {hypothesis['quality_score']:.3f}")
    
    # Get summary
    summary = generator.get_generation_summary()
    print(f"\nGeneration Summary:")
    print(json.dumps(summary, indent=2))
