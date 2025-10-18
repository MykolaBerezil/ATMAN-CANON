#!/usr/bin/env python3
"""
ATMAN-CANON Evidence Calculator
Advanced reasoning system for evaluating evidence quality, relevance, and reliability
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import json
import logging
from datetime import datetime

class EvidenceCalculator:
    """
    Core evidence evaluation engine for ATMAN-CANON framework
    Implements Bayesian evidence assessment with uncertainty quantification
    """
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize the Evidence Calculator
        
        Args:
            confidence_threshold: Minimum confidence level for evidence acceptance
        """
        self.confidence_threshold = confidence_threshold
        self.evidence_history = []
        self.logger = logging.getLogger(__name__)
        
        # Evidence quality metrics
        self.quality_weights = {
            'source_reliability': 0.25,
            'data_completeness': 0.20,
            'temporal_relevance': 0.15,
            'cross_validation': 0.20,
            'uncertainty_bounds': 0.20
        }
        
    def evaluate_evidence(self, evidence: Dict[str, Any]) -> Dict[str, float]:
        """
        Comprehensive evidence evaluation using multiple quality metrics
        
        Args:
            evidence: Dictionary containing evidence data and metadata
            
        Returns:
            Dictionary with evaluation scores and overall confidence
        """
        try:
            # Extract evidence components
            data = evidence.get('data', {})
            metadata = evidence.get('metadata', {})
            source_info = evidence.get('source', {})
            
            # Calculate individual quality metrics
            source_score = self._evaluate_source_reliability(source_info)
            completeness_score = self._evaluate_data_completeness(data)
            temporal_score = self._evaluate_temporal_relevance(metadata)
            validation_score = self._evaluate_cross_validation(evidence)
            uncertainty_score = self._evaluate_uncertainty_bounds(data)
            
            # Weighted overall confidence
            overall_confidence = (
                source_score * self.quality_weights['source_reliability'] +
                completeness_score * self.quality_weights['data_completeness'] +
                temporal_score * self.quality_weights['temporal_relevance'] +
                validation_score * self.quality_weights['cross_validation'] +
                uncertainty_score * self.quality_weights['uncertainty_bounds']
            )
            
            # Evidence evaluation result
            evaluation = {
                'source_reliability': source_score,
                'data_completeness': completeness_score,
                'temporal_relevance': temporal_score,
                'cross_validation': validation_score,
                'uncertainty_bounds': uncertainty_score,
                'overall_confidence': overall_confidence,
                'accepted': overall_confidence >= self.confidence_threshold,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Store in history
            self.evidence_history.append({
                'evidence_id': evidence.get('id', f"evidence_{len(self.evidence_history)}"),
                'evaluation': evaluation,
                'raw_evidence': evidence
            })
            
            self.logger.info(f"Evidence evaluated: confidence={overall_confidence:.3f}, accepted={evaluation['accepted']}")
            
            return evaluation
            
        except Exception as e:
            self.logger.error(f"Evidence evaluation failed: {e}")
            return self._default_evaluation(error=str(e))
    
    def _evaluate_source_reliability(self, source_info: Dict[str, Any]) -> float:
        """Evaluate the reliability of the evidence source"""
        if not source_info:
            return 0.1
        
        # Source reliability factors
        factors = {
            'peer_reviewed': source_info.get('peer_reviewed', False),
            'institutional_affiliation': source_info.get('institution', '') != '',
            'citation_count': min(source_info.get('citations', 0) / 100.0, 1.0),
            'author_reputation': source_info.get('author_h_index', 0) / 50.0,
            'publication_venue': source_info.get('venue_impact_factor', 0) / 10.0
        }
        
        # Weighted reliability score
        weights = [0.3, 0.2, 0.2, 0.15, 0.15]
        score = sum(w * (1.0 if isinstance(factors[k], bool) and factors[k] else min(factors[k], 1.0) if not isinstance(factors[k], bool) else 0.0)
                   for w, k in zip(weights, factors.keys()))
        
        return min(score, 1.0)
    
    def _evaluate_data_completeness(self, data: Dict[str, Any]) -> float:
        """Evaluate completeness and quality of the data"""
        if not data:
            return 0.0
        
        # Data quality indicators
        completeness_factors = {
            'missing_values': 1.0 - (data.get('missing_count', 0) / max(data.get('total_count', 1), 1)),
            'data_variance': min(data.get('variance', 0) / data.get('expected_variance', 1), 1.0),
            'sample_size': min(data.get('sample_size', 0) / 1000.0, 1.0),
            'measurement_precision': data.get('precision', 0.5),
            'data_consistency': data.get('consistency_score', 0.5)
        }
        
        return np.mean(list(completeness_factors.values()))
    
    def _evaluate_temporal_relevance(self, metadata: Dict[str, Any]) -> float:
        """Evaluate temporal relevance of the evidence"""
        if not metadata.get('timestamp'):
            return 0.5
        
        try:
            evidence_time = datetime.fromisoformat(metadata['timestamp'].replace('Z', '+00:00'))
            current_time = datetime.utcnow().replace(tzinfo=evidence_time.tzinfo)
            age_days = (current_time - evidence_time).days
            
            # Decay function for temporal relevance
            half_life = metadata.get('relevance_half_life', 365)  # days
            temporal_score = np.exp(-0.693 * age_days / half_life)
            
            return min(temporal_score, 1.0)
            
        except Exception:
            return 0.5
    
    def _evaluate_cross_validation(self, evidence: Dict[str, Any]) -> float:
        """Evaluate cross-validation and corroboration"""
        validation_sources = evidence.get('validation', {})
        
        if not validation_sources:
            return 0.3  # Default for single source
        
        # Cross-validation factors
        independent_sources = len(validation_sources.get('independent_sources', []))
        replication_studies = len(validation_sources.get('replications', []))
        consensus_level = validation_sources.get('consensus_score', 0.5)
        
        # Calculate validation score
        source_score = min(independent_sources / 3.0, 1.0)  # Optimal at 3+ sources
        replication_score = min(replication_studies / 2.0, 1.0)  # Optimal at 2+ replications
        
        validation_score = (source_score * 0.4 + replication_score * 0.3 + consensus_level * 0.3)
        
        return min(validation_score, 1.0)
    
    def _evaluate_uncertainty_bounds(self, data: Dict[str, Any]) -> float:
        """Evaluate uncertainty quantification and bounds"""
        if not data:
            return 0.0
        
        # Uncertainty indicators
        confidence_interval = data.get('confidence_interval', {})
        error_bars = data.get('error_margins', {})
        statistical_power = data.get('statistical_power', 0.5)
        
        # Uncertainty quality score
        ci_quality = 1.0 if confidence_interval else 0.0
        error_quality = 1.0 if error_bars else 0.0
        power_quality = statistical_power
        
        uncertainty_score = (ci_quality * 0.4 + error_quality * 0.3 + power_quality * 0.3)
        
        return min(uncertainty_score, 1.0)
    
    def _default_evaluation(self, error: str = "") -> Dict[str, float]:
        """Return default evaluation for error cases"""
        return {
            'source_reliability': 0.0,
            'data_completeness': 0.0,
            'temporal_relevance': 0.0,
            'cross_validation': 0.0,
            'uncertainty_bounds': 0.0,
            'overall_confidence': 0.0,
            'accepted': False,
            'error': error,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def get_evidence_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all evaluated evidence"""
        if not self.evidence_history:
            return {'total_evidence': 0, 'accepted_count': 0, 'average_confidence': 0.0}
        
        evaluations = [item['evaluation'] for item in self.evidence_history]
        accepted_count = sum(1 for eval in evaluations if eval['accepted'])
        avg_confidence = np.mean([eval['overall_confidence'] for eval in evaluations])
        
        return {
            'total_evidence': len(self.evidence_history),
            'accepted_count': accepted_count,
            'rejection_count': len(self.evidence_history) - accepted_count,
            'acceptance_rate': accepted_count / len(self.evidence_history),
            'average_confidence': avg_confidence,
            'confidence_threshold': self.confidence_threshold
        }
    
    def export_evidence_log(self, filepath: str) -> bool:
        """Export evidence evaluation history to JSON file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.evidence_history, f, indent=2, default=str)
            return True
        except Exception as e:
            self.logger.error(f"Failed to export evidence log: {e}")
            return False

# Example usage and testing
if __name__ == "__main__":
    # Initialize evidence calculator
    calculator = EvidenceCalculator(confidence_threshold=0.7)
    
    # Example evidence evaluation
    sample_evidence = {
        'id': 'evidence_001',
        'data': {
            'sample_size': 1000,
            'missing_count': 50,
            'total_count': 1000,
            'variance': 0.25,
            'expected_variance': 0.30,
            'precision': 0.95,
            'consistency_score': 0.88
        },
        'metadata': {
            'timestamp': '2025-01-15T10:00:00Z',
            'relevance_half_life': 180
        },
        'source': {
            'peer_reviewed': True,
            'institution': 'MIT',
            'citations': 150,
            'author_h_index': 25,
            'venue_impact_factor': 8.5
        },
        'validation': {
            'independent_sources': ['source_a', 'source_b', 'source_c'],
            'replications': ['study_1', 'study_2'],
            'consensus_score': 0.85
        }
    }
    
    # Evaluate evidence
    result = calculator.evaluate_evidence(sample_evidence)
    print("Evidence Evaluation Result:")
    print(json.dumps(result, indent=2))
    
    # Get summary
    summary = calculator.get_evidence_summary()
    print("\nEvidence Summary:")
    print(json.dumps(summary, indent=2))
