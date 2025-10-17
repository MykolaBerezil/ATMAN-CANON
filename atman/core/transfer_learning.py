#!/usr/bin/env python3
"""
ATMAN-CANON Transfer Learning Engine
Sophisticated transfer learning system for cross-domain knowledge application
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
import json
import logging
from datetime import datetime
from collections import defaultdict
import pickle

class TransferLearningEngine:
    """
    Advanced transfer learning system for ATMAN-CANON framework
    Applies knowledge across domains and contexts with adaptive similarity metrics
    """
    
    def __init__(self, similarity_threshold: float = 0.6, max_transfer_depth: int = 3):
        """
        Initialize the Transfer Learning Engine
        
        Args:
            similarity_threshold: Minimum similarity for knowledge transfer
            max_transfer_depth: Maximum depth for recursive knowledge transfer
        """
        self.similarity_threshold = similarity_threshold
        self.max_transfer_depth = max_transfer_depth
        self.knowledge_base = {}
        self.domain_mappings = {}
        self.transfer_history = []
        self.learned_patterns = {}
        self.logger = logging.getLogger(__name__)
        
        # Transfer learning strategies
        self.transfer_strategies = {
            'direct_mapping': 0.3,
            'analogical_transfer': 0.25,
            'abstract_pattern_transfer': 0.2,
            'compositional_transfer': 0.15,
            'meta_learning_transfer': 0.1
        }
        
    def learn_from_domain(self, domain_name: str, knowledge_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Learn patterns and knowledge from a specific domain
        
        Args:
            domain_name: Name of the source domain
            knowledge_items: List of knowledge items with features and outcomes
            
        Returns:
            Learning summary with extracted patterns and features
        """
        try:
            # Initialize domain if not exists
            if domain_name not in self.knowledge_base:
                self.knowledge_base[domain_name] = {
                    'patterns': [],
                    'features': set(),
                    'outcomes': [],
                    'relationships': [],
                    'meta_features': {}
                }
            
            domain_kb = self.knowledge_base[domain_name]
            
            # Extract patterns from knowledge items
            extracted_patterns = self._extract_patterns(knowledge_items)
            domain_kb['patterns'].extend(extracted_patterns)
            
            # Extract features
            all_features = set()
            for item in knowledge_items:
                features = item.get('features', {})
                all_features.update(features.keys())
                domain_kb['outcomes'].append(item.get('outcome'))
            
            domain_kb['features'].update(all_features)
            
            # Extract relationships
            relationships = self._extract_relationships(knowledge_items)
            domain_kb['relationships'].extend(relationships)
            
            # Compute meta-features
            meta_features = self._compute_meta_features(knowledge_items)
            domain_kb['meta_features'].update(meta_features)
            
            # Update learned patterns
            self._update_learned_patterns(domain_name, extracted_patterns)
            
            learning_summary = {
                'domain': domain_name,
                'items_processed': len(knowledge_items),
                'patterns_extracted': len(extracted_patterns),
                'features_identified': len(all_features),
                'relationships_found': len(relationships),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"Learned from domain '{domain_name}': {len(extracted_patterns)} patterns, {len(all_features)} features")
            
            return learning_summary
            
        except Exception as e:
            self.logger.error(f"Learning from domain '{domain_name}' failed: {e}")
            return {'error': str(e), 'domain': domain_name}
    
    def transfer_knowledge(self, source_domain: str, target_domain: str, 
                          target_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transfer knowledge from source domain to target domain
        
        Args:
            source_domain: Name of the source domain
            target_domain: Name of the target domain
            target_context: Context and constraints for the target domain
            
        Returns:
            Transfer results with adapted knowledge and confidence scores
        """
        try:
            if source_domain not in self.knowledge_base:
                return {'error': f"Source domain '{source_domain}' not found", 'transferred_items': []}
            
            source_kb = self.knowledge_base[source_domain]
            
            # Initialize target domain if needed
            if target_domain not in self.knowledge_base:
                self.knowledge_base[target_domain] = {
                    'patterns': [], 'features': set(), 'outcomes': [], 
                    'relationships': [], 'meta_features': {}
                }
            
            # Calculate domain similarity
            domain_similarity = self._calculate_domain_similarity(source_domain, target_domain, target_context)
            
            if domain_similarity < self.similarity_threshold:
                self.logger.warning(f"Low similarity ({domain_similarity:.3f}) between domains '{source_domain}' and '{target_domain}'")
            
            # Apply different transfer strategies
            transferred_knowledge = []
            
            # Direct mapping transfer
            direct_transfers = self._direct_mapping_transfer(source_kb, target_context, domain_similarity)
            transferred_knowledge.extend(direct_transfers)
            
            # Analogical transfer
            analogical_transfers = self._analogical_transfer(source_kb, target_context, domain_similarity)
            transferred_knowledge.extend(analogical_transfers)
            
            # Abstract pattern transfer
            pattern_transfers = self._abstract_pattern_transfer(source_kb, target_context, domain_similarity)
            transferred_knowledge.extend(pattern_transfers)
            
            # Compositional transfer
            compositional_transfers = self._compositional_transfer(source_kb, target_context, domain_similarity)
            transferred_knowledge.extend(compositional_transfers)
            
            # Meta-learning transfer
            meta_transfers = self._meta_learning_transfer(source_kb, target_context, domain_similarity)
            transferred_knowledge.extend(meta_transfers)
            
            # Rank and filter transfers
            ranked_transfers = self._rank_transfers(transferred_knowledge, target_context)
            
            # Apply transfers to target domain
            self._apply_transfers_to_domain(target_domain, ranked_transfers)
            
            # Record transfer
            transfer_record = {
                'source_domain': source_domain,
                'target_domain': target_domain,
                'domain_similarity': domain_similarity,
                'transferred_items': len(ranked_transfers),
                'transfer_quality': np.mean([t['confidence'] for t in ranked_transfers]) if ranked_transfers else 0.0,
                'timestamp': datetime.utcnow().isoformat()
            }
            self.transfer_history.append(transfer_record)
            
            self.logger.info(f"Transferred {len(ranked_transfers)} items from '{source_domain}' to '{target_domain}'")
            
            return {
                'source_domain': source_domain,
                'target_domain': target_domain,
                'domain_similarity': domain_similarity,
                'transferred_knowledge': ranked_transfers,
                'transfer_summary': transfer_record
            }
            
        except Exception as e:
            self.logger.error(f"Knowledge transfer failed: {e}")
            return {'error': str(e), 'transferred_items': []}
    
    def _extract_patterns(self, knowledge_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract patterns from knowledge items"""
        patterns = []
        
        # Feature co-occurrence patterns
        feature_cooccurrence = defaultdict(int)
        for item in knowledge_items:
            features = list(item.get('features', {}).keys())
            for i, feat1 in enumerate(features):
                for feat2 in features[i+1:]:
                    feature_cooccurrence[(feat1, feat2)] += 1
        
        # Convert to patterns
        total_items = len(knowledge_items)
        for (feat1, feat2), count in feature_cooccurrence.items():
            if count / total_items > 0.3:  # Significant co-occurrence
                patterns.append({
                    'type': 'feature_cooccurrence',
                    'pattern': {'features': [feat1, feat2], 'frequency': count / total_items},
                    'confidence': count / total_items,
                    'support': count
                })
        
        # Outcome prediction patterns
        feature_outcome_patterns = self._extract_outcome_patterns(knowledge_items)
        patterns.extend(feature_outcome_patterns)
        
        # Sequential patterns
        sequential_patterns = self._extract_sequential_patterns(knowledge_items)
        patterns.extend(sequential_patterns)
        
        return patterns
    
    def _extract_outcome_patterns(self, knowledge_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract patterns that predict outcomes"""
        patterns = []
        
        # Group items by outcome
        outcome_groups = defaultdict(list)
        for item in knowledge_items:
            outcome = item.get('outcome')
            if outcome is not None:
                outcome_groups[outcome].append(item)
        
        # Find discriminative features for each outcome
        for outcome, items in outcome_groups.items():
            if len(items) < 2:
                continue
            
            # Calculate feature importance for this outcome
            feature_counts = defaultdict(int)
            for item in items:
                for feature in item.get('features', {}).keys():
                    feature_counts[feature] += 1
            
            # Find features that are common in this outcome
            total_outcome_items = len(items)
            for feature, count in feature_counts.items():
                if count / total_outcome_items > 0.7:  # Feature appears in 70%+ of cases
                    patterns.append({
                        'type': 'outcome_prediction',
                        'pattern': {'feature': feature, 'outcome': outcome, 'frequency': count / total_outcome_items},
                        'confidence': count / total_outcome_items,
                        'support': count
                    })
        
        return patterns
    
    def _extract_sequential_patterns(self, knowledge_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract sequential patterns from knowledge items"""
        patterns = []
        
        # Look for temporal sequences
        temporal_items = [item for item in knowledge_items if 'sequence' in item]
        
        if len(temporal_items) > 1:
            # Find common subsequences
            sequences = [item['sequence'] for item in temporal_items]
            common_subsequences = self._find_common_subsequences(sequences)
            
            for subseq, frequency in common_subsequences.items():
                if frequency > 1:
                    patterns.append({
                        'type': 'sequential_pattern',
                        'pattern': {'sequence': subseq, 'frequency': frequency / len(temporal_items)},
                        'confidence': frequency / len(temporal_items),
                        'support': frequency
                    })
        
        return patterns
    
    def _find_common_subsequences(self, sequences: List[List[Any]]) -> Dict[Tuple, int]:
        """Find common subsequences across multiple sequences"""
        subsequence_counts = defaultdict(int)
        
        for sequence in sequences:
            # Generate all subsequences of length 2-4
            for length in range(2, min(5, len(sequence) + 1)):
                for i in range(len(sequence) - length + 1):
                    subseq = tuple(sequence[i:i+length])
                    subsequence_counts[subseq] += 1
        
        return subsequence_counts
    
    def _extract_relationships(self, knowledge_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relationships between features and outcomes"""
        relationships = []
        
        # Calculate correlations between features
        feature_values = defaultdict(list)
        for item in knowledge_items:
            features = item.get('features', {})
            for feature, value in features.items():
                if isinstance(value, (int, float)):
                    feature_values[feature].append(value)
        
        # Calculate pairwise correlations
        feature_names = list(feature_values.keys())
        for i, feat1 in enumerate(feature_names):
            for feat2 in feature_names[i+1:]:
                if len(feature_values[feat1]) > 2 and len(feature_values[feat2]) > 2:
                    correlation = np.corrcoef(feature_values[feat1], feature_values[feat2])[0, 1]
                    if abs(correlation) > 0.5:  # Significant correlation
                        relationships.append({
                            'type': 'correlation',
                            'features': [feat1, feat2],
                            'strength': correlation,
                            'sample_size': len(feature_values[feat1])
                        })
        
        return relationships
    
    def _compute_meta_features(self, knowledge_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute meta-features that describe the domain"""
        meta_features = {}
        
        # Domain complexity
        all_features = set()
        for item in knowledge_items:
            all_features.update(item.get('features', {}).keys())
        
        meta_features['feature_count'] = len(all_features)
        meta_features['item_count'] = len(knowledge_items)
        meta_features['average_features_per_item'] = np.mean([len(item.get('features', {})) for item in knowledge_items])
        
        # Outcome diversity
        outcomes = [item.get('outcome') for item in knowledge_items if item.get('outcome') is not None]
        unique_outcomes = set(outcomes)
        meta_features['outcome_diversity'] = len(unique_outcomes)
        meta_features['outcome_distribution'] = dict(zip(*np.unique(outcomes, return_counts=True))) if outcomes else {}
        
        # Feature value distributions
        numeric_features = []
        for item in knowledge_items:
            for value in item.get('features', {}).values():
                if isinstance(value, (int, float)):
                    numeric_features.append(value)
        
        if numeric_features:
            meta_features['numeric_feature_mean'] = np.mean(numeric_features)
            meta_features['numeric_feature_std'] = np.std(numeric_features)
        
        return meta_features
    
    def _update_learned_patterns(self, domain_name: str, patterns: List[Dict[str, Any]]) -> None:
        """Update the global learned patterns database"""
        if domain_name not in self.learned_patterns:
            self.learned_patterns[domain_name] = []
        
        self.learned_patterns[domain_name].extend(patterns)
    
    def _calculate_domain_similarity(self, source_domain: str, target_domain: str, 
                                   target_context: Dict[str, Any]) -> float:
        """Calculate similarity between source and target domains"""
        if source_domain not in self.knowledge_base:
            return 0.0
        
        source_kb = self.knowledge_base[source_domain]
        
        # Feature overlap
        source_features = source_kb['features']
        target_features = set(target_context.get('features', []))
        
        if source_features and target_features:
            feature_overlap = len(source_features & target_features) / len(source_features | target_features)
        else:
            feature_overlap = 0.0
        
        # Meta-feature similarity
        source_meta = source_kb['meta_features']
        target_meta = target_context.get('meta_features', {})
        
        meta_similarity = self._calculate_meta_similarity(source_meta, target_meta)
        
        # Pattern similarity
        pattern_similarity = self._calculate_pattern_similarity(source_domain, target_context)
        
        # Weighted combination
        overall_similarity = (
            feature_overlap * 0.4 +
            meta_similarity * 0.3 +
            pattern_similarity * 0.3
        )
        
        return overall_similarity
    
    def _calculate_meta_similarity(self, source_meta: Dict[str, Any], target_meta: Dict[str, Any]) -> float:
        """Calculate similarity between meta-features"""
        if not source_meta or not target_meta:
            return 0.0
        
        common_keys = set(source_meta.keys()) & set(target_meta.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            source_val = source_meta[key]
            target_val = target_meta[key]
            
            if isinstance(source_val, (int, float)) and isinstance(target_val, (int, float)):
                # Numeric similarity
                max_val = max(abs(source_val), abs(target_val))
                if max_val > 0:
                    similarity = 1.0 - abs(source_val - target_val) / max_val
                else:
                    similarity = 1.0
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_pattern_similarity(self, source_domain: str, target_context: Dict[str, Any]) -> float:
        """Calculate similarity between learned patterns"""
        if source_domain not in self.learned_patterns:
            return 0.0
        
        source_patterns = self.learned_patterns[source_domain]
        target_patterns = target_context.get('patterns', [])
        
        if not source_patterns or not target_patterns:
            return 0.0
        
        # Simple pattern matching based on type and structure
        pattern_matches = 0
        for source_pattern in source_patterns:
            for target_pattern in target_patterns:
                if source_pattern['type'] == target_pattern['type']:
                    pattern_matches += 1
                    break
        
        return pattern_matches / len(source_patterns)
    
    def _direct_mapping_transfer(self, source_kb: Dict[str, Any], target_context: Dict[str, Any], 
                               domain_similarity: float) -> List[Dict[str, Any]]:
        """Direct mapping transfer strategy"""
        transfers = []
        
        # Transfer patterns with high confidence
        for pattern in source_kb['patterns']:
            if pattern['confidence'] > 0.8:
                transfer_confidence = pattern['confidence'] * domain_similarity
                transfers.append({
                    'type': 'direct_mapping',
                    'source_pattern': pattern,
                    'adapted_pattern': pattern,  # Direct mapping - no adaptation
                    'confidence': transfer_confidence,
                    'strategy': 'direct_mapping'
                })
        
        return transfers
    
    def _analogical_transfer(self, source_kb: Dict[str, Any], target_context: Dict[str, Any], 
                           domain_similarity: float) -> List[Dict[str, Any]]:
        """Analogical transfer strategy"""
        transfers = []
        
        # Create analogical mappings
        source_features = list(source_kb['features'])
        target_features = target_context.get('features', [])
        
        if len(source_features) > 0 and len(target_features) > 0:
            # Simple analogical mapping (could be more sophisticated)
            for pattern in source_kb['patterns']:
                if pattern['type'] == 'feature_cooccurrence':
                    source_feats = pattern['pattern']['features']
                    # Map to target features by position (simple heuristic)
                    if len(target_features) >= len(source_feats):
                        adapted_pattern = pattern.copy()
                        adapted_pattern['pattern']['features'] = target_features[:len(source_feats)]
                        
                        transfer_confidence = pattern['confidence'] * domain_similarity * 0.8  # Reduce for analogical uncertainty
                        transfers.append({
                            'type': 'analogical_transfer',
                            'source_pattern': pattern,
                            'adapted_pattern': adapted_pattern,
                            'confidence': transfer_confidence,
                            'strategy': 'analogical_transfer'
                        })
        
        return transfers
    
    def _abstract_pattern_transfer(self, source_kb: Dict[str, Any], target_context: Dict[str, Any], 
                                 domain_similarity: float) -> List[Dict[str, Any]]:
        """Abstract pattern transfer strategy"""
        transfers = []
        
        # Transfer abstract patterns (structure without specific features)
        for pattern in source_kb['patterns']:
            if pattern['type'] == 'sequential_pattern':
                # Abstract the sequence structure
                adapted_pattern = {
                    'type': 'abstract_sequential',
                    'pattern': {
                        'sequence_length': len(pattern['pattern']['sequence']),
                        'frequency': pattern['pattern']['frequency']
                    },
                    'confidence': pattern['confidence']
                }
                
                transfer_confidence = pattern['confidence'] * domain_similarity * 0.7
                transfers.append({
                    'type': 'abstract_pattern_transfer',
                    'source_pattern': pattern,
                    'adapted_pattern': adapted_pattern,
                    'confidence': transfer_confidence,
                    'strategy': 'abstract_pattern_transfer'
                })
        
        return transfers
    
    def _compositional_transfer(self, source_kb: Dict[str, Any], target_context: Dict[str, Any], 
                              domain_similarity: float) -> List[Dict[str, Any]]:
        """Compositional transfer strategy"""
        transfers = []
        
        # Combine multiple source patterns for compositional transfer
        patterns = source_kb['patterns']
        if len(patterns) >= 2:
            for i, pattern1 in enumerate(patterns):
                for pattern2 in patterns[i+1:]:
                    if pattern1['type'] != pattern2['type']:  # Different types can be composed
                        composed_pattern = {
                            'type': 'compositional',
                            'pattern': {
                                'components': [pattern1['pattern'], pattern2['pattern']],
                                'composition_type': f"{pattern1['type']}+{pattern2['type']}"
                            },
                            'confidence': min(pattern1['confidence'], pattern2['confidence'])
                        }
                        
                        transfer_confidence = composed_pattern['confidence'] * domain_similarity * 0.6
                        transfers.append({
                            'type': 'compositional_transfer',
                            'source_pattern': [pattern1, pattern2],
                            'adapted_pattern': composed_pattern,
                            'confidence': transfer_confidence,
                            'strategy': 'compositional_transfer'
                        })
        
        return transfers
    
    def _meta_learning_transfer(self, source_kb: Dict[str, Any], target_context: Dict[str, Any], 
                              domain_similarity: float) -> List[Dict[str, Any]]:
        """Meta-learning transfer strategy"""
        transfers = []
        
        # Transfer meta-learning insights
        meta_features = source_kb['meta_features']
        
        if meta_features:
            meta_pattern = {
                'type': 'meta_learning',
                'pattern': {
                    'learning_strategy': 'adaptive',
                    'complexity_handling': meta_features.get('feature_count', 0),
                    'outcome_prediction_approach': 'ensemble' if meta_features.get('outcome_diversity', 0) > 2 else 'simple'
                },
                'confidence': 0.6
            }
            
            transfer_confidence = 0.6 * domain_similarity * 0.5
            transfers.append({
                'type': 'meta_learning_transfer',
                'source_pattern': meta_features,
                'adapted_pattern': meta_pattern,
                'confidence': transfer_confidence,
                'strategy': 'meta_learning_transfer'
            })
        
        return transfers
    
    def _rank_transfers(self, transfers: List[Dict[str, Any]], target_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank transfers by quality and relevance"""
        # Sort by confidence score
        ranked = sorted(transfers, key=lambda t: t['confidence'], reverse=True)
        
        # Apply diversity filtering to avoid redundant transfers
        diverse_transfers = []
        seen_types = set()
        
        for transfer in ranked:
            transfer_type = transfer['type']
            if transfer_type not in seen_types or len(diverse_transfers) < 3:
                diverse_transfers.append(transfer)
                seen_types.add(transfer_type)
        
        return diverse_transfers
    
    def _apply_transfers_to_domain(self, target_domain: str, transfers: List[Dict[str, Any]]) -> None:
        """Apply transferred knowledge to target domain"""
        if target_domain not in self.knowledge_base:
            self.knowledge_base[target_domain] = {
                'patterns': [], 'features': set(), 'outcomes': [], 
                'relationships': [], 'meta_features': {}
            }
        
        target_kb = self.knowledge_base[target_domain]
        
        for transfer in transfers:
            adapted_pattern = transfer['adapted_pattern']
            
            # Add to target domain patterns
            target_kb['patterns'].append({
                'pattern': adapted_pattern,
                'confidence': transfer['confidence'],
                'transfer_source': transfer['strategy'],
                'timestamp': datetime.utcnow().isoformat()
            })
    
    def get_transfer_summary(self) -> Dict[str, Any]:
        """Get summary statistics of transfer learning history"""
        if not self.transfer_history:
            return {'total_transfers': 0, 'average_quality': 0.0, 'domains_involved': 0}
        
        total_transfers = len(self.transfer_history)
        avg_quality = np.mean([record['transfer_quality'] for record in self.transfer_history])
        
        all_domains = set()
        for record in self.transfer_history:
            all_domains.add(record['source_domain'])
            all_domains.add(record['target_domain'])
        
        return {
            'total_transfers': total_transfers,
            'average_quality': avg_quality,
            'domains_involved': len(all_domains),
            'average_similarity': np.mean([record['domain_similarity'] for record in self.transfer_history]),
            'knowledge_base_size': len(self.knowledge_base)
        }
    
    def save_knowledge_base(self, filepath: str) -> bool:
        """Save knowledge base to file"""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'knowledge_base': self.knowledge_base,
                    'learned_patterns': self.learned_patterns,
                    'transfer_history': self.transfer_history
                }, f)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save knowledge base: {e}")
            return False
    
    def load_knowledge_base(self, filepath: str) -> bool:
        """Load knowledge base from file"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.knowledge_base = data['knowledge_base']
                self.learned_patterns = data['learned_patterns']
                self.transfer_history = data['transfer_history']
            return True
        except Exception as e:
            self.logger.error(f"Failed to load knowledge base: {e}")
            return False

# Example usage and testing
if __name__ == "__main__":
    # Initialize transfer learning engine
    engine = TransferLearningEngine(similarity_threshold=0.6, max_transfer_depth=3)
    
    # Example: Learn from physics domain
    physics_knowledge = [
        {
            'features': {'temperature': 300, 'pressure': 1.0, 'volume': 22.4},
            'outcome': 'ideal_gas_behavior',
            'sequence': ['heat', 'expand', 'cool', 'contract']
        },
        {
            'features': {'temperature': 400, 'pressure': 2.0, 'volume': 22.4},
            'outcome': 'ideal_gas_behavior',
            'sequence': ['heat', 'expand']
        }
    ]
    
    physics_summary = engine.learn_from_domain('physics', physics_knowledge)
    print("Physics Learning Summary:")
    print(json.dumps(physics_summary, indent=2))
    
    # Example: Transfer to economics domain
    economics_context = {
        'features': ['price', 'demand', 'supply'],
        'meta_features': {'feature_count': 3, 'outcome_diversity': 2},
        'patterns': [
            {'type': 'correlation', 'confidence': 0.8}
        ]
    }
    
    transfer_result = engine.transfer_knowledge('physics', 'economics', economics_context)
    print("\nTransfer Result:")
    print(json.dumps(transfer_result, indent=2, default=str))
    
    # Get summary
    summary = engine.get_transfer_summary()
    print(f"\nTransfer Summary:")
    print(json.dumps(summary, indent=2))
