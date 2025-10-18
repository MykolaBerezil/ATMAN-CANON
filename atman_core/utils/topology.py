"""
Topological Operations for Non-Orientable Reasoning Spaces
Implements Möbius transformations and non-orientable geometric operations.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging

class MobiusTransformer:
    """
    Implements Möbius transformations for reasoning space topology.
    Handles non-orientable reasoning spaces and perspective transformations.
    """
    
    def __init__(self, dimension: int = 3):
        self.dimension = dimension
        self.transformation_matrix = np.eye(dimension + 1, dtype=complex)
        self.orientation_state = 1  # +1 for orientable, -1 for non-orientable
        self.logger = logging.getLogger(__name__)
        
    def create_mobius_strip(self, reasoning_space: np.ndarray) -> np.ndarray:
        """
        Transform reasoning space into Möbius strip topology.
        
        Args:
            reasoning_space: Input reasoning space coordinates
            
        Returns:
            Transformed coordinates with Möbius topology
        """
        if reasoning_space.shape[1] != self.dimension:
            raise ValueError(f"Input space must have {self.dimension} dimensions")
        
        # Apply Möbius transformation: z -> (az + b) / (cz + d)
        # Extended to higher dimensions using quaternions/octonions
        transformed_space = np.zeros_like(reasoning_space, dtype=complex)
        
        for i, point in enumerate(reasoning_space):
            # Convert to complex representation
            z = complex(point[0], point[1] if len(point) > 1 else 0)
            
            # Apply Möbius transformation
            a, b, c, d = self._get_mobius_coefficients()
            transformed_z = (a * z + b) / (c * z + d)
            
            # Handle the twist (non-orientable property)
            twist_factor = np.exp(1j * np.pi * i / len(reasoning_space))
            transformed_z *= twist_factor
            
            # Convert back to real coordinates
            transformed_space[i, 0] = transformed_z.real
            if self.dimension > 1:
                transformed_space[i, 1] = transformed_z.imag
            if self.dimension > 2:
                # Preserve additional dimensions with topological constraints
                for j in range(2, self.dimension):
                    transformed_space[i, j] = point[j] * np.cos(np.angle(transformed_z))
        
        return transformed_space.real
    
    def apply_perspective_transformation(self, evidence_set: List[Dict[str, Any]], 
                                       perspective_angle: float) -> List[Dict[str, Any]]:
        """
        Apply perspective transformation to evidence based on viewpoint.
        
        Args:
            evidence_set: List of evidence items
            perspective_angle: Angle of perspective transformation (radians)
            
        Returns:
            Evidence set with transformed perspectives
        """
        transformed_evidence = []
        
        for evidence in evidence_set:
            transformed = evidence.copy()
            
            # Apply perspective transformation to confidence
            if 'confidence' in evidence:
                original_conf = evidence['confidence']
                # Perspective transformation affects confidence based on angle
                perspective_factor = np.cos(perspective_angle) ** 2
                transformed['confidence'] = original_conf * perspective_factor
            
            # Transform feature weights based on perspective
            if 'features' in evidence:
                transformed['features'] = {}
                for feature, value in evidence['features'].items():
                    # Apply rotation in feature space
                    if isinstance(value, (int, float)):
                        rotation_matrix = np.array([
                            [np.cos(perspective_angle), -np.sin(perspective_angle)],
                            [np.sin(perspective_angle), np.cos(perspective_angle)]
                        ])
                        # Project to 2D for transformation
                        point_2d = np.array([value, 0])
                        transformed_point = rotation_matrix @ point_2d
                        transformed['features'][feature] = transformed_point[0]
                    else:
                        transformed['features'][feature] = value
            
            # Add perspective metadata
            transformed['perspective_metadata'] = {
                'transformation_angle': perspective_angle,
                'orientation_state': self.orientation_state,
                'topology_type': 'mobius_strip'
            }
            
            transformed_evidence.append(transformed)
        
        return transformed_evidence
    
    def detect_orientation_flip(self, before_space: np.ndarray, 
                              after_space: np.ndarray) -> bool:
        """
        Detect if reasoning space has undergone orientation flip.
        
        Args:
            before_space: Reasoning space before transformation
            after_space: Reasoning space after transformation
            
        Returns:
            True if orientation has flipped
        """
        if before_space.shape != after_space.shape:
            return False
        
        # Calculate determinant to check orientation
        if before_space.shape[1] >= 2:
            # Use first few points to form vectors
            n_points = min(3, len(before_space))
            
            before_vectors = np.diff(before_space[:n_points], axis=0)
            after_vectors = np.diff(after_space[:n_points], axis=0)
            
            if before_vectors.shape[0] >= 2 and before_vectors.shape[1] >= 2:
                # Calculate cross product determinant for 2D case
                before_det = np.linalg.det(before_vectors[:2, :2])
                after_det = np.linalg.det(after_vectors[:2, :2])
                
                # Orientation flip if determinants have opposite signs
                return np.sign(before_det) != np.sign(after_det)
        
        return False
    
    def create_klein_bottle_embedding(self, reasoning_paths: List[np.ndarray]) -> np.ndarray:
        """
        Embed reasoning paths in Klein bottle topology.
        
        Args:
            reasoning_paths: List of reasoning path coordinates
            
        Returns:
            Klein bottle embedded coordinates
        """
        if not reasoning_paths:
            return np.array([])
        
        # Flatten all paths into single coordinate array
        all_points = np.vstack(reasoning_paths)
        n_points = len(all_points)
        
        # Klein bottle parametrization in 4D
        # x = (R + r*cos(v/2)*sin(u) - r*sin(v/2)*sin(2*u)) * cos(v)
        # y = (R + r*cos(v/2)*sin(u) - r*sin(v/2)*sin(2*u)) * sin(v)
        # z = r*sin(v/2)*sin(u) + r*cos(v/2)*sin(2*u)
        # w = r*cos(v/2)*cos(u) - r*sin(v/2)*cos(2*u)
        
        R, r = 3.0, 1.0  # Klein bottle parameters
        embedded_points = np.zeros((n_points, 4))
        
        for i, point in enumerate(all_points):
            # Map point coordinates to Klein bottle parameters
            u = 2 * np.pi * (point[0] % 1.0) if len(point) > 0 else 0
            v = 2 * np.pi * (point[1] % 1.0) if len(point) > 1 else 0
            
            # Klein bottle embedding
            cos_v_2 = np.cos(v / 2)
            sin_v_2 = np.sin(v / 2)
            cos_u = np.cos(u)
            sin_u = np.sin(u)
            cos_2u = np.cos(2 * u)
            sin_2u = np.sin(2 * u)
            cos_v = np.cos(v)
            sin_v = np.sin(v)
            
            base_term = R + r * cos_v_2 * sin_u - r * sin_v_2 * sin_2u
            
            embedded_points[i, 0] = base_term * cos_v
            embedded_points[i, 1] = base_term * sin_v
            embedded_points[i, 2] = r * sin_v_2 * sin_u + r * cos_v_2 * sin_2u
            embedded_points[i, 3] = r * cos_v_2 * cos_u - r * sin_v_2 * cos_2u
        
        return embedded_points
    
    def compute_topological_invariants(self, reasoning_space: np.ndarray) -> Dict[str, float]:
        """
        Compute topological invariants of the reasoning space.
        
        Args:
            reasoning_space: Coordinate array of reasoning space
            
        Returns:
            Dictionary of topological invariants
        """
        invariants = {}
        
        if len(reasoning_space) < 3:
            return invariants
        
        # Euler characteristic approximation
        # For discrete point set, approximate using triangulation
        n_points = len(reasoning_space)
        n_edges = n_points * (n_points - 1) // 2  # Complete graph approximation
        n_faces = n_points * (n_points - 1) * (n_points - 2) // 6  # Triangulation approximation
        
        euler_char = n_points - n_edges + n_faces
        invariants['euler_characteristic'] = euler_char
        
        # Genus estimation (for closed surfaces)
        genus = (2 - euler_char) / 2
        invariants['genus'] = max(0, genus)
        
        # Betti numbers approximation
        invariants['betti_0'] = 1  # Connected components (assuming connected)
        invariants['betti_1'] = max(0, n_edges - n_points + 1)  # Loops
        
        # Orientability check
        invariants['orientable'] = self.orientation_state > 0
        
        # Curvature approximation
        if reasoning_space.shape[1] >= 2:
            # Discrete curvature based on angle defects
            curvatures = []
            for i in range(min(10, len(reasoning_space))):  # Sample points
                neighbors = self._find_nearest_neighbors(reasoning_space, i, k=6)
                if len(neighbors) >= 3:
                    angles = self._compute_angles_at_point(reasoning_space, i, neighbors)
                    angle_defect = 2 * np.pi - np.sum(angles)
                    curvatures.append(angle_defect)
            
            if curvatures:
                invariants['mean_curvature'] = np.mean(curvatures)
                invariants['gaussian_curvature'] = np.var(curvatures)
        
        return invariants
    
    def _get_mobius_coefficients(self) -> Tuple[complex, complex, complex, complex]:
        """Get Möbius transformation coefficients."""
        # Standard coefficients for interesting transformations
        a = complex(1, 0.1)
        b = complex(0.5, 0)
        c = complex(0, 0.1)
        d = complex(1, -0.1)
        
        # Ensure ad - bc != 0 (non-degenerate transformation)
        det = a * d - b * c
        if abs(det) < 1e-10:
            d = complex(1, 0)
        
        return a, b, c, d
    
    def _find_nearest_neighbors(self, points: np.ndarray, center_idx: int, k: int) -> List[int]:
        """Find k nearest neighbors to a point."""
        center = points[center_idx]
        distances = np.linalg.norm(points - center, axis=1)
        distances[center_idx] = np.inf  # Exclude the center point itself
        
        nearest_indices = np.argsort(distances)[:k]
        return nearest_indices.tolist()
    
    def _compute_angles_at_point(self, points: np.ndarray, center_idx: int, 
                               neighbor_indices: List[int]) -> np.ndarray:
        """Compute angles between vectors from center to neighbors."""
        center = points[center_idx]
        vectors = points[neighbor_indices] - center
        
        angles = []
        n_neighbors = len(neighbor_indices)
        
        for i in range(n_neighbors):
            v1 = vectors[i]
            v2 = vectors[(i + 1) % n_neighbors]
            
            # Compute angle between vectors
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            cos_angle = np.clip(cos_angle, -1, 1)  # Numerical stability
            angle = np.arccos(cos_angle)
            angles.append(angle)
        
        return np.array(angles)
    
    def flip_orientation(self):
        """Flip the orientation state of the reasoning space."""
        self.orientation_state *= -1
        self.logger.info(f"Orientation flipped to: {'orientable' if self.orientation_state > 0 else 'non-orientable'}")
    
    def reset_transformation(self):
        """Reset transformation matrix to identity."""
        self.transformation_matrix = np.eye(self.dimension + 1, dtype=complex)
        self.orientation_state = 1
        self.logger.info("Transformation matrix reset to identity")
