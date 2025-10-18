"""
ATMAN-CANON Core Framework
A fractal architecture for safe, cross-domain reasoning and learning.
"""

__version__ = "0.1.0"
__author__ = "ATMAN-CANON Team"

from .core.rbmk import RBMKFramework, RBMKReasoner
from .core.evidence_calculator import EvidenceCalculator
from .core.transfer_learning import TransferLearningEngine
from .core.invariants import InvariantDetector

__all__ = [
    'RBMKFramework',
    'RBMKReasoner',
    'EvidenceCalculator', 
    'TransferLearningEngine',
    'InvariantDetector'
]
