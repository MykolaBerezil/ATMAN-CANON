#!/usr/bin/env python3
"""
ATMAN-CANON Framework Structure Tests
Validates that the framework components can be imported and initialized correctly.
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add the framework to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestFrameworkImports(unittest.TestCase):
    """Test that all framework components can be imported."""
    
    def test_atman_core_import(self):
        """Test that atman_core package can be imported."""
        try:
            import atman_core
            self.assertTrue(hasattr(atman_core, '__version__'))
        except ImportError as e:
            self.fail(f"Failed to import atman_core: {e}")
    
    def test_core_components_import(self):
        """Test that core components can be imported."""
        try:
            from atman_core import RBMKFramework, EvidenceCalculator, InvariantDetector
            
            # Test that classes exist
            self.assertTrue(callable(RBMKFramework))
            self.assertTrue(callable(EvidenceCalculator))
            self.assertTrue(callable(InvariantDetector))
            
        except ImportError as e:
            self.fail(f"Failed to import core components: {e}")
    
    def test_utils_import(self):
        """Test that utility components can be imported."""
        try:
            from atman_core.utils import RenormalizationSafety, KappaBlockLogic
            
            # Test that classes exist
            self.assertTrue(callable(RenormalizationSafety))
            self.assertTrue(callable(KappaBlockLogic))
            
        except ImportError as e:
            self.fail(f"Failed to import utils: {e}")
    
    def test_core_modules_import(self):
        """Test that individual core modules can be imported."""
        try:
            from atman_core.core import rbmk, evidence_calculator, transfer_learning, invariants
            
            # Test that modules exist
            self.assertTrue(hasattr(rbmk, 'RBMKReasoner'))
            self.assertTrue(hasattr(evidence_calculator, 'EvidenceCalculator'))
            self.assertTrue(hasattr(transfer_learning, 'TransferLearningEngine'))
            self.assertTrue(hasattr(invariants, 'InvariantDetector'))
            
        except ImportError as e:
            self.fail(f"Failed to import core modules: {e}")

class TestFrameworkInitialization(unittest.TestCase):
    """Test that framework components can be initialized."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock numpy and other dependencies that might not be available
        self.numpy_mock = MagicMock()
        self.scipy_mock = MagicMock()
        
    def test_rbmk_initialization(self):
        """Test RBMK reasoner can be initialized."""
        try:
            from atman_core import RBMKFramework
            
            # Test basic initialization
            reasoner = RBMKFramework()
            self.assertIsNotNone(reasoner)
            
            # Test that it has expected methods
            self.assertTrue(hasattr(reasoner, 'process_knowledge_item'))
            self.assertTrue(hasattr(reasoner, 'belief_network'))
            
        except Exception as e:
            self.fail(f"Failed to initialize RBMKReasoner: {e}")
    
    def test_evidence_calculator_initialization(self):
        """Test evidence calculator can be initialized."""
        try:
            from atman_core import EvidenceCalculator
            
            # Test basic initialization
            calc = EvidenceCalculator()
            self.assertIsNotNone(calc)
            
            # Test that it has expected methods
            self.assertTrue(hasattr(calc, 'evaluate_evidence'))
            self.assertTrue(hasattr(calc, 'confidence_threshold'))
            
        except Exception as e:
            self.fail(f"Failed to initialize EvidenceCalculator: {e}")
    
    def test_safety_initialization(self):
        """Test safety components can be initialized."""
        try:
            from atman_core.utils import RenormalizationSafety, KappaBlockLogic
            
            # Test safety initialization
            safety = RenormalizationSafety()
            self.assertIsNotNone(safety)
            self.assertTrue(hasattr(safety, 'apply_renormalization'))
            
            # Test kappa logic initialization
            kappa = KappaBlockLogic()
            self.assertIsNotNone(kappa)
            self.assertTrue(hasattr(kappa, 'create_logical_block'))
            
        except Exception as e:
            self.fail(f"Failed to initialize safety components: {e}")

class TestFrameworkFunctionality(unittest.TestCase):
    """Test basic functionality of framework components."""
    
    def test_evidence_evaluation(self):
        """Test evidence evaluation functionality."""
        try:
            from atman_core import EvidenceCalculator
            
            calc = EvidenceCalculator()
            
            # Test with sample evidence
            evidence = {
                'source': 'test',
                'confidence': 0.8,
                'data': {'value': 42}
            }
            
            metrics = calc.evaluate_evidence(evidence)
            
            # Should return metrics dictionary
            self.assertIsInstance(metrics, dict)
            self.assertIn('overall_confidence', metrics)
            
        except Exception as e:
            self.fail(f"Evidence evaluation test failed: {e}")
    
    def test_safety_mechanisms(self):
        """Test safety mechanism functionality."""
        try:
            from atman_core.utils import RenormalizationSafety
            
            safety = RenormalizationSafety()
            
            # Test with sample dangerous state
            dangerous_state = {
                'confidence': 0.9999,  # Too high
                'features': {'extreme': 10000},
                'recursion_depth': 50  # Too deep
            }
            
            safe_state = safety.apply_renormalization(dangerous_state)
            
            # Should return a modified state
            self.assertIsInstance(safe_state, dict)
            self.assertIn('confidence', safe_state)
            
            # Confidence should be reduced
            self.assertLess(safe_state['confidence'], dangerous_state['confidence'])
            
        except Exception as e:
            self.fail(f"Safety mechanism test failed: {e}")
    
    def test_invariant_detection(self):
        """Test invariant detection functionality."""
        try:
            from atman_core import InvariantDetector
            
            detector = InvariantDetector()
            
            # Test with sample data
            data = [
                {'x': 1, 'y': 2, 'sum': 3},
                {'x': 2, 'y': 3, 'sum': 5},
                {'x': 3, 'y': 4, 'sum': 7}
            ]
            
            invariants = detector.detect_invariants(data)
            
            # Should return invariant information
            self.assertIsInstance(invariants, dict)
            
        except Exception as e:
            self.fail(f"Invariant detection test failed: {e}")

class TestExampleScripts(unittest.TestCase):
    """Test that example scripts can be imported and run."""
    
    def test_basic_reasoning_import(self):
        """Test that basic reasoning example can be imported."""
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))
            import basic_reasoning
            
            # Should have main function
            self.assertTrue(hasattr(basic_reasoning, 'main'))
            self.assertTrue(callable(basic_reasoning.main))
            
        except ImportError as e:
            self.fail(f"Failed to import basic_reasoning example: {e}")
    
    def test_safety_invariants_import(self):
        """Test that safety invariants example can be imported."""
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))
            import safety_invariants
            
            # Should have main function
            self.assertTrue(hasattr(safety_invariants, 'main'))
            self.assertTrue(callable(safety_invariants.main))
            
        except ImportError as e:
            self.fail(f"Failed to import safety_invariants example: {e}")

class TestImplementationCompatibility(unittest.TestCase):
    """Test that implementations can use the framework."""
    
    def test_blockchain_implementation_import(self):
        """Test that blockchain implementation can import framework."""
        try:
            # Add blockchain implementation to path
            blockchain_path = os.path.join(os.path.dirname(__file__), '..', 'implementations', 'blockchain_mev')
            sys.path.insert(0, blockchain_path)
            
            # Test that the updated app can import framework
            import app_updated
            
            # Should have Flask app
            self.assertTrue(hasattr(app_updated, 'app'))
            
        except ImportError as e:
            self.fail(f"Failed to import blockchain implementation: {e}")

def run_framework_validation():
    """Run comprehensive framework validation."""
    print("ATMAN-CANON Framework Validation")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestFrameworkImports))
    suite.addTests(loader.loadTestsFromTestCase(TestFrameworkInitialization))
    suite.addTests(loader.loadTestsFromTestCase(TestFrameworkFunctionality))
    suite.addTests(loader.loadTestsFromTestCase(TestExampleScripts))
    suite.addTests(loader.loadTestsFromTestCase(TestImplementationCompatibility))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    
    if result.wasSuccessful():
        print("✓ All framework validation tests passed!")
        print("✓ Framework structure is correct")
        print("✓ Components can be imported and initialized")
        print("✓ Basic functionality works")
        print("✓ Example scripts are compatible")
        print("✓ Implementation compatibility confirmed")
        return True
    else:
        print(f"✗ {len(result.failures)} test(s) failed")
        print(f"✗ {len(result.errors)} error(s) occurred")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                error_msg = traceback.split('AssertionError: ')[-1].split('\n')[0]
                print(f"  - {test}: {error_msg}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                error_msg = traceback.split('\n')[-2]
                print(f"  - {test}: {error_msg}")
        
        return False

if __name__ == "__main__":
    success = run_framework_validation()
    sys.exit(0 if success else 1)
