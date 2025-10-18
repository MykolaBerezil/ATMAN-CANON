# ATMAN-CANON: A Fractal Consciousness Framework for Emergent Intelligence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ATMAN-CANON** is a Python-based computational framework for developing and testing theories of emergent intelligence based on the principles of fractal consciousness. It provides a set of core components for building adaptive, cross-domain reasoning systems that can handle uncertainty, detect invariants, and operate safely under complex conditions.

This repository is structured as a framework-first, research-ready toolkit for academics, developers, and AI enthusiasts interested in exploring advanced concepts in artificial intelligence and computational philosophy.

## Key Features

- **Recursive Bayesian Meta-Knowledge (RBMK):** A novel framework for recursive reasoning and uncertainty quantification.
- **Evidence Evaluation:** A comprehensive engine for assessing the quality, relevance, and reliability of evidence.
- **Cross-Domain Transfer Learning:** A powerful mechanism for transferring knowledge between different domains.
- **Invariant Detection:** The ability to automatically discover and validate invariant patterns in data.
- **Renormalization Safety:** A set of safety protocols to prevent runaway feedback loops and ensure bounded, stable operation.
- **κ-Block Logic:** A formal system for ensuring logical consistency in reasoning chains.

## Repository Structure

The repository is organized into the following directories:

- `atman_core/`: The core framework package containing all the essential components.
- `examples/`: Example scripts demonstrating how to use the framework.
- `implementations/`: Domain-specific implementations that use the ATMAN-CANON framework.
- `tests/`: Unit tests for validating the framework's functionality.
- `docs/`: (Coming Soon) Detailed documentation and theoretical papers.

## Installation

To install the ATMAN-CANON framework, clone the repository and install it in editable mode using pip:

```bash
git clone https://github.com/MykolaBerezil/ATMAN-CANON.git
cd ATMAN-CANON
pip install -r requirements.txt
pip install -e .
```

This will install the `atman_core` package and all its dependencies.

## Getting Started

To get started with the framework, explore the example scripts in the `examples/` directory:

- **Basic Reasoning:** `python examples/basic_reasoning.py`
  - Demonstrates evidence evaluation, invariant detection, and RBMK reasoning.

- **Safety and Invariants:** `python examples/safety_invariants.py`
  - Shows the safety mechanisms, including renormalization and κ-block logic, in action.

## Using the Framework

Here is a simple example of how to use the `EvidenceCalculator`:

```python
from atman_core import EvidenceCalculator

# Initialize the calculator
evidence_calc = EvidenceCalculator()

# Create a piece of evidence
evidence = {
    'id': 'E001',
    'domain': 'medical',
    'data': {
        'type': 'observation',
        'patient_id': 'P123',
        'symptoms': ['fever', 'cough'],
        'temperature': 38.5
    },
    'confidence': 0.85,
    'source': 'trusted_medical_journal',
    'timestamp': '2024-01-15T14:30:00',
    'conclusion': 'patient shows signs of respiratory infection'
}

# Evaluate the evidence
metrics = evidence_calc.evaluate_evidence(evidence)
accepted = metrics.get('overall_confidence', 0) > evidence_calc.confidence_threshold

print(f"Evidence Quality Score: {metrics.get('overall_confidence', 0):.3f}")
print(f"Accepted: {accepted}")
```

## Implementations

This repository includes a sample implementation for the blockchain/MEV domain, located in `implementations/blockchain_mev/`. This demonstrates how to integrate the ATMAN-CANON framework into a real-world application.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

