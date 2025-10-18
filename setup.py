#!/usr/bin/env python3
"""
Setup script for ATMAN-CANON: Fractal Consciousness Framework for Emergent Intelligence
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "ATMAN-CANON: Fractal Consciousness Framework for Emergent Intelligence"

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="atman-canon",
    version="5.0.0",
    author="ATMAN Research Team",
    author_email="research@atman-canon.org",
    description="Fractal Consciousness Framework for Emergent Intelligence",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/MykolaBerezil/ATMAN-CANON",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.812",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "myst-parser>=0.15",
        ],
        "examples": [
            "matplotlib>=3.3",
            "seaborn>=0.11",
            "plotly>=5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "atman-demo=examples.basic_reasoning:main",
            "atman-safety=examples.safety_invariants:main",
        ],
    },
    package_data={
        "atman_core": ["*.json", "*.yaml", "*.yml"],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "artificial-intelligence",
        "consciousness",
        "fractal-ai",
        "emergent-intelligence",
        "computational-philosophy",
        "theory-of-everything",
        "transfer-learning",
        "rbmk-framework",
        "consciousness-studies",
        "agi-research",
        "bayesian-reasoning",
        "meta-knowledge",
        "invariant-detection",
        "safety-mechanisms",
        "renormalization",
        "topological-reasoning",
    ],
    project_urls={
        "Bug Reports": "https://github.com/MykolaBerezil/ATMAN-CANON/issues",
        "Source": "https://github.com/MykolaBerezil/ATMAN-CANON",
        "Documentation": "https://atman-canon.readthedocs.io/",
        "Research Papers": "https://github.com/MykolaBerezil/ATMAN-CANON/tree/main/docs/papers",
    },
)
