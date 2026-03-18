#!/usr/bin/env python3
"""
FireRedVAD-Engineering - Setup Script

Package as Python wheel:
    python setup.py sdist bdist_wheel

Install from wheel:
    pip install dist/fireredvad_engineering-1.1.0-py3-none-any.whl

Development install:
    pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = (Path(__file__).parent / "requirements.txt").read_text().strip().split("\n")
requirements = [r.strip() for r in requirements if r.strip() and not r.startswith("#")]

setup(
    name="fireredvad-engineering",
    version="1.1.0",
    author="FireRed Team",
    author_email="your.email@example.com",
    description="Production-Ready Streaming Voice Activity Detection with ONNX Runtime",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/leospark/FireRedVAD-Engineering",
    license="MIT",
    
    # Packages
    packages=find_packages(exclude=["examples", "output", "tests"]),
    
    # Include non-Python files
    package_data={
        "fireredvad": ["core/*.py"],
        "fireredvad.core": ["*.py"],
        "models": ["*.onnx", "*.onnx.data", "*.ark"],
    },
    
    # Data files (for models)
    data_files=[
        ("fireredvad_engineering/models", [
            "models/model_with_caches.onnx",
            "models/model_with_caches.onnx.data",
            "models/cmvn.ark",
        ]),
    ],
    
    # Dependencies
    install_requires=requirements,
    
    # Python version
    python_requires=">=3.8",
    
    # Entry points (CLI)
    entry_points={
        "console_scripts": [
            "fireredvad=fireredvad.cli:main",
        ],
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    
    # Keywords
    keywords="vad voice activity detection onnx speech audio kaldi firered",
)
