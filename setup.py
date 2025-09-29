#!/usr/bin/env python3
"""
ContextAgent: Context-Aware Proactive LLM Agents with Open-World Sensory Perceptions

A research framework for developing and evaluating context-aware proactive LLM agents
that harness extensive sensory contexts for enhanced proactive services.
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="contextagent",
    version="1.0.0",
    author="Bufang Yang, Lilin Xu, Liekang Zeng, Kaiwei Liu, Siyang Jiang, Wenrui Lu, Hongkai Chen, Xiaofan Jiang, Guoliang Xing, Zhenyu Yan",
    author_email="bfyang@cuhk.edu.hk",
    description="Context-Aware Proactive LLM Agents with Open-World Sensory Perceptions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bf-yang/ContextAgent",
    project_urls={
        "Bug Reports": "https://github.com/bf-yang/ContextAgent/issues",
        "Source": "https://github.com/bf-yang/ContextAgent",
        "Documentation": "https://github.com/bf-yang/ContextAgent#readme",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
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
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "contextagent-icl=icl.inference_api:main",
            "contextagent-sft=sft.train:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
