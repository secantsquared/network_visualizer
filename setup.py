"""
Setup script for Network Analyzer package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="network-analyzer",
    version="0.1.0",
    author="Network Analyzer Team",
    author_email="contact@example.com",
    description="A tool for building and analyzing knowledge networks from Wikipedia and course data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/network-analyzer",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Researchers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.8",
    install_requires=[
        "networkx>=3.0",
        "requests>=2.25.0",
        "requests-cache>=0.9.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "pyvis>=0.3.0",
        "tqdm>=4.60.0",
        "aiohttp>=3.8.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-asyncio>=0.18.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.910",
        ],
    },
    entry_points={
        "console_scripts": [
            "network-analyzer=network_analyzer.scripts.main:main",
            "network-analyzer-unified=network_analyzer.scripts.unified_main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)