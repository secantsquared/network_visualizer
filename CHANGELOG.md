# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2025-07-09

### Added
- Initial release of Network Analyzer tool
- Multi-source network building from Wikipedia articles and Coursera course data
- Multiple network construction algorithms:
  - Breadth-first search (sync/async)
  - Random walk with restart
  - Depth-first search with backtracking
  - Topic-focused crawling
  - Hub-and-spoke expansion
- Advanced network analysis features:
  - Community detection using modularity optimization
  - Centrality measures (degree, PageRank, betweenness)
  - Network statistics and metrics
  - Influence propagation simulation (Independent Cascade and Linear Threshold models)
- Interactive visualizations:
  - Web-based network graphs with physics simulation
  - Multiple layout algorithms (Barnes-Hut, Force Atlas2, etc.)
  - Static community plots
  - Influence propagation visualizations
- Data export capabilities in GraphML format
- Comprehensive filtering of irrelevant pages and content
- Command-line interfaces for both Wikipedia-only and multi-source networks
- Proper Python package structure with organized submodules
- Configuration management system
- Async rate limiting for API requests
- Extensive test suite

### Changed
- Reorganized project structure into proper Python package layout
- Moved core modules to `src/network_analyzer/` directory
- Organized codebase into logical subpackages:
  - `core/` - Core network building functionality
  - `data_sources/` - Data source adapters and implementations
  - `analysis/` - Advanced analysis tools
  - `visualization/` - Visualization components
  - `utils/` - Utility functions and helpers
- Moved command-line scripts to dedicated `scripts/` directory
- Added proper package configuration files (`pyproject.toml`, `setup.py`)
- Enhanced README with comprehensive documentation and examples

### Fixed
- Improved error handling and robustness in network building
- Enhanced data validation and filtering
- Better async request management and rate limiting

### Documentation
- Added comprehensive README with installation instructions
- Included usage examples for both basic and advanced features
- Added project structure documentation
- Provided configuration options reference
- Added Coursera dataset attribution and Kaggle source link
- Included citation information for academic use

### Technical Details
- Built with Python 3.8+ compatibility
- Uses NetworkX 3.0+ for graph operations
- Implements MediaWiki API integration for Wikipedia data
- Supports hybrid data sources combining Wikipedia and Coursera datasets
- Includes extensive test coverage with multiple visualization test cases
- Provides configurable performance settings for different use cases

---

*This changelog follows the [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format. For more information about changes, see the [commit history](https://github.com/username/network-analyzer/commits/main).*