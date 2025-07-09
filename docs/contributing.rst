Contributing
============

We welcome contributions to Network Analyzer! This guide will help you get started with contributing to the project.

Getting Started
---------------

Development Setup
~~~~~~~~~~~~~~~~

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:

   .. code-block:: bash

       git clone https://github.com/yourusername/network-analyzer.git
       cd network-analyzer

3. **Create a virtual environment**:

   .. code-block:: bash

       python -m venv venv
       source venv/bin/activate  # On Windows: venv\Scripts\activate

4. **Install development dependencies**:

   .. code-block:: bash

       pip install -e ".[dev]"

5. **Create a feature branch**:

   .. code-block:: bash

       git checkout -b feature/your-feature-name

Development Dependencies
~~~~~~~~~~~~~~~~~~~~~~~

The development installation includes:

- **pytest**: Testing framework
- **pytest-asyncio**: Async testing support
- **black**: Code formatting
- **flake8**: Linting
- **mypy**: Type checking
- **sphinx**: Documentation generation
- **pre-commit**: Git hooks for code quality

Setting Up Pre-commit Hooks
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install pre-commit hooks to ensure code quality:

.. code-block:: bash

    pre-commit install

This will run formatting, linting, and type checking on every commit.

Code Style and Standards
------------------------

Code Formatting
~~~~~~~~~~~~~~

We use **Black** for code formatting:

.. code-block:: bash

    black src/ tests/ scripts/

Configuration is in ``pyproject.toml``:

.. code-block:: toml

    [tool.black]
    line-length = 88
    target-version = ['py38']
    include = '\.pyi?$'

Linting
~~~~~~~

We use **flake8** for linting:

.. code-block:: bash

    flake8 src/ tests/ scripts/

Configuration is in ``.flake8``:

.. code-block:: ini

    [flake8]
    max-line-length = 88
    extend-ignore = E203, W503
    exclude = venv/, build/, dist/

Type Checking
~~~~~~~~~~~~

We use **mypy** for type checking:

.. code-block:: bash

    mypy src/

Configuration is in ``pyproject.toml``:

.. code-block:: toml

    [tool.mypy]
    python_version = "3.8"
    warn_return_any = true
    warn_unused_configs = true
    disallow_untyped_defs = true

Documentation Standards
~~~~~~~~~~~~~~~~~~~~~~

- Use **Google-style docstrings** for all functions and classes
- Include type hints for all function parameters and return values
- Add docstring examples for public functions
- Keep docstrings concise but comprehensive

Example docstring:

.. code-block:: python

    def build_network(self, seed_topics: List[str]) -> nx.Graph:
        """Build a network starting from the given seed topics.
        
        Args:
            seed_topics: List of topic names to start network building from.
            
        Returns:
            A NetworkX graph representing the built network.
            
        Raises:
            NetworkBuildError: If network building fails.
            
        Example:
            >>> builder = WikipediaNetworkBuilder(config)
            >>> graph = builder.build_network(["Machine Learning"])
            >>> print(graph.number_of_nodes())
            25
        """

Testing
-------

Running Tests
~~~~~~~~~~~~

Run the full test suite:

.. code-block:: bash

    pytest

Run tests with coverage:

.. code-block:: bash

    pytest --cov=network_analyzer --cov-report=html

Run specific test files:

.. code-block:: bash

    pytest tests/test_network_builder.py

Test Structure
~~~~~~~~~~~~~

Tests are organized in the ``tests/`` directory:

.. code-block:: text

    tests/
    ├── __init__.py
    ├── conftest.py                 # Test configuration and fixtures
    ├── test_config.py             # Configuration tests
    ├── test_network_builder.py    # Network builder tests
    ├── test_data_sources.py       # Data source tests
    ├── test_analysis.py           # Analysis tests
    ├── test_visualization.py      # Visualization tests
    └── integration/               # Integration tests
        ├── test_full_workflow.py
        └── test_cli.py

Writing Tests
~~~~~~~~~~~~

Use pytest fixtures and follow these patterns:

.. code-block:: python

    import pytest
    from unittest.mock import Mock, patch
    from network_analyzer import NetworkConfig, WikipediaNetworkBuilder


    @pytest.fixture
    def sample_config():
        """Sample configuration for testing."""
        return NetworkConfig(
            max_depth=1,
            max_articles_to_process=5,
            links_per_article=3
        )


    @pytest.fixture
    def mock_wikipedia_response():
        """Mock Wikipedia API response."""
        return {
            "query": {
                "pages": {
                    "123": {
                        "title": "Machine Learning",
                        "links": [
                            {"title": "Artificial Intelligence"},
                            {"title": "Data Science"}
                        ]
                    }
                }
            }
        }


    def test_network_builder_initialization(sample_config):
        """Test that NetworkBuilder initializes correctly."""
        builder = WikipediaNetworkBuilder(sample_config)
        assert builder.config == sample_config
        assert builder.graph is not None


    @patch('network_analyzer.core.network_builder.requests.get')
    def test_build_network_basic(mock_get, sample_config, mock_wikipedia_response):
        """Test basic network building functionality."""
        mock_get.return_value.json.return_value = mock_wikipedia_response
        
        builder = WikipediaNetworkBuilder(sample_config)
        graph = builder.build_network(["Machine Learning"])
        
        assert graph.number_of_nodes() > 0
        assert "Machine Learning" in graph.nodes()

Testing Guidelines
~~~~~~~~~~~~~~~~~

1. **Write tests for all new functionality**
2. **Use descriptive test names** that explain what is being tested
3. **Mock external dependencies** (API calls, file systems)
4. **Test both success and failure cases**
5. **Include integration tests** for complex workflows
6. **Keep tests fast and independent**

Async Testing
~~~~~~~~~~~~

For async functionality, use pytest-asyncio:

.. code-block:: python

    import pytest
    import asyncio
    from network_analyzer import NetworkConfig, WikipediaNetworkBuilder


    @pytest.mark.asyncio
    async def test_async_network_building():
        """Test async network building."""
        config = NetworkConfig(
            async_enabled=True,
            max_depth=1,
            max_articles_to_process=3
        )
        
        builder = WikipediaNetworkBuilder(config)
        graph = await builder.build_network_async(["Machine Learning"])
        
        assert graph.number_of_nodes() > 0

Contributing Guidelines
----------------------

Issues and Feature Requests
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Before creating an issue:**

1. Check if the issue already exists
2. Search closed issues for similar problems
3. Try to reproduce the issue with the latest version

**When creating an issue:**

1. Use a clear, descriptive title
2. Provide steps to reproduce the problem
3. Include relevant code snippets
4. Specify your environment (Python version, OS, etc.)
5. Add labels if you have permission

**Feature requests:**

1. Explain the use case and motivation
2. Provide examples of how the feature would be used
3. Consider proposing an implementation approach

Pull Requests
~~~~~~~~~~~~

**Before submitting a pull request:**

1. Create an issue to discuss the change (for significant changes)
2. Fork the repository and create a feature branch
3. Make your changes following the code style guidelines
4. Add tests for new functionality
5. Update documentation if needed
6. Run the full test suite

**Pull request process:**

1. **Create a descriptive title** and description
2. **Link to related issues** using "Closes #123" or "Fixes #123"
3. **Include a summary** of changes made
4. **Add test results** if applicable
5. **Be responsive** to review feedback

**Pull request template:**

.. code-block:: markdown

    ## Description
    Brief description of the changes made.

    ## Type of Change
    - [ ] Bug fix
    - [ ] New feature
    - [ ] Breaking change
    - [ ] Documentation update

    ## Testing
    - [ ] Tests pass locally
    - [ ] New tests added for new functionality
    - [ ] Documentation updated

    ## Checklist
    - [ ] Code follows style guidelines
    - [ ] Self-review completed
    - [ ] Comments added for complex code
    - [ ] Documentation updated

Code Review Process
~~~~~~~~~~~~~~~~~~

**For reviewers:**

1. Focus on code quality, not personal preferences
2. Be constructive and specific in feedback
3. Approve if the code meets standards
4. Test the changes locally if possible

**For contributors:**

1. Be open to feedback and suggestions
2. Address all review comments
3. Update the PR description if scope changes
4. Rebase if requested

Development Workflow
-------------------

Branching Strategy
~~~~~~~~~~~~~~~~~

- **main**: Stable, production-ready code
- **develop**: Integration branch for new features
- **feature/**: Feature branches for new functionality
- **bugfix/**: Bug fix branches
- **hotfix/**: Critical fixes for production

Branch naming conventions:

.. code-block:: bash

    feature/add-community-detection
    bugfix/fix-async-timeout
    hotfix/fix-critical-memory-leak

Commit Messages
~~~~~~~~~~~~~~

Follow conventional commit format:

.. code-block:: text

    type(scope): subject

    body

    footer

Types:
- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes
- **refactor**: Code refactoring
- **test**: Test changes
- **chore**: Maintenance tasks

Examples:

.. code-block:: text

    feat(analysis): add influence propagation analysis
    
    - Implement independent cascade model
    - Add linear threshold model
    - Include visualization for propagation results
    
    Closes #123

    fix(network): resolve async timeout issues
    
    - Increase default timeout values
    - Add retry logic for failed requests
    - Improve error handling for network issues

Release Process
~~~~~~~~~~~~~~

1. **Create release branch** from develop
2. **Update version numbers** in relevant files
3. **Update CHANGELOG.md** with new features and fixes
4. **Run full test suite** and fix any issues
5. **Create pull request** to main branch
6. **Tag release** after merge
7. **Deploy to PyPI** (maintainers only)

Documentation
-------------

Building Documentation
~~~~~~~~~~~~~~~~~~~~~

Build documentation locally:

.. code-block:: bash

    cd docs/
    pip install -r requirements.txt
    make html

View documentation:

.. code-block:: bash

    open _build/html/index.html

Documentation Guidelines
~~~~~~~~~~~~~~~~~~~~~~~

1. **Keep documentation up to date** with code changes
2. **Use clear, concise language**
3. **Include code examples** for new features
4. **Add API documentation** for public functions
5. **Update README** for significant changes

Community Guidelines
-------------------

Code of Conduct
~~~~~~~~~~~~~~~

We follow the Python Software Foundation's Code of Conduct. Please be respectful and inclusive in all interactions.

Getting Help
~~~~~~~~~~~

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Documentation**: Comprehensive guides and API reference

Recognition
~~~~~~~~~~

Contributors are recognized in:

- **CONTRIBUTORS.md**: List of all contributors
- **Release notes**: Acknowledgment of contributions
- **Documentation**: Author attributions

Areas for Contribution
---------------------

We welcome contributions in these areas:

**Code Contributions:**
- New data source adapters
- Additional network analysis algorithms
- Performance optimizations
- Visualization improvements
- Bug fixes and improvements

**Documentation:**
- Tutorial improvements
- API documentation
- Example notebooks
- Translation to other languages

**Testing:**
- Unit test coverage improvements
- Integration test scenarios
- Performance benchmarks
- Cross-platform testing

**Infrastructure:**
- CI/CD improvements
- Docker configurations
- Package management
- Release automation

Getting Started with Your First Contribution
-------------------------------------------

1. **Look for "good first issue" labels** on GitHub
2. **Start with documentation** improvements
3. **Fix typos or improve error messages**
4. **Add tests** for existing functionality
5. **Implement small feature requests**

Example first contributions:

- Add type hints to untyped functions
- Improve error messages
- Add docstring examples
- Fix documentation typos
- Add unit tests for edge cases

Thank You
---------

Thank you for your interest in contributing to Network Analyzer! Your contributions help make this project better for everyone.

For questions about contributing, please open an issue or start a discussion on GitHub.