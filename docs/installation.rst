Installation
============

This guide covers the installation of Network Analyzer and its dependencies.

Requirements
------------

* Python 3.8 or higher
* pip package manager
* Git (for development installation)

System Dependencies
-------------------

Network Analyzer primarily uses Python packages, but some dependencies may require system-level libraries:

**Linux (Ubuntu/Debian)**:

.. code-block:: bash

    sudo apt-get update
    sudo apt-get install python3-dev python3-pip git

**macOS**:

.. code-block:: bash

    # Using Homebrew
    brew install python git

**Windows**:

Download and install Python from the official website: https://www.python.org/downloads/windows/

Installation Methods
--------------------

From Source (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~

Clone the repository and install in development mode:

.. code-block:: bash

    git clone https://github.com/username/network-analyzer.git
    cd network-analyzer
    pip install -e .

This installs the package in "editable" mode, meaning changes to the source code will be immediately available.

From PyPI (When Available)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pip install network-analyzer


Virtual Environment (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It's recommended to install Network Analyzer in a virtual environment:

.. code-block:: bash

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -e .

Development Installation
------------------------

For development work, install with additional development dependencies:

.. code-block:: bash

    git clone https://github.com/username/network-analyzer.git
    cd network-analyzer
    pip install -e ".[dev]"

This includes testing, formatting, and type checking tools.

Verifying Installation
----------------------

Test your installation by running:

.. code-block:: bash

    python -c "import network_analyzer; print('Installation successful!')"

Or use the CLI:

.. code-block:: bash

    network-analyzer --help

Common Installation Issues
--------------------------

**ImportError: No module named 'network_analyzer'**

Ensure you're in the correct virtual environment and the package is installed.

**Permission Denied**

On some systems, you may need to use ``sudo`` or install with ``--user`` flag:

.. code-block:: bash

    pip install --user -e .

**Network Timeout**

If you encounter network timeouts during installation:

.. code-block:: bash

    pip install --timeout=300 -e .

**Dependency Conflicts**

If you encounter dependency conflicts, try creating a fresh virtual environment:

.. code-block:: bash

    python -m venv fresh_env
    source fresh_env/bin/activate
    pip install -e .

Platform-Specific Notes
------------------------

**macOS Apple Silicon (M1/M2)**

Some dependencies may need special handling:

.. code-block:: bash

    # Install using conda for better ARM64 support
    conda install -c conda-forge networkx pandas matplotlib
    pip install -e .

**Windows**

On Windows, you may need to install Visual Studio Build Tools for some dependencies:

1. Download and install Visual Studio Build Tools
2. Or use conda: ``conda install -c conda-forge networkx pandas matplotlib``

Updating
--------

To update to the latest version:

.. code-block:: bash

    git pull origin main
    pip install -e .

Uninstalling
------------

To remove Network Analyzer:

.. code-block:: bash

    pip uninstall network-analyzer

Next Steps
----------

After installation, check out the :doc:`quickstart` guide to begin using Network Analyzer.