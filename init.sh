#!/bin/bash

# Try finding Python 3.12 in common locations
PYTHON=${PYTHON:-$(command -v /usr/local/bin/python3.12 || command -v /usr/bin/python3.12)}

# Exit if Python 3.12 is not found
[ -z "$PYTHON" ] && echo "Python 3.12 not found." && exit 1

# Create a virtual environment if it doesn't exist, suppressing output
if [ ! -d .venv ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv .venv
    [ ! -d .venv ] && echo "Failed to create virtual environment." && exit 1
fi

# Activate the virtual environment if it's not already active
if [ -z "$VIRTUAL_ENV" ]; then
    source .venv/bin/activate
fi

# Upgrade pip, setuptools, and wheel, and install required packages with minimal output
echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade -q pip setuptools wheel

echo "Installing required packages..."
pip install -q -r requirements.txt

# Run the Python script
echo "Running webui.py..."
python webui.py