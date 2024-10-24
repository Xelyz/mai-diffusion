#!/bin/bash

if [ -z "$VIRTUAL_ENV" ]; then
    source .venv/bin/activate
fi

# Run the Python script
echo "Running webui.py..."
python webui.py