#!/bin/bash

if [ -z "$VIRTUAL_ENV" ]; then
    source .venv/bin/activate
fi

python MAI.py fit --config configs/mai/autoencoder_cli.yaml