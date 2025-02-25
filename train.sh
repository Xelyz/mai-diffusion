#!/bin/bash

if [ -z "$VIRTUAL_ENV" ]; then
    source .venv/bin/activate
fi

python main.py fit --config configs/mai/autoencoder_cli_tap.yaml --seed 114514