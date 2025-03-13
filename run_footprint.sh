#!/bin/bash

# Change to project root directory
cd "$(dirname "$0")"

# Run the footprint script with any provided arguments
python -m modules.utils.footprint "$@"
