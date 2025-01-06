#!/bin/bash
# ML environment setup script
# Source: https://github.com/ml-tools/env-setup
# Modified for local use

if [ $# -eq 0 ]; then
    echo "Error: Please provide name of conda virtual environment as an argument."
    echo "Usage: ./init_env.sh MyEnv"
    exit 1
fi

export VirtualEnv=$1

# Create and activate conda environment
conda create -y -n $VirtualEnv python=3.10.14
source ~/.bashrc
conda activate $VirtualEnv

# Install required packages
pip install --upgrade pip
pip install ipykernel
python -m ipykernel install --user --name=$VirtualEnv --display-name "$VirtualEnv"
pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir flash-attn==2.6.3

echo "Environment '$VirtualEnv' created and configured successfully"
echo "To activate: conda activate $VirtualEnv"
echo "To deactivate: conda deactivate"
