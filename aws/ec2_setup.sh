#!/bin/bash
set -e

# Git config
GIT_REPO_URL="https://github.com/cjblack/climbing-analysis.git"
GIT_BRANCH="development"
PROJECT_DIR="climbing-analysis"
PY_ENV_NAME="climb-env"

# Update
sudo apt update
sudo apt install -y git python3-pip python3-venv

# Setup repo
cd ~
if [! -d "$PROJECT_DIR"]; then
    git clone $GIT_REPO_URL
fi

cd "$PROJECT_DIR"
git fetch
git checkout $GIT_BRANCH    
git pull origin $GIT_BRANCH

# Create venv
cd ~
python3 -m venv $PY_ENV_NAME
source $PY_ENV_NAME/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install requirements
pip install -r ~/$PROJECT_DIR/aws/requirements.txt