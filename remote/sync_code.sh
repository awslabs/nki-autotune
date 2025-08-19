#!/bin/bash

# Code Sync Script for nki-autotune on Remote Server
# Uses configuration from config.sh to sync code only

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load configuration
CONFIG_FILE="$SCRIPT_DIR/config.sh"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Configuration file not found at $CONFIG_FILE"
    echo "Please copy config.template.sh to config.sh and customize it"
    exit 1
fi

source "$CONFIG_FILE"

# Validate required configuration
if [[ -z "$REMOTE_HOST" || -z "$REMOTE_PARENT_DIR" ]]; then
    echo "Error: Missing required configuration. Please check config.sh"
    echo "Required: REMOTE_HOST, REMOTE_PARENT_DIR"
    exit 1
fi

# Derive paths
REMOTE_REPO_DIR="$REMOTE_PARENT_DIR/nki-autotune"
LOCAL_REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"  # Parent of remote directory

echo "=== nki-autotune Code Sync ==="
echo "Local directory: $LOCAL_REPO_DIR"
echo "Remote target: $REMOTE_HOST:$REMOTE_REPO_DIR"
echo ""

# Step 1: Create remote directory structure if it doesn't exist
echo "1. Creating remote directory structure..."
ssh "$REMOTE_HOST" "mkdir -p $REMOTE_PARENT_DIR"

# Step 2: Sync repository using rsync with .gitignore exclusions
echo "2. Syncing code to remote server..."
rsync -avz --delete \
    --exclude='__pycache__/' \
    --exclude='*.py[cod]' \
    --exclude='*.so' \
    --exclude='build/' \
    --exclude='develop-eggs/' \
    --exclude='dist/' \
    --exclude='downloads/' \
    --exclude='eggs/' \
    --exclude='.eggs/' \
    --exclude='lib/' \
    --exclude='lib64/' \
    --exclude='parts/' \
    --exclude='sdist/' \
    --exclude='var/' \
    --exclude='wheels/' \
    --exclude='*.egg-info/' \
    --exclude='.installed.cfg' \
    --exclude='*.egg' \
    --exclude='MANIFEST' \
    --exclude='.Python' \
    --exclude='*.whl' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='*.pyd' \
    --exclude='pip-log.txt' \
    --exclude='pip-delete-this-directory.txt' \
    --exclude='htmlcov/' \
    --exclude='.tox/' \
    --exclude='.coverage' \
    --exclude='.coverage.*' \
    --exclude='.cache' \
    --exclude='nosetests.xml' \
    --exclude='coverage.xml' \
    --exclude='*.cover' \
    --exclude='.hypothesis/' \
    --exclude='.pytest_cache/' \
    --exclude='.env' \
    --exclude='.venv' \
    --exclude='env/' \
    --exclude='venv/' \
    --exclude='ENV/' \
    --exclude='env.bak/' \
    --exclude='venv.bak/' \
    --exclude='.vscode/' \
    --exclude='.idea/' \
    --exclude='*.swp' \
    --exclude='*.swo' \
    --exclude='.DS_Store' \
    --exclude='Thumbs.db' \
    --exclude='ehthumbs.db' \
    --exclude='Desktop.ini' \
    --exclude='*.log' \
    --exclude='*.csv' \
    --exclude='*.tsv' \
    --exclude='*.xlsx' \
    --exclude='*.pdf' \
    --exclude='*.png' \
    --exclude='*.txt' \
    --exclude='*.pkl' \
    --exclude='*.out' \
    --exclude='*.err' \
    --exclude='artifacts/' \
    --exclude='private' \
    --exclude='*.neff' \
    --exclude='*.ntff' \
    --exclude='remote/config.sh' \
    --exclude='cache/' \
    --exclude='generated_kernels/' \
    -e ssh \
    $ADDITIONAL_RSYNC_FLAGS \
    "$LOCAL_REPO_DIR/" "$REMOTE_HOST:$REMOTE_REPO_DIR/"

echo ""
echo "=== Code Sync Complete ==="
echo "✓ Code synced to $REMOTE_HOST:$REMOTE_REPO_DIR"
echo ""
echo "To connect to remote: ssh $REMOTE_HOST"
echo "To navigate to repo: cd $REMOTE_REPO_DIR"
