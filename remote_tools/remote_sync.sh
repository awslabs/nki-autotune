#!/bin/bash

# Remote sync script for nki-autotune development
# Syncs local changes to remote development instance

set -e  # Exit on any error

# Configuration file path
CONFIG_FILE="remote_tools/.remote_config"

# Check if config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "❌ Error: Configuration file '$CONFIG_FILE' not found!"
    echo ""
    echo "Please create it by copying the template:"
    echo "  cp .remote_config.template .remote_config"
    echo ""
    echo "Then edit .remote_config with your remote server details."
    exit 1
fi

# Source the configuration
source "$CONFIG_FILE"

# Validate required variables
if [[ -z "$REMOTE_HOST" || -z "$REMOTE_CODE_PATH" ]]; then
    echo "❌ Error: Missing required configuration in $CONFIG_FILE"
    echo "Required variables: REMOTE_HOST, REMOTE_CODE_PATH"
    exit 1
fi

# Default rsync options if not specified
RSYNC_OPTIONS=${RSYNC_OPTIONS:-"-av --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='.remote_config'"}

# Construct remote path
REMOTE_PATH="${REMOTE_HOST}:${REMOTE_CODE_PATH}/"

echo "🔄 Syncing to remote..."
echo "   Local:  $(pwd)"
echo "   Remote: $REMOTE_PATH"
echo ""

# Perform the sync
if rsync $RSYNC_OPTIONS ./ "$REMOTE_PATH"; then
    echo ""
    echo "✅ Sync completed successfully!"
else
    echo ""
    echo "❌ Sync failed!"
    exit 1
fi
