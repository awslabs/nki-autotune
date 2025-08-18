#!/bin/bash

# Remote execution script for nki-autotune development
# Executes commands on remote development instance with proper environment setup

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

# Check if command was provided
if [[ $# -eq 0 ]]; then
    echo "❌ Error: No command provided!"
    echo ""
    echo "Usage: $0 <command>"
    echo "Example: $0 'python examples/gemm.py --mode both'"
    exit 1
fi

# Combine all arguments into a single command
COMMAND="$*"

# Default SSH options if not specified
SSH_OPTIONS=${SSH_OPTIONS:-""}

# Build the remote command
REMOTE_CMD=""

# Add virtual environment activation if specified
if [[ -n "$REMOTE_VENV_PATH" ]]; then
    REMOTE_CMD="source ${REMOTE_VENV_PATH}/bin/activate && "
fi

# Add directory change and the actual command
REMOTE_CMD="${REMOTE_CMD}cd ${REMOTE_CODE_PATH} && ${COMMAND}"

echo "🚀 Executing on remote..."
echo "   Host: ${REMOTE_HOST}"
echo "   Dir:  $REMOTE_CODE_PATH"
if [[ -n "$REMOTE_VENV_PATH" ]]; then
    echo "   Venv: $REMOTE_VENV_PATH"
fi
echo "   Cmd:  $COMMAND"
echo ""

# Execute the command
ssh $SSH_OPTIONS "${REMOTE_HOST}" "$REMOTE_CMD"
