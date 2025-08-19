#!/bin/bash

# Template Configuration for Remote Sync
# Copy this file to config.sh and customize for your environment

# Remote server hostname/IP
REMOTE_HOST="your-server-hostname"

# Parent directory on remote where nki-autotune will be synced
# The final path will be: ${REMOTE_PARENT_DIR}/nki-autotune
REMOTE_PARENT_DIR="/path/to/parent/directory"

# Optional: Additional rsync flags (e.g., "--dry-run" for testing)
ADDITIONAL_RSYNC_FLAGS=""
