# Remote Code Sync Tools

This directory contains tools for syncing the nki-autotune repository code to remote servers.

## Setup

1. Copy the template configuration:
   ```bash
   cd remote
   cp config.template.sh config.sh
   ```

2. Edit `config.sh` to match your environment:
   ```bash
   # Remote server hostname/IP
   REMOTE_HOST="your-server-hostname"
   
   # Parent directory on remote where nki-autotune will be synced
   REMOTE_PARENT_DIR="/path/to/parent/directory"
   ```

## Usage

Run the code sync script:
```bash
./remote/sync_code.sh
```

This will sync the repository code to `${REMOTE_PARENT_DIR}/nki-autotune` on the remote server.

**Note:** This script only syncs code. You need to manually install the package in editable mode on the remote server once:
```bash
ssh your-server-hostname
source /path/to/your/venv/bin/activate
cd /path/to/parent/directory/nki-autotune
pip install -e .
```

After that, subsequent code syncs will automatically be reflected due to the editable installation.

## Configuration

- `config.template.sh`: Template configuration file (tracked in git)
- `config.sh`: Your custom configuration (gitignored)

The sync script automatically excludes files listed in `.gitignore` and the custom `config.sh` file.

## Troubleshooting

- Ensure you can SSH to the remote server without password (use SSH keys)
- Verify the remote parent directory exists and is writable
- Use `ADDITIONAL_RSYNC_FLAGS="--dry-run"` in config.sh to test without making changes
