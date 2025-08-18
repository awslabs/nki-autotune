# Remote Development Tools

This directory contains tools for hybrid development with NKI autotune, allowing you to edit code locally while running it on remote Trainium/Inferentia instances.

## Quick Start

1. **Set up your remote instance** (one-time):
   - Install Python virtual environment on your remote Trainium instance
   - Clone/sync this repository to your remote instance
   - Install dependencies in the remote venv

2. **Configure local connection**:
   ```bash
   cp remote_tools/.remote_config.template remote_tools/.remote_config
   # Edit remote_tools/.remote_config with your settings
   ```

3. **Start developing**:
   ```bash
   python remote_tools/dev_run.py --auto-fetch -- examples/gemm.py --mode both
   ```

## Configuration

### Required Setup

Create `remote_tools/.remote_config` with your remote server details:

```bash
# SSH hostname/alias (should be configured in your ~/.ssh/config)
REMOTE_HOST=your-remote-host

# Path to the nki-autotune project on remote machine  
REMOTE_CODE_PATH=/path/to/remote/nki-autotune

# Path to the virtual environment on remote machine (optional)
REMOTE_VENV_PATH=/path/to/remote/venv
```

**Note**: The tools rely on your SSH configuration in `~/.ssh/config` for authentication and user details.

### Optional Settings

You can also add these optional configuration variables:

```bash
# Custom rsync options (default excludes .git, __pycache__, etc.)
RSYNC_OPTIONS="-av --exclude='.git' --exclude='__pycache__' --exclude='*.pyc'"

# Custom SSH options
SSH_OPTIONS="-o ConnectTimeout=10"
```

## Tools

### `dev_run.py` - Main Development Tool
**The tool you'll use most often** - combines sync and execution in one command.

#### Argument Separation with `--`
The tool uses the standard `--` separator to clearly distinguish between `dev_run.py` arguments and target script arguments:

```bash
python remote_tools/dev_run.py [DEV_RUN_OPTIONS] -- [TARGET_SCRIPT] [SCRIPT_ARGS...]
```

**Why `--` separator?**
- **Unambiguous**: Clear separation prevents argument conflicts
- **Standard**: Follows common CLI conventions (used by docker, git, etc.)
- **Flexible**: Supports any target script arguments without interference
- **Reliable**: No risk of confusion between wrapper and script arguments

#### Usage Examples

```bash
# Run GEMM benchmarks with script arguments
python remote_tools/dev_run.py --auto-fetch -- examples/gemm.py --mode both

# Run specific GEMM mode
python remote_tools/dev_run.py --auto-fetch -- examples/gemm.py --mode lhs_rhs

# Run script with automatic cache fetching (recommended)
python remote_tools/dev_run.py --auto-fetch -- examples/transpose.py

# Run tests
python remote_tools/dev_run.py -- -m pytest autotune/test/test_matmul.py -v

# Run any Python script with arguments
python remote_tools/dev_run.py -- examples/softmax.py --arg1 value1 --arg2 value2

# Run script without auto-fetch
python remote_tools/dev_run.py -- examples/gemm.py --mode both
```

**Features**:
- Automatically syncs your local changes first
- Smart command handling (auto-prepends `python` for .py files)
- Handles Python module execution (`-m pytest`)
- **Auto-fetch**: Automatically detects and downloads generated cache after script execution
- **Professional progress bars**: Shows percentage, transfer rate, and ETA using tqdm
- **Smart sync**: Two-phase process (scan → transfer) with accurate file counting
- **Dynamic progress**: Progress bars adapt when additional files are discovered
- **Cache exclusion**: Prevents uploading downloaded cache back to remote
- Clear progress display with success/failure status

### `remote_sync.sh` - File Synchronization
Syncs your local changes to the remote instance using rsync.

```bash
./remote_tools/remote_sync.sh
```

**Features**:
- Efficient sync (only changed files)
- Excludes unnecessary files (.git, __pycache__, etc.)
- Clear status messages and error handling
- Respects custom rsync options from config

### `remote_exec.sh` - Command Execution  
Executes commands on the remote instance with proper environment setup.

```bash
./remote_tools/remote_exec.sh 'python examples/gemm.py --help'
./remote_tools/remote_exec.sh 'python --version'
```

**Features**:
- Automatically activates remote virtual environment
- Changes to correct remote directory
- Streams command output back to you
- Handles SSH options and provides execution details

## Enhanced Progress Display

The `dev_run.py` tool features intelligent progress bars with professional visual feedback:

### Upload Progress
```
🔍 Scanning files to transfer...
📤 Uploading:  50%|██████████          | 1/2 files [00:02<00:02]
📤 Uploading: 100%|████████████████████| 2/2 files [00:02<00:00]
✅ Successfully uploaded 4 files
```

### Download Progress  
```
📦 Downloading: 47 files [00:03]
📦 Downloading: 94 files [00:05]
✅ Successfully downloaded 94 files for nki_tile_transpose
```

**Key Features:**
- **Smart upload scanning**: Two-phase operation for uploads (scan → transfer) with accurate file counting
- **Dynamic progress**: Upload progress bars adapt when additional files are discovered
- **Simple download tracking**: Shows file count and elapsed time for downloads
- **Cache exclusion**: Downloaded cache files are never uploaded back to remote

## Log Access & Analysis

The autotune framework saves detailed performance metrics, error logs, and cache data on the remote machine. Use the `--auto-fetch` flag to automatically download these after script execution.

### Understanding Cache Structure
The autotune cache follows this directory structure:
```
/mnt/efs/autotune-dev-cache/
├── {kernel_name}/
│   ├── {input_tensor_shapes}/
│   │   ├── {config_params}/
│   │   │   ├── performance_result.pkl    # Detailed metrics
│   │   │   ├── compilation_output.log    # Compiler logs
│   │   │   └── execution_output.log      # Runtime logs
│   │   └── perf_metrics.json            # Workload summary
```

## Workflow Examples

### Typical Development Session
```bash
# Make some code changes locally...

# Run benchmarks with automatic cache download
python remote_tools/dev_run.py --auto-fetch -- examples/gemm.py --mode both

# Run tests to verify changes
python remote_tools/dev_run.py -- -m pytest autotune/test/ -v

# Try another example with auto-fetch for analysis
python remote_tools/dev_run.py --auto-fetch -- examples/softmax.py

# Run without cache download
python remote_tools/dev_run.py -- examples/transpose.py
```

### Advanced Usage
```bash
# Use individual tools for more control
./remote_tools/remote_sync.sh
./remote_tools/remote_exec.sh 'pip install -e .'
./remote_tools/remote_exec.sh 'python examples/gemm.py --mode lhs_rhs'
```

## SSH Configuration

For best experience, configure SSH connection in your `~/.ssh/config`:

```
Host your-remote-host
    HostName your.ec2.instance.com
    User your-username
    IdentityFile ~/.ssh/your-key.pem
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

This allows you to use `your-remote-host` as the `REMOTE_HOST` in your config.

## Troubleshooting

### Connection Issues
- Verify SSH access: `ssh your-remote-host`
- Check your `remote_tools/.remote_config` settings
- Ensure the remote paths exist and are accessible

### Sync Issues  
- Check rsync is available: `which rsync`
- Verify remote directory permissions
- Try running `./remote_tools/remote_sync.sh` directly to see detailed output

### Execution Issues
- Verify the remote virtual environment path
- Check that NKI autotune is installed on the remote instance
- Ensure your remote instance has Trainium/Inferentia hardware drivers

### Performance
- Consider adjusting `RSYNC_OPTIONS` for your specific needs
- Keep the remote instance "warm" to avoid cold start delays
- Use individual shell scripts for fine-grained control when needed
