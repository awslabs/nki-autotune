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
   python remote_tools/dev_run.py examples/gemm.py --mode both
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

```bash
# Run GEMM benchmarks
python remote_tools/dev_run.py examples/gemm.py --mode both

# Run tests
python remote_tools/dev_run.py -m pytest autotune/test/test_matmul.py -v

# Run any Python script
python remote_tools/dev_run.py examples/softmax.py

# Just sync files without running anything
python remote_tools/dev_run.py --sync-only "echo done"

# Skip syncing, just run command
python remote_tools/dev_run.py --no-sync "python --version"

# Access logs from remote cache
python remote_tools/dev_run.py --fetch-logs /mnt/efs/autotune-dev-cache/transpose/129x128
python remote_tools/dev_run.py --fetch-logs /mnt/efs/autotune-dev-cache/nki_tile_transpose --local-dir ./my_logs
```

**Features**:
- Automatically syncs your local changes first
- Smart command handling (auto-prepends `python` for .py files)
- Handles Python module execution (`-m pytest`)
- Clear progress display with success/failure status
- Options for sync-only or execute-only modes
- Log access functionality to download cache/results from remote

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

## Log Access & Analysis

The autotune framework saves detailed performance metrics, error logs, and cache data on the remote machine. You can access these logs during development:

### Fetching Logs
```bash
# Download entire cache directory for a kernel
python remote_tools/dev_run.py --fetch-logs /mnt/efs/autotune-dev-cache/nki_tile_transpose

# Download to a specific local directory
python remote_tools/dev_run.py --fetch-logs /mnt/efs/autotune-dev-cache/transpose/129x128 --local-dir ./analysis

# Download specific performance results
python remote_tools/dev_run.py --fetch-logs /mnt/efs/autotune-dev-cache/gemm/1024x2048x4096/M1024-N2048-K4096
```

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

# Run benchmarks with real-time feedback
python remote_tools/dev_run.py examples/gemm.py --mode both --cache-dir /mnt/efs/autotune-dev-cache

# Run tests to verify changes
python remote_tools/dev_run.py -m pytest autotune/test/ -v

# Download results for analysis
python remote_tools/dev_run.py --fetch-logs /mnt/efs/autotune-dev-cache/gemm --local-dir ./gemm_results

# Try another example
python remote_tools/dev_run.py examples/softmax.py
```

### Advanced Usage
```bash
# Sync files only (no execution)
python remote_tools/dev_run.py --sync-only "echo sync complete"

# Execute without syncing (useful if no local changes)
python remote_tools/dev_run.py --no-sync "python examples/attention.py"

# Use individual tools for more control
./remote_tools/remote_sync.sh
./remote_tools/remote_exec.sh 'pip install -e .'
./remote_tools/remote_exec.sh 'python examples/gemm.py --mode lhs_rhs'

# Download logs from multiple runs for comparison
python remote_tools/dev_run.py --fetch-logs /mnt/efs/autotune-dev-cache/matmul --local-dir ./matmul_analysis
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
- Use `--no-sync` when running multiple commands without local changes
- Consider adjusting `RSYNC_OPTIONS` for your specific needs
- Keep the remote instance "warm" to avoid cold start delays
