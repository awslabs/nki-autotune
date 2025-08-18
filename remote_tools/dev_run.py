#!/usr/bin/env python3

"""
Remote development runner for nki-autotune
Automatically syncs local changes and executes commands on remote Trainium instance
"""

import argparse
import subprocess
import sys
from pathlib import Path


def load_config():
    """Load configuration from .remote_config file"""
    config_file = Path("remote_tools/.remote_config")

    if not config_file.exists():
        print("❌ Error: Configuration file '.remote_config' not found!")
        print("")
        print("Please create it by copying the template:")
        print("  cp .remote_config.template .remote_config")
        print("")
        print("Then edit .remote_config with your remote server details.")
        sys.exit(1)

    config = {}
    with open(config_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                config[key.strip()] = value.strip()

    # Validate required config
    required = ["REMOTE_HOST", "REMOTE_CODE_PATH"]
    missing = [key for key in required if key not in config or not config[key]]

    if missing:
        print(f"❌ Error: Missing required configuration: {', '.join(missing)}")
        print("Please check your .remote_config file.")
        sys.exit(1)

    return config


def run_command(cmd, description=None):
    """Run a command and return success status"""
    if description:
        print(f"🔄 {description}")

    try:
        result = subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {cmd}")
        print(f"Exit code: {e.returncode}")
        return False


def sync_to_remote():
    """Sync local changes to remote instance"""
    from tqdm import tqdm

    print("=" * 60)
    print("STEP 1: Syncing local changes to remote...")
    print("=" * 60)

    # Load config
    config = load_config()

    remote_host = config["REMOTE_HOST"]
    remote_path = config["REMOTE_CODE_PATH"]
    ssh_options = config.get("SSH_OPTIONS", "")
    rsync_options = config.get(
        "RSYNC_OPTIONS",
        "-av --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='.remote_config' --exclude='cache'",
    )

    print("🔄 Syncing to remote...")
    print(f"   Local:  {Path.cwd()}")
    print(f"   Remote: {remote_host}:{remote_path}/")

    try:
        # Phase 1: Dry run to determine actual files to transfer
        print("🔍 Scanning files to transfer...")
        full_remote_path = f"{remote_host}:{remote_path}/"
        dry_run_cmd = f"rsync {rsync_options} --dry-run {ssh_options} ./ {full_remote_path}"

        dry_run_result = subprocess.run(dry_run_cmd, shell=True, capture_output=True, text=True)

        if dry_run_result.returncode != 0:
            print("⚠️  Could not scan files, using basic progress...")
            total_files = None
        else:
            # Parse dry-run output to count actual file transfers
            total_files = 0
            for line in dry_run_result.stdout.split("\n"):
                line = line.strip()
                # Count lines that represent file transfers (not directories, metadata, or status messages)
                if (
                    line
                    and not line.endswith("/")
                    and not line.startswith("receiving")
                    and not line.startswith("sent")
                    and not line.startswith("total size")
                    and not line.startswith("building file list")
                    and not line.startswith("created directory")
                ):
                    if "/" in line or ("." in line and not line.startswith(".")):
                        total_files += 1

        # Phase 2: Actual transfer with accurate progress
        rsync_cmd = f"rsync {rsync_options} --progress {ssh_options} ./ {full_remote_path}"

        # Run rsync with real-time progress bar
        process = subprocess.Popen(
            rsync_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        # Initialize progress bar with accurate count
        if total_files and total_files > 0:
            pbar = tqdm(
                total=total_files,
                desc="📤 Uploading",
                unit="files",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} files [{elapsed}<{remaining}]",
            )
        else:
            pbar = tqdm(desc="📤 Uploading", unit="files", bar_format="{desc}: {n_fmt} files [{elapsed}]")

        files_processed = 0

        # Process rsync output line by line
        if process.stdout:
            for line in process.stdout:
                line = line.strip()
                # Look for file transfer lines (not directories or metadata)
                if (
                    line
                    and not line.endswith("/")
                    and not line.startswith("receiving")
                    and not line.startswith("sent")
                    and not line.startswith("total size")
                ):
                    if (
                        "/" in line
                        and not line.startswith("building file list")
                        and not line.startswith("created directory")
                    ):
                        # This looks like a file being transferred
                        files_processed += 1

                        # If we exceed the predicted total, expand the progress bar
                        if total_files and files_processed > total_files:
                            pbar.total = files_processed + 10  # Add buffer for more potential files

                        pbar.n = files_processed
                        pbar.refresh()

        # Wait for process to complete
        process.wait()

        # Close progress bar
        pbar.close()

        if process.returncode == 0:
            print(f"✅ Successfully uploaded {files_processed} files")
            return True
        else:
            print(f"❌ rsync failed with return code {process.returncode}")
            return False

    except Exception as e:
        print(f"❌ Sync failed: {e}")
        return False


def execute_remote(command):
    """Execute command on remote instance"""
    print("=" * 60)
    print("STEP 2: Executing command on remote...")
    print("=" * 60)

    exec_script = Path("remote_tools/remote_exec.sh")
    if not exec_script.exists():
        print(f"❌ Error: Execution script not found: {exec_script}")
        return False

    # Escape the command properly for shell execution
    escaped_command = command.replace('"', '\\"')
    return run_command(f'{exec_script} "{escaped_command}"')


def detect_recent_cache_dirs(config, cache_root="/mnt/efs/autotune-dev-cache"):
    """Detect recently modified cache directories on remote"""
    remote_host = config["REMOTE_HOST"]
    ssh_options = config.get("SSH_OPTIONS", "")

    # Find directories modified in the last 5 minutes
    find_cmd = f'ssh {ssh_options} {remote_host} "find {cache_root} -type d -name "id*" -newermt \\"5 minutes ago\\" | head -10"'

    try:
        result = subprocess.run(find_cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            recent_dirs = []
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    # Extract kernel name from path like /mnt/efs/autotune-dev-cache/nki_tile_transpose/129x128/id0
                    parts = line.strip().split("/")
                    if len(parts) >= 5:
                        kernel_name = parts[4]  # Extract kernel name
                        kernel_path = f"{cache_root}/{kernel_name}"
                        if kernel_path not in recent_dirs:
                            recent_dirs.append(kernel_path)
            return recent_dirs
        return []
    except Exception as e:
        print(f"⚠️  Could not detect recent cache directories: {e}")
        return []


def auto_fetch_recent_logs(config, local_dir):
    """Automatically fetch recently generated cache directories"""
    print("=" * 60)
    print("STEP 3: Auto-fetching recent cache...")
    print("=" * 60)

    recent_dirs = detect_recent_cache_dirs(config)

    if not recent_dirs:
        print("⚠️  No recent cache directories detected.")
        print("💡 Try running the command again or check if the script generated cache files.")
        return False

    print(f"🔍 Found {len(recent_dirs)} recently generated cache director{'ies' if len(recent_dirs) > 1 else 'y'}:")
    for dir_path in recent_dirs:
        print(f"   {dir_path}")

    # Fetch all recent directories
    success = True
    for remote_path in recent_dirs:
        kernel_name = remote_path.split("/")[-1]
        print(f"\n📦 Fetching cache for: {kernel_name}")
        if not fetch_logs(remote_path, local_dir, config):
            success = False

    return success


def fetch_logs(remote_path, local_dir, config):
    """Download logs from remote cache to local directory"""
    from tqdm import tqdm

    # Create local directory if it doesn't exist
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)

    remote_host = config["REMOTE_HOST"]
    ssh_options = config.get("SSH_OPTIONS", "")

    print(f"📁 Remote path: {remote_path}")
    print(f"📁 Local dir:   {local_dir}")

    try:
        # Direct transfer with basic progress bar
        rsync_cmd = f"rsync -av --progress {ssh_options} {remote_host}:{remote_path} {local_dir}/"

        # Run rsync with real-time progress bar
        process = subprocess.Popen(
            rsync_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        # Initialize progress bar without predetermined total
        pbar = tqdm(desc="📦 Downloading", unit="files", bar_format="{desc}: {n_fmt} files [{elapsed}]")

        files_processed = 0

        # Process rsync output line by line
        if process.stdout:
            for line in process.stdout:
                line = line.strip()
                # Look for file transfer lines (not directories or metadata)
                if (
                    line
                    and not line.endswith("/")
                    and not line.startswith("receiving")
                    and not line.startswith("sent")
                    and not line.startswith("total size")
                ):
                    if (
                        "/" in line
                        and not line.startswith("building file list")
                        and not line.startswith("created directory")
                    ):
                        # This looks like a file being transferred
                        files_processed += 1
                        pbar.n = files_processed
                        pbar.refresh()

        # Wait for process to complete
        process.wait()

        # Close progress bar
        pbar.close()

        if process.returncode == 0:
            kernel_name = remote_path.split("/")[-1]
            print(f"✅ Successfully downloaded {files_processed} files for {kernel_name}")
            return True
        else:
            print(f"❌ rsync failed with return code {process.returncode}")
            return False

    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Sync local changes and run commands on remote Trainium instance",
        epilog="""
Examples:
  %(prog)s --auto-fetch -- examples/gemm.py --mode both
  %(prog)s --local-dir ./results -- examples/transpose.py --size 1024
  %(prog)s -- python examples/softmax.py --batch-size 32
  %(prog)s -- -m pytest autotune/test/test_matmul.py -v
  %(prog)s -- examples/gemm.py  # No script arguments needed

Note: Use '--' to separate dev_run.py arguments from target script arguments.
      Everything after '--' is passed directly to the target script.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--local-dir", metavar="DIR", default="./cache", help="Local directory for downloaded logs (default: ./cache)"
    )

    parser.add_argument(
        "--auto-fetch", action="store_true", help="Automatically fetch generated cache after running script"
    )

    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to execute on remote instance (use -- to separate from dev_run.py arguments)",
    )

    args = parser.parse_args()

    # Extract command arguments (everything after --)
    all_command_args = args.command

    # Handle the case where -- was used explicitly
    if all_command_args and all_command_args[0] == "--":
        all_command_args = all_command_args[1:]

    # Load and validate configuration
    config = load_config()

    print(f"🎯 Remote Development Runner")
    print(f"   Target: {config['REMOTE_HOST']}")
    print(f"   Path:   {config['REMOTE_CODE_PATH']}")
    if "REMOTE_VENV_PATH" in config and config["REMOTE_VENV_PATH"]:
        print(f"   Venv:   {config['REMOTE_VENV_PATH']}")
    print("")

    success = True

    # Validate command is provided for normal operation
    if not all_command_args:
        print("❌ Error: No command provided!")
        print("")
        print("Usage: python dev_run.py [OPTIONS] -- COMMAND [COMMAND_ARGS...]")
        print("Use --help for detailed usage information and examples.")
        sys.exit(1)

    # Validate that the target script exists if it's a local Python file
    target_script = all_command_args[0]
    if target_script.endswith(".py") and not target_script.startswith("-"):
        script_path = Path(target_script)
        if not script_path.exists():
            print(f"❌ Error: Python script not found: {target_script}")
            print("Make sure the script path is correct relative to the current directory.")
            sys.exit(1)

    # Join command arguments
    command = " ".join(all_command_args)

    # Handle Python module execution (e.g., -m pytest)
    if all_command_args[0] == "-m":
        command = f"python {command}"
    elif not command.startswith("python") and target_script.endswith(".py"):
        # Auto-prepend python for .py files
        command = f"python {command}"

    # Step 1: Always sync local changes to remote
    success = sync_to_remote()
    if not success:
        print("\n❌ Sync failed, aborting.")
        sys.exit(1)

    # Step 2: Execute command on remote
    success = execute_remote(command)
    if not success:
        print("\n❌ Development run failed!")
        sys.exit(1)

    # Step 3: Auto-fetch logs if requested
    if args.auto_fetch:
        auto_success = auto_fetch_recent_logs(config, args.local_dir)
        if auto_success:
            print("\n" + "=" * 60)
            print("✅ Development run and auto-fetch completed successfully!")
        else:
            print("\n" + "=" * 60)
            print("✅ Development run completed successfully!")
            print("⚠️  Auto-fetch did not find recent cache directories.")
    else:
        print("\n" + "=" * 60)
        print("✅ Development run completed successfully!")


if __name__ == "__main__":
    main()
