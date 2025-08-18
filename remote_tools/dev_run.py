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
    print("=" * 60)
    print("STEP 1: Syncing local changes to remote...")
    print("=" * 60)

    sync_script = Path("remote_tools/remote_sync.sh")
    if not sync_script.exists():
        print(f"❌ Error: Sync script not found: {sync_script}")
        return False

    return run_command(str(sync_script))


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


def fetch_logs(remote_path, local_dir, config):
    """Download logs from remote cache to local directory"""
    print("=" * 60)
    print("FETCHING LOGS from remote...")
    print("=" * 60)

    # Create local directory if it doesn't exist
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)

    remote_host = config["REMOTE_HOST"]
    ssh_options = config.get("SSH_OPTIONS", "")

    # Build rsync command to download logs
    rsync_cmd = f"rsync -av {ssh_options} {remote_host}:{remote_path} {local_dir}/"

    print(f"📁 Remote path: {remote_path}")
    print(f"📁 Local dir:   {local_dir}")
    print(f"🔄 Downloading...")
    print("")

    success = run_command(rsync_cmd, "Downloading logs from remote...")

    if success:
        print(f"\n✅ Logs successfully downloaded to: {local_dir}")
        # List what was downloaded
        try:
            downloaded_files = list(local_path.rglob("*"))
            if downloaded_files:
                print(f"📋 Downloaded {len(downloaded_files)} items:")
                for item in sorted(downloaded_files)[:10]:  # Show first 10 items
                    print(f"   {item.relative_to(local_path)}")
                if len(downloaded_files) > 10:
                    print(f"   ... and {len(downloaded_files) - 10} more items")
        except Exception:
            pass

    return success


def main():
    parser = argparse.ArgumentParser(
        description="Sync local changes and run commands on remote Trainium instance",
        epilog="""
Examples:
  %(prog)s examples/gemm.py --mode both
  %(prog)s "python examples/softmax.py"
  %(prog)s -m pytest autotune/test/test_matmul.py -v
  %(prog)s "python -c 'import autotune; print(autotune.__version__)'"
  
Log Access:
  %(prog)s --fetch-logs /mnt/efs/autotune-dev-cache/transpose/129x128
  %(prog)s --fetch-logs /mnt/efs/autotune-dev-cache/nki_tile_transpose --local-dir ./my_logs
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("command", nargs="*", help="Command to execute on remote instance")

    parser.add_argument("--sync-only", action="store_true", help="Only sync files, do not execute command")

    parser.add_argument("--no-sync", action="store_true", help="Skip syncing, only execute command")

    parser.add_argument(
        "--fetch-logs", metavar="REMOTE_PATH", help="Download logs from remote cache directory to local machine"
    )

    parser.add_argument(
        "--local-dir", metavar="DIR", default="./cache", help="Local directory for downloaded logs (default: ./cache)"
    )

    args = parser.parse_args()

    # Load and validate configuration
    config = load_config()

    print(f"🎯 Remote Development Runner")
    print(f"   Target: {config['REMOTE_HOST']}")
    print(f"   Path:   {config['REMOTE_CODE_PATH']}")
    if "REMOTE_VENV_PATH" in config and config["REMOTE_VENV_PATH"]:
        print(f"   Venv:   {config['REMOTE_VENV_PATH']}")
    print("")

    success = True

    # Handle fetch-logs mode
    if args.fetch_logs:
        success = fetch_logs(args.fetch_logs, args.local_dir, config)
        print("\n" + "=" * 60)
        if success:
            print("✅ Log fetch completed successfully!")
        else:
            print("❌ Log fetch failed!")
            sys.exit(1)
        return

    # Validate command is provided for normal operation
    if not args.command:
        print("❌ Error: No command provided!")
        print("Use --help for usage information.")
        sys.exit(1)

    # Join command arguments
    command = " ".join(args.command)

    # Handle Python module execution (e.g., -m pytest)
    if args.command[0] == "-m":
        command = f"python {command}"
    elif not command.startswith("python") and args.command[0].endswith(".py"):
        # Auto-prepend python for .py files
        command = f"python {command}"

    # Step 1: Sync (unless --no-sync)
    if not args.no_sync:
        success = sync_to_remote()
        if not success:
            print("\n❌ Sync failed, aborting.")
            sys.exit(1)

    # Step 2: Execute (unless --sync-only)
    if not args.sync_only and success:
        success = execute_remote(command)

    print("\n" + "=" * 60)
    if success:
        print("✅ Development run completed successfully!")
    else:
        print("❌ Development run failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
