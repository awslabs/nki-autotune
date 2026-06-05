#!/usr/bin/env bash
#
# Shared constants and helpers for the autotune transport shells.
# Sourced by ssh_host.sh and kaizen.sh — not executed directly.

# --- Hardcoded cache roots (edit here to change for all transports) ---
# MUST live under $HOME on the remote: only $HOME is S3-backed on Kaizen
# and visible to the reverse s5cmd sync. /ustore/* is ephemeral/invisible.
# The literal \$HOME is intentional — it expands on the remote box, not here.
transport_cache_root_dir="\$HOME/autotune_cache"
local_cache_root_dir="/workplace/weittang/autotune_cache"

# Local path to the nki-autotune repo (the dir to sync to the box).
repo_root_dir="/workplace/weittang/nki-autotune"

# Remote subdir under the box's $HOME where the repo lands.
remote_repo_subdir="nki-autotune"

# Line that activates the kernel-env venv on the remote box before running.
# Overridable via AUTOTUNE_REMOTE_ACTIVATE for boxes with a different layout
# (e.g. the Kaizen py312 conda image). The literal \$HOME expands remotely.
remote_activate="${AUTOTUNE_REMOTE_ACTIVATE:-source \$HOME/venvs/kernel-env/bin/activate}"

# Neuron platform target exported on the box for the run.
neuron_platform_target="${NEURON_PLATFORM_TARGET_OVERRIDE:-trn2}"

# Files synced INTO the repo on the box should exclude these.
sync_excludes=(.git __pycache__ "*.pyc" .pytest_cache .mypy_cache build .venv node_modules)

die() {
    echo "ERROR: $*" >&2
    exit 1
}

# Parse --cmd / --no-setup (shared between both transports). Each transport
# parses its own --host / --name first and passes the rest here.
USER_CMD=""
NO_SETUP=0
parse_common_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --cmd) USER_CMD="$2"; shift 2 ;;
            --no-setup) NO_SETUP=1; shift ;;
            *) die "unknown argument: $1" ;;
        esac
    done
    [[ -n "$USER_CMD" ]] || die "--cmd is required"
}

# The full remote command: activate venv, cd into the repo, export the
# platform target, run the user's command with --cache-root-dir injected.
remote_run_cmd() {
    printf '%s && cd ~/%s && export NEURON_PLATFORM_TARGET_OVERRIDE=%s && %s --cache-root-dir %s' \
        "$remote_activate" "$remote_repo_subdir" "$neuron_platform_target" \
        "$USER_CMD" "$transport_cache_root_dir"
}

# The remote env-setup command (idempotent). Installs the kernel-env venv
# under the box's $HOME so it survives regardless of the box's default paths.
remote_setup_cmd() {
    printf 'AUTOTUNE_VENV_DIR=$HOME/venvs/kernel-env bash ~/%s/install_neuron.sh --local' \
        "$remote_repo_subdir"
}
