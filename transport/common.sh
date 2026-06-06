#!/usr/bin/env bash
#
# Shared constants and helpers for the autotune transport.
# Sourced by kaizen.sh — not executed directly.

# --- Repo location (edit here to change for all transports) ---
# Local path to the nki-autotune repo (the dir to sync to the box).
repo_root_dir="/workplace/weittang/nki-autotune"

# Remote subdir under the box's $HOME where the repo lands.
remote_repo_subdir="nki-autotune"

# The cache directory is supplied per-run via --cache: an absolute path,
# mirrored on both machines. The run writes there on the remote and the reverse
# sync pulls it back to the SAME path locally. On Kaizen it MUST live under
# $HOME — only $HOME is S3-backed and visible to the reverse sync.

# Line that activates the kernel-env venv on the remote box before running.
# Overridable via AUTOTUNE_REMOTE_ACTIVATE for boxes with a different layout
# (e.g. the Kaizen py312 conda image). The literal \$HOME expands remotely.
remote_activate="${AUTOTUNE_REMOTE_ACTIVATE:-source \$HOME/venvs/kernel-env/bin/activate}"

# PYTHONPATH so a bare `python examples/X.py` resolves the first-party packages.
# install.sh installs only the THIRD-PARTY deps (nkipy/spike); nkigym + autotune
# are imported from the synced source tree. Mirrors what the self-bootstrapping
# examples insert (repo root + both src trees); set relative to ~/<repo>, which
# the commands below cd into. Applied to BOTH preflight and run so a script that
# imports nkigym at module top level (before argparse) doesn't crash either one.
remote_pythonpath="PYTHONPATH=.:nkigym/src:autotune/src"

# Neuron platform target exported on the box for the run.
neuron_platform_target="${NEURON_PLATFORM_TARGET_OVERRIDE:-trn2}"

# Desktop start parameters. Used ONLY when --name's desktop is not RUNNING (e.g.
# it hit its --timeout and went FAILED) and kaizen.sh must start a fresh one.
# Env-overridable; defaults are the validated trn2 + py312 inference DLC. A
# name's $HOME (and thus the kernel-env venv) is S3-backed and survives the
# restart, so a re-started desktop already has the venv from a prior install.sh.
desktop_instance="${KAIZEN_DESKTOP_INSTANCE:-trn2.48xlarge}"
desktop_image="${KAIZEN_DESKTOP_IMAGE:-763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference-neuronx:2.9.0-neuronx-py312-sdk2.30.0-ubuntu24.04}"
desktop_timeout="${KAIZEN_DESKTOP_TIMEOUT:-86400}"

# Files synced INTO the repo on the box should exclude these.
sync_excludes=(.git __pycache__ "*.pyc" .pytest_cache .mypy_cache build .venv node_modules)

die() {
    echo "ERROR: $*" >&2
    exit 1
}

# Parse --cmd / --cache. kaizen.sh parses its own --name first and passes the
# rest here. The venv is NOT set up here — run install.sh on the box yourself
# once (see install.sh).
USER_CMD=""
CACHE_DIR=""
parse_common_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --cmd) USER_CMD="$2"; shift 2 ;;
            --cache) CACHE_DIR="$2"; shift 2 ;;
            *) die "unknown argument: $1" ;;
        esac
    done
    [[ -n "$USER_CMD" ]] || die "--cmd is required"
    [[ -n "$CACHE_DIR" ]] || die "--cache is required"
    [[ "$CACHE_DIR" = /* ]] || die "--cache must be an absolute path (got: $CACHE_DIR)"
    case "$USER_CMD" in
        *.py|*.py\ *) ;;
        *) die "--cmd must run a .py script (got: $USER_CMD)" ;;
    esac
}

# The first .py token of USER_CMD — the script we run and verify accepts --cache.
user_script() {
    local tok
    for tok in $USER_CMD; do
        case "$tok" in *.py) printf '%s' "$tok"; return 0 ;; esac
    done
    die "no .py script found in --cmd: $USER_CMD"
}

# Remote preflight (one round trip): fail fast (exit 3) unless the script's
# --help advertises a --cache option — so a script that ignores --cache can't
# silently drop the reverse-synced artifacts — then print the remote $HOME so
# the caller can map --cache to its S3-export prefix. Runs inside the venv.
remote_check_cmd() {
    printf '%s && cd ~/%s && %s python %s --help 2>&1 | grep -q -- --cache || { echo "TRANSPORT: %s does not accept --cache" >&2; exit 3; }; echo "TRANSPORT_HOME=$HOME"' \
        "$remote_activate" "$remote_repo_subdir" "$remote_pythonpath" "$(user_script)" "$(user_script)"
}

# Name of the transport's completion manifest, written inside --cache as the
# run's LAST action. Lists every other output file as 'size<TAB>relpath' (sizes
# from the lag-free desktop FS). The reverse sync waits for THIS file to export,
# then verifies every listed file landed locally at its stated size — a reliable
# done-signal that needs no cooperation from the user's script.
transport_manifest=".transport_manifest"

# The full remote command. First wipe + recreate --cache so the desktop cache
# holds ONLY this run's output (mirrors the local wipe; also keeps the manifest
# below — built from `find $CACHE_DIR` — free of leftover sibling subfolders).
# Then activate venv, cd into the repo, export the platform target, run the
# user's command with --cache <dir> appended (mirrored path). On success, write
# transport_manifest LAST: a sentinel whose presence in S3 means "run finished"
# and whose contents are the completeness oracle. Built with find -printf,
# excluding the manifest itself; `&&` so a failed run writes no manifest (and
# the transport's own non-zero exit still propagates).
remote_run_cmd() {
    printf 'rm -rf %s && mkdir -p %s && %s && cd ~/%s && export NEURON_PLATFORM_TARGET_OVERRIDE=%s && %s %s --cache %s && find %s -type f ! -name %s -printf "%%s\\t%%P\\n" | LC_ALL=C sort > %s/%s' \
        "$CACHE_DIR" "$CACHE_DIR" \
        "$remote_activate" "$remote_repo_subdir" "$neuron_platform_target" \
        "$remote_pythonpath" "$USER_CMD" "$CACHE_DIR" \
        "$CACHE_DIR" "$transport_manifest" "$CACHE_DIR" "$transport_manifest"
}
