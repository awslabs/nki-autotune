#!/usr/bin/env bash
#
# SSH transport: sync repo -> verify venv -> run --cmd -> reverse-sync cache.
#
# An rsync/ssh sibling of kaizen.sh for an ALWAYS-ON SSH host — the gym-* lab
# Trn2 boxes in ~/.ssh/config (gym-1 = 16.50.111.158, gym-2 = 16.50.231.42, both
# user=ubuntu, key=weittang-melbourne.pem). Use this when the Kaizen fleet is
# unusable: e.g. a host-driver/runtime mismatch that fails nrt_init
# (dmem_buf_copyin / ucode_ll_create err 6). gym-1 runs a RELEASED host driver
# (2.28.0.0) that initializes the device cleanly, where Kaizen's internal
# 2.x.9798.0 build does not.
#
# Why this is simpler than kaizen.sh: rsync is SYNCHRONOUS and bidirectional, so
# there is no S3 ~60s export lag, no completion manifest, and no polling — the
# reverse sync is a single rsync the moment the remote run returns. The SSH host
# is always on, so there is no desktop start/wait/status machinery either.
#
# The 3 steps:
#   [1/3] rsync the whole repo into ~/<repo> on the host (nkigym/ + autotune/
#         included; scripts resolve them from this synced source via PYTHONPATH).
#   [2/3] Run --cmd in the venv. If the script accepts --cache, point it at a
#         scratch dir under the host's $HOME and append `--cache <dir>`;
#         otherwise run it as-is.
#   [3/3] If --cache was appended, rsync that remote dir back to the local
#         --cache path. If not, there are no remote artifacts, so this is
#         skipped — the run's terminal output is in output.log either way.
#
# Before [1/3] we (a) decide LOCALLY whether the --cmd script accepts --cache (a
# static source scan — same as kaizen.sh, a BRANCH not a gate) and (b) tee every
# line of our own stdout+stderr into <--cache>/output.log, so the local --cache
# always carries the run's output even if an infra failure aborts early.
#
# Usage:
#   transport/ssh_host.sh --host <h> --cmd "python path/to/xxx.py" --cache /abs/path
#
# Prerequisites (caller's responsibility — fails loud if missing):
#   - <h> reachable over SSH (entry in ~/.ssh/config with user + IdentityFile)
#   - the kernel-env venv already built on the host (see install.sh)
#   - nkipy + spike editable-installed in that venv (the runner hard-imports them)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=transport/common.sh
source "$SCRIPT_DIR/common.sh"

HOST=""
rest=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host) HOST="$2"; shift 2 ;;
        *) rest+=("$1"); shift ;;
    esac
done
[[ -n "$HOST" ]] || die "--host is required"
parse_common_args ${rest[@]+"${rest[@]}"}

command -v ssh   >/dev/null 2>&1 || die "ssh not on PATH"
command -v rsync >/dev/null 2>&1 || die "rsync not on PATH"

# Non-interactive SSH: BatchMode fails loud instead of hanging on a password
# prompt; StrictHostKeyChecking=no auto-learns the host key (the gym-* lab boxes
# are ephemeral and their host keys rotate — a stale known_hosts entry would
# otherwise abort the transport). Host / user / IdentityFile come from ~/.ssh/config.
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=no)

# The full remote command. $1 is "1" when the user's script accepts --cache and
# "0" otherwise; $2 is the remote scratch cache dir. accepts-cache: wipe +
# recreate the remote cache (so it holds ONLY this run's output — the example
# only wipes its own leaf subdir, not stale siblings), activate the venv, cd into
# the repo, export the platform target, run with --cache <dir> appended. A failed
# step short-circuits the && chain so ssh returns the non-zero exit. no-cache:
# just activate + run as-is.
ssh_remote_run_cmd() {
    local accepts_cache="$1" remote_cache="$2"
    if [[ "$accepts_cache" == "1" ]]; then
        printf 'rm -rf %s && mkdir -p %s && %s && cd ~/%s && export NEURON_PLATFORM_TARGET_OVERRIDE=%s && %s %s --cache %s' \
            "$remote_cache" "$remote_cache" \
            "$remote_activate" "$remote_repo_subdir" "$neuron_platform_target" \
            "$remote_pythonpath" "$USER_CMD" "$remote_cache"
    else
        printf '%s && cd ~/%s && export NEURON_PLATFORM_TARGET_OVERRIDE=%s && %s %s' \
            "$remote_activate" "$remote_repo_subdir" "$neuron_platform_target" \
            "$remote_pythonpath" "$USER_CMD"
    fi
}

# Wipe the local --cache up front so it holds ONLY this run's artifacts, even if
# the run fails partway (a stale local cache would otherwise masquerade as this
# run's output).
rm -rf "$CACHE_DIR"
mkdir -p "$CACHE_DIR"

# Tee every line of our own stdout+stderr from here on into <cache>/output.log
# while still printing to the terminal — so the cache carries a full record of
# the run, INCLUDING an infra failure (host down, venv missing) that aborts
# before the remote run ever happens.
exec > >(tee "$CACHE_DIR/$transport_log") 2>&1
echo "==> Wiped + recreated local cache $CACHE_DIR (run log: $CACHE_DIR/$transport_log)"

# Decide LOCALLY whether the --cmd script accepts --cache, before the round trip.
# Match --cache only as a QUOTED option token, the form an arg parser registers
# (argparse add_argument / click.option) — rejects bare prose in a docstring and
# unrelated longer options ("--cache-dir"). BRANCH, not a gate.
local_script="$(user_script)"
case "$local_script" in
    /*) ;;
    *) local_script="$repo_root_dir/$local_script" ;;
esac
[[ -f "$local_script" ]] || die "script not found locally: $local_script"
SCRIPT_ACCEPTS_CACHE=0
grep -Eq "['\"]--cache['\"]" "$local_script" && SCRIPT_ACCEPTS_CACHE=1
if [[ "$SCRIPT_ACCEPTS_CACHE" == "1" ]]; then
    echo "==> $(user_script) accepts --cache (local scan): will append --cache + reverse-sync artifacts"
else
    echo "==> $(user_script) does not reference --cache (local scan): running as-is; only output.log returns"
fi

echo "==> Preflight: resolve $HOST \$HOME (and confirm the venv is set up)"
CHECK_OUT="$(ssh "${SSH_OPTS[@]}" "$HOST" "$(remote_check_cmd)" 2>&1)" || {
    echo "$CHECK_OUT" | tail -5
    die "preflight failed — $HOST unreachable, or the venv is missing (run install.sh on $HOST)"
}
REMOTE_HOME="$(printf '%s\n' "$CHECK_OUT" | sed -n 's/^TRANSPORT_HOME=//p' | tail -1)"
[[ -n "$REMOTE_HOME" ]] || die "could not resolve remote \$HOME from preflight"
# rsync can pull from any remote path (unlike the Kaizen S3-export-under-$HOME
# constraint), but anchoring the remote cache under $HOME keeps it on persistent
# storage and writable. The leaf mirrors the local --cache basename.
REMOTE_CACHE="$REMOTE_HOME/autotune_cache/$(basename "$CACHE_DIR")"

echo "==> [1/3] Syncing $repo_root_dir/ -> $HOST:~/$remote_repo_subdir/"
rsync_excludes=()
for e in "${sync_excludes[@]}"; do
    rsync_excludes+=(--exclude "$e")
done
rsync -az --delete -e "ssh ${SSH_OPTS[*]}" "${rsync_excludes[@]}" \
    "$repo_root_dir/" "$HOST:$remote_repo_subdir/"

if [[ "$SCRIPT_ACCEPTS_CACHE" == "1" ]]; then
    echo "==> [2/3] Executing on $HOST: $USER_CMD --cache $REMOTE_CACHE"
else
    echo "==> [2/3] Executing on $HOST: $USER_CMD"
fi
ssh "${SSH_OPTS[@]}" "$HOST" "$(ssh_remote_run_cmd "$SCRIPT_ACCEPTS_CACHE" "$REMOTE_CACHE")"

# [3/3] Reverse-sync the remote --cache artifacts back — ONLY when the script
# accepts --cache (otherwise it wrote nothing remotely). rsync is synchronous, so
# this needs no manifest/poll: the moment the remote run returned above, the
# artifacts are complete. No --delete here — the local cache already holds this
# run's output.log, which is local-only and not on the remote.
if [[ "$SCRIPT_ACCEPTS_CACHE" != "1" ]]; then
    echo "==> [3/3] No --cache on the script — skipping artifact reverse-sync; output.log holds the run output"
else
    echo "==> [3/3] Reverse-syncing $HOST:$REMOTE_CACHE/ -> $CACHE_DIR/"
    rsync -az -e "ssh ${SSH_OPTS[*]}" "$HOST:$REMOTE_CACHE/" "$CACHE_DIR/"
fi
echo "==> Done. Cache in $CACHE_DIR/ (run log: $CACHE_DIR/$transport_log)"
