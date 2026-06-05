#!/usr/bin/env bash
#
# SSH transport: sync repo -> set up env -> run --cmd -> download artifacts.
#
# Usage:
#   transport/ssh_host.sh --host <h> --cmd "python xxx.py" [--no-setup]
#
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

# rsync exclude flags from the shared list.
rsync_excludes=()
for e in "${sync_excludes[@]}"; do
    rsync_excludes+=(--exclude "$e")
done

echo "==> [1/4] Syncing $repo_root_dir/ -> $HOST:~/$remote_repo_subdir/"
rsync -az --delete "${rsync_excludes[@]}" \
    "$repo_root_dir/" "$HOST:$remote_repo_subdir/"

if [[ "$NO_SETUP" -eq 0 ]]; then
    echo "==> [2/4] Setting up env on $HOST (idempotent)"
    ssh "$HOST" "$(remote_setup_cmd)"
else
    echo "==> [2/4] Skipping env setup (--no-setup)"
fi

echo "==> [3/4] Executing on $HOST"
ssh "$HOST" "$(remote_run_cmd)"

echo "==> [4/4] Downloading artifacts -> $local_cache_root_dir/"
mkdir -p "$local_cache_root_dir"
# transport_cache_root_dir holds a literal $HOME — expand it on the remote.
remote_cache="$(ssh "$HOST" "echo $transport_cache_root_dir")"
rsync -az "$HOST:$remote_cache/" "$local_cache_root_dir/"
echo "==> Done. Artifacts in $local_cache_root_dir/"
