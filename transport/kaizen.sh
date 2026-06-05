#!/usr/bin/env bash
#
# Kaizen transport: sync repo -> set up env -> run --cmd -> download artifacts.
#
# Usage:
#   transport/kaizen.sh --name <desktop> --cmd "python xxx.py" [--no-setup]
#
# Prerequisites (caller's responsibility — fails loud if missing):
#   - mwinit -o done; ~/.aws/config has kaizen-access + cluster-role profiles
#   - s5cmd on PATH; kaizen CLI installed
#   - the named desktop is already RUNNING (this script does NOT start one)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=transport/common.sh
source "$SCRIPT_DIR/common.sh"

NAME=""
rest=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --name) NAME="$2"; shift 2 ;;
        *) rest+=("$1"); shift ;;
    esac
done
[[ -n "$NAME" ]] || die "--name is required"
parse_common_args ${rest[@]+"${rest[@]}"}

command -v kaizen >/dev/null 2>&1 || die "kaizen CLI not on PATH"
command -v s5cmd  >/dev/null 2>&1 || die "s5cmd not on PATH"

echo "==> Resolving desktop $NAME (s3SyncUri + region)"
INFO="$(AWS_PROFILE=kaizen-access kaizen desktop info --name "$NAME" --output json 2>&1 | tail -1)"
echo "$INFO" | grep -q '"RUNNING"' || die "desktop $NAME is not RUNNING (run: kaizen desktop start ...)"
S3_URI="$(echo "$INFO" | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["s3SyncUri"])')"
REGION="$(echo "$INFO" | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["region"])')"
case "$S3_URI" in */) ;; *) S3_URI="$S3_URI/" ;; esac

echo "==> [1/4] Syncing $repo_root_dir/ -> desktop \$HOME/$remote_repo_subdir/"
s5cmd_excludes=()
for e in "${sync_excludes[@]}"; do
    s5cmd_excludes+=(--exclude "*/$e/*")
done
AWS_PROFILE=cluster-role AWS_REGION="$REGION" s5cmd sync "${s5cmd_excludes[@]}" \
    "$repo_root_dir/" "$S3_URI$remote_repo_subdir/"

if [[ "$NO_SETUP" -eq 0 ]]; then
    echo "==> [2/4] Setting up env on desktop (idempotent)"
    AWS_PROFILE=kaizen-access kaizen desktop connect --name "$NAME" --cmd "$(remote_setup_cmd)"
else
    echo "==> [2/4] Skipping env setup (--no-setup)"
fi

echo "==> [3/4] Executing on desktop"
AWS_PROFILE=kaizen-access kaizen desktop connect --name "$NAME" --cmd "$(remote_run_cmd)"

echo "==> [4/4] Downloading artifacts -> $local_cache_root_dir/"
mkdir -p "$local_cache_root_dir"
# transport_cache_root_dir is $HOME/autotune_cache; under $HOME it maps to the
# S3 sync prefix path 'autotune_cache/'. Reverse-sync with poll/retry because
# the $HOME->S3 export lags up to ~60s.
remote_rel="autotune_cache/"
got_results=0
for attempt in $(seq 1 12); do
    AWS_PROFILE=cluster-role AWS_REGION="$REGION" s5cmd sync \
        "$S3_URI$remote_rel" "$local_cache_root_dir/" || true
    if [[ -f "$local_cache_root_dir/results.json" ]]; then
        echo "==> results.json present after $attempt attempt(s)"
        got_results=1
        break
    fi
    echo "    waiting for reverse S3 export (attempt $attempt/12)..."
    sleep 10
done
[[ "$got_results" -eq 1 ]] || die "reverse S3 export did not surface results.json after 12 attempts (~120s); artifacts may be incomplete in $local_cache_root_dir/"
echo "==> Done. Artifacts in $local_cache_root_dir/"
