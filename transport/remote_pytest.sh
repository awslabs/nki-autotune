#!/usr/bin/env bash
#
# remote_pytest.sh — sync the repo to the Kaizen desktop and run pytest there.
#
# This dev box has NO Python env; the kernel-env venv (numpy/networkx/nki/
# pytest/...) lives on the Kaizen desktop. Unit tests can't run locally, so
# this syncs the working tree to the desktop's S3-backed $HOME and runs pytest
# in the venv, returning the exit code. Mirrors transport/kaizen.sh's sync +
# `connect --cmd` primitives, but for pytest instead of an example script.
#
# Usage:
#   transport/remote_pytest.sh <pytest-args...>
#   transport/remote_pytest.sh test/transforms/test_software_pipeline.py -q
#   transport/remote_pytest.sh test/ir/ test/codegen/ -q
#
# Env:
#   NAME   desktop name (default: default)
#
# Prerequisites (same as kaizen.sh): mwinit -o; AWS_PROFILE kaizen-access +
# cluster-role configured; s5cmd on PATH; kernel-env venv built on the desktop.
set -euo pipefail

NAME="${NAME:-default}"
REPO_LOCAL="/workplace/weittang/nki-autotune"

if [[ $# -eq 0 ]]; then
    echo "usage: $0 <pytest-args...>" >&2
    exit 2
fi

echo "==> Resolving desktop $NAME"
INFO="$(AWS_PROFILE=kaizen-access kaizen desktop info --name "$NAME" --output json 2>&1 | tail -1)"
S3URI="$(echo "$INFO" | python3 -c "import sys,json; print(json.loads(sys.stdin.read().split('JSON_OUTPUT:')[-1])['s3SyncUri'])")"
REGION="$(echo "$INFO" | python3 -c "import sys,json; print(json.loads(sys.stdin.read().split('JSON_OUTPUT:')[-1])['region'])")"

echo "==> Syncing repo -> desktop \$HOME/nki-autotune/"
AWS_PROFILE=cluster-role AWS_REGION="$REGION" s5cmd sync \
    --exclude "*.pyc" --exclude "__pycache__/*" \
    "${REPO_LOCAL}/" "${S3URI}nki-autotune/" >/dev/null 2>&1

echo "==> Running pytest on $NAME: $*"
REMOTE_CMD="source \$HOME/venvs/kernel-env/bin/activate && cd \$HOME/nki-autotune && PYTHONPATH=.:nkigym/src:autotune/src python -m pytest $* 2>&1"
AWS_PROFILE=kaizen-access kaizen desktop connect --name "$NAME" --cmd "$REMOTE_CMD"
