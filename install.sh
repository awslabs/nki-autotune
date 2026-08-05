#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "Usage: $0 --host HOST" >&2
    exit 2
}

[[ $# -eq 2 && "$1" == "--host" ]] || usage
HOST="$2"
[[ -n "$HOST" && "$HOST" != -* && "$HOST" != *[[:space:]]* ]] || usage

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HOME/venvs/kernel-env"
REMOTE_ROOT=".cache/nkigym-profile/install"
PIP_INDEX="https://pip.repos.neuron.amazonaws.com"
SSH_OPTIONS=(-o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=no)

command -v python3 >/dev/null 2>&1 || { echo "python3 is not on PATH" >&2; exit 1; }
command -v ssh >/dev/null 2>&1 || { echo "ssh is not on PATH" >&2; exit 1; }
command -v rsync >/dev/null 2>&1 || { echo "rsync is not on PATH" >&2; exit 1; }

if [[ ! -x "$VENV/bin/python" ]]; then
    echo "==> Creating local environment at $VENV"
    python3 -m venv "$VENV"
fi

echo "==> Installing local environment"
"$VENV/bin/python" -m pip install --only-binary=:all: --extra-index-url "$PIP_INDEX" -e "$ROOT/nkigym" black pytest

echo "==> Uploading nkigym to $HOST"
ssh "${SSH_OPTIONS[@]}" "$HOST" "mkdir -p \"\$HOME\"/$REMOTE_ROOT"
rsync -az --delete -e "ssh ${SSH_OPTIONS[*]}" \
    --exclude __pycache__ --exclude '*.pyc' \
    "$ROOT/nkigym/" "$HOST:$REMOTE_ROOT/nkigym/"

echo "==> Installing remote environment"
ssh "${SSH_OPTIONS[@]}" "$HOST" "bash -s -- '$REMOTE_ROOT/nkigym' '$PIP_INDEX'" <<'REMOTE'
set -euo pipefail

SOURCE="$HOME/$1"
VENV="$HOME/venvs/kernel-env"
PIP_INDEX="$2"

if [[ ! -x "$VENV/bin/python" ]]; then
    echo "==> Creating remote environment at $VENV"
    python3 -m venv "$VENV"
fi

"$VENV/bin/python" -m pip install --only-binary=:all: --extra-index-url "$PIP_INDEX" "$SOURCE"
REMOTE

echo "==> Installed local and remote environments"
