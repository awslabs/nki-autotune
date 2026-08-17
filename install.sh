#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "Usage: $0 --host HOST..." >&2
    exit 2
}

valid_host() {
    local host="$1"
    [[ -n "$host" && "$host" != -* && "$host" != *[[:space:]]* ]]
}

[[ $# -ge 2 && "$1" == "--host" ]] || usage
shift
REMOTE_HOSTS=("$@")
for remote_host in "${REMOTE_HOSTS[@]}"; do
    valid_host "$remote_host" || usage
done

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
"$VENV/bin/python" -m pip install --only-binary=:all: --extra-index-url "$PIP_INDEX" \
    -e "$ROOT/nkigym" black isort pytest pytest-timeout

install_remote_environment() {
    local remote_host="$1"

    echo "==> Uploading nkigym to $remote_host"
    ssh "${SSH_OPTIONS[@]}" "$remote_host" "mkdir -p \"\$HOME\"/$REMOTE_ROOT"
    rsync -az --delete -e "ssh ${SSH_OPTIONS[*]}" \
        --exclude __pycache__ --exclude '*.pyc' \
        "$ROOT/nkigym/" "$remote_host:$REMOTE_ROOT/nkigym/"

    echo "==> Installing remote environment on $remote_host"
    ssh "${SSH_OPTIONS[@]}" "$remote_host" "bash -s -- '$REMOTE_ROOT/nkigym' '$PIP_INDEX'" <<'REMOTE'
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
}

for remote_host in "${REMOTE_HOSTS[@]}"; do
    install_remote_environment "$remote_host"
done

echo "==> Installed local environment and remote environments on ${REMOTE_HOSTS[*]}"
