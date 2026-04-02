#!/usr/bin/env bash
#
# Install NKI kernel development Python venv on remote gym hosts.
#
# Usage:
#   bash install_neuron.sh           # install on all hosts
#   bash install_neuron.sh gym-1     # install on specific host
#   bash install_neuron.sh --local   # install on this machine
#
set -euo pipefail

HOSTS=(gym-1 gym-2 gym-3 gym-4 gym-5 gym-6 gym-7)
WHEEL_DIR="/home/ubuntu/shared_workplace/artifacts"
REMOTE_WHEEL_DIR="/tmp/nki-wheels"
VENV_DIR="/home/ubuntu/venvs/kernel-env"
PYTHON="python3.12"

WHEEL_FILES=(
    nki-0.4.0b4-cp312-cp312-linux_x86_64.whl
    neuronx_cc-2.0.245138.0a0+b9c30905-cp312-cp312-linux_x86_64.whl
    nkipy-0.1.0-py3-none-any.whl
    spike-0.1.0-cp312-cp312-linux_x86_64.whl
)

install_local() {
    local wheel_dir="${1:-$WHEEL_DIR}"

    echo "==> Installing Neuron drivers and runtime..."

    . /etc/os-release
    sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main
EOF
    wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB \
        | sudo apt-key add -

    sudo apt-get update -y
    sudo apt-get install -y linux-headers-"$(uname -r)"
    sudo apt-get install -y git
    sudo apt-get install -y aws-neuronx-dkms=2.*
    sudo apt-get install -y aws-neuronx-collectives=2.*
    sudo apt-get install -y aws-neuronx-runtime-lib=2.*
    sudo apt-get install -y aws-neuronx-tools=2.*

    if ! command -v $PYTHON &>/dev/null; then
        echo "==> Installing $PYTHON via deadsnakes PPA..."
        sudo apt-get install -y software-properties-common
        sudo add-apt-repository -y ppa:deadsnakes/ppa
        sudo apt-get update -y
        sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
    fi

    export PATH=/opt/aws/neuron/bin:$PATH
    echo "==> Neuron drivers and runtime installed."

    echo "==> Creating Python venv at $VENV_DIR..."
    mkdir -p "$(dirname "$VENV_DIR")"
    $PYTHON -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip

    echo "==> Installing wheels from $wheel_dir..."
    local wheels=()
    for f in "${WHEEL_FILES[@]}"; do
        wheels+=("$wheel_dir/$f")
    done
    pip install "${wheels[@]}"

    echo "==> Verifying installation..."
    python -c "
import nki; print(f'  nki {nki.__version__}')
import neuronxcc; print('  neuronxcc OK')
import nkipy; print('  nkipy OK')
import spike; print('  spike OK')
"
    echo "==> Done."
}

install_remote() {
    local host="$1"
    echo "==> [$host] Copying wheels..."
    ssh "$host" "mkdir -p $REMOTE_WHEEL_DIR"
    scp -q "${WHEEL_FILES[@]/#/$WHEEL_DIR/}" "$host:$REMOTE_WHEEL_DIR/"

    echo "==> [$host] Running install..."
    ssh "$host" "bash -s -- --local $REMOTE_WHEEL_DIR" < "$0"

    echo "==> [$host] Cleaning up wheels..."
    ssh "$host" "rm -rf $REMOTE_WHEEL_DIR"
    echo "==> [$host] Done."
}

if [[ "${1:-}" == "--local" ]]; then
    install_local "${2:-$WHEEL_DIR}"
    exit 0
fi

if [[ $# -gt 0 ]]; then
    HOSTS=("$@")
fi

for host in "${HOSTS[@]}"; do
    install_remote "$host" &
done

wait
echo "==> All hosts done."
