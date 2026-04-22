#!/usr/bin/env bash
#
# Install NKI kernel development Python venv on remote gym hosts using the
# public Neuron SDK per:
#   https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/get-started/setup-env.html
# and nkipy/spike from source per:
#   https://github.com/aws-neuron/nkipy
#
# Usage:
#   ./install_neuron.sh host1 [host2 ...]   # install on listed SSH hosts
#   ./install_neuron.sh --local             # install on this machine
#
set -euo pipefail

VENV_DIR="/home/ubuntu/venvs/kernel-env"
PYTHON="python3.12"
NEURON_PIP_INDEX="https://pip.repos.neuron.amazonaws.com"
NKIPY_REPO="https://github.com/aws-neuron/nkipy.git"
NKIPY_SRC_DIR="/tmp/nkipy"

install_local() {
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
    python -m pip install --upgrade pip

    echo "==> Configuring Neuron pip repo..."
    pip config set global.extra-index-url "$NEURON_PIP_INDEX"

    echo "==> Installing NKI and neuronx-cc..."
    pip install neuronx-cc==2.* torch-neuronx==2.9.* torchvision nki

    echo "==> Cloning nkipy at $NKIPY_SRC_DIR..."
    mkdir -p "$(dirname "$NKIPY_SRC_DIR")"
    if [[ -d "$NKIPY_SRC_DIR/.git" ]]; then
        git -C "$NKIPY_SRC_DIR" pull --ff-only
    else
        git clone "$NKIPY_REPO" "$NKIPY_SRC_DIR"
    fi

    echo "==> Installing nkipy and spike from source..."
    pip install "$NKIPY_SRC_DIR/nkipy" "$NKIPY_SRC_DIR/spike"

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
    echo "==> [$host] Running install..."
    ssh "$host" "bash -s -- --local" < "$0"
    echo "==> [$host] Done."
}

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 host1 [host2 ...] | --local" >&2
    exit 1
fi

if [[ "${1}" == "--local" ]]; then
    install_local
    exit 0
fi

for host in "$@"; do
    install_remote "$host" &
done

wait
echo "==> All hosts done."
