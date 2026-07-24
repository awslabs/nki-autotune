#!/usr/bin/env bash
#
# Fresh-box setup for nki-autotune: build the Python venv (Neuron SDK stack)
# and install the project requirements.
#
# KernelIR.dump() shells out to black to format emitted kernel.py files, so
# black is installed alongside the test and formatting tools in the venv.
#
# It also installs the aws-neuron/nkipy monorepo (nkipy + spike) in editable
# mode from a local clone. Both are absent from PyPI and the autotune runner
# hard-imports them (nkipy.runtime.BaremetalExecutor / spike). `spike` is a
# native nanobind/CMake extension, so its build needs a C++ toolchain.
#
# Every step is idempotent — safe to re-run against an existing venv.
#
# Usage:
#   ./install.sh                          # venv -> ~/venvs/kernel-env, python3
#   VENV=/path/to/venv ./install.sh
#   PYTHON=python3.12 ./install.sh
#   NKIPY_SRC=/path/to/nkipy ./install.sh # clone location for nkipy + spike
#
set -euo pipefail

VENV="${VENV:-$HOME/venvs/kernel-env}"
PYTHON="${PYTHON:-}"
NKIPY_SRC="${NKIPY_SRC:-$HOME/src/nkipy}"
NKIPY_REPO="https://github.com/aws-neuron/nkipy.git"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

# apt needs root; use sudo only when we are not already root.
if [[ "$(id -u)" -eq 0 ]]; then
    SUDO=""
else
    SUDO="sudo"
fi

# --- [1/5] Create or reuse the venv ----------------------------------------
if [[ -f "$VENV/bin/activate" ]]; then
    echo "==> [1/5] Reusing existing venv at $VENV"
else
    echo "==> [1/5] Creating venv at $VENV"
    if [[ -z "$PYTHON" ]]; then
        if command -v python3.12 >/dev/null 2>&1; then
            PYTHON=python3.12
        else
            PYTHON=python3
        fi
    fi
    command -v "$PYTHON" >/dev/null 2>&1 || die "interpreter '$PYTHON' not found (set PYTHON=...)"
    # Ubuntu splits the venv module into a separate apt package; install it on demand.
    if ! "$PYTHON" -c "import venv, ensurepip" 2>/dev/null; then
        echo "==> [1/5] '$PYTHON' lacks venv/ensurepip — installing python3-venv via apt"
        $SUDO apt-get update -qq
        $SUDO apt-get install -y -q python3-venv python3-pip
    fi
    "$PYTHON" -m venv "$VENV"
fi

# Activation scripts can reference unset vars; relax `nounset` across the source.
set +u
# shellcheck disable=SC1091
source "$VENV/bin/activate"
set -u

# --- [2/5] Upgrade pip + point at the Neuron package repository ------------
echo "==> [2/5] Upgrading pip + configuring the Neuron pip index"
python -m pip install --quiet --upgrade pip
python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

# --- [3/5] Neuron SDK: utilities + compiler + framework --------------------
# Loose specifiers (2.*, 2.9.*) are no-ops when an existing venv already
# satisfies them, so this re-affirms rather than downgrades a pinned env.
echo "==> [3/5] Installing wget, awscli + Neuron compiler/framework"
python -m pip install wget awscli
python -m pip install neuronx-cc==2.* torch-neuronx==2.9.* torchvision nki

# --- [4/5] NKIPy + Spike (editable, from source) ---------------------------
# nkipy + spike live in the aws-neuron/nkipy monorepo as subdirs and are NOT
# published to PyPI, so we clone the repo and pip-install each in editable mode
# (-e), per the project's pip install guide. They depend on the neuronx-cc / nki
# wheels installed in [3/8], and the Neuron pip index is already set in [2/8].
# `spike` is a native nanobind/CMake extension, so its editable build needs a
# C++ toolchain + CMake + Python headers; its CMakeLists gracefully skips the
# libnrt-linked runtime when the Neuron Runtime isn't found, so the install
# still succeeds off-box (the _spike native module is built on a Trn box).
echo "==> [4/5] Installing nkipy + spike (editable) from $NKIPY_SRC"
if ! command -v cmake >/dev/null 2>&1 || ! command -v g++ >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1; then
    echo "==> [4/5] Installing spike build deps (git, cmake, build-essential, python3-dev) via apt"
    $SUDO apt-get update -qq
    $SUDO apt-get install -y -q git cmake build-essential python3-dev
fi
if [[ -d "$NKIPY_SRC/.git" ]]; then
    echo "==> [4/5] Reusing existing nkipy clone at $NKIPY_SRC"
else
    echo "==> [4/5] Cloning $NKIPY_REPO -> $NKIPY_SRC"
    mkdir -p "$(dirname "$NKIPY_SRC")"
    git clone "$NKIPY_REPO" "$NKIPY_SRC"
fi
python -m pip install -e "$NKIPY_SRC/nkipy" -e "$NKIPY_SRC/spike"

# --- [5/5] Python project requirements + formatters ------------------------
# Third-party deps declared across autotune/ + nkigym/ pyproject.toml, plus two
# the code imports but neither manifest declares: networkx (the IR graph
# backbone in nkigym/ir) and ml_dtypes (bf16 dtype resolution in
# runner/types.py). The first-party autotune/nkigym packages are NOT installed
# here. Tests use the root pytest configuration, while example commands set
# PYTHONPATH to the current worktree. An editable nkigym install pins one
# worktree path and breaks subprocess tests in another worktree. black + isort
# drive dump()'s kernel.py formatting.
echo "==> [5/5] Installing Python project requirements + formatters"
python -m pip install \
    numpy networkx ml_dtypes \
    matplotlib tabulate tqdm \
    hypothesis pytest pre-commit \
    black isort

# --- Smoke test -------------------------------------------------------------
echo "==> Verifying toolchain"
python -c "import numpy, networkx, ml_dtypes, matplotlib, tabulate, tqdm, hypothesis, pytest, nki, nkipy" ||
    die "venv broken — a core requirement (numpy/networkx/ml_dtypes/.../nki/nkipy) is not importable"
# `spike` is a native module that builds only when the Neuron Runtime is present;
# a failed import off-box is expected, so warn rather than die.
if python -c "import spike" 2>/dev/null; then
    echo "    spike OK (native _spike runtime present)"
else
    echo "    WARN: 'import spike' failed — native _spike module not built (expected off a Neuron box; needs NRT at /opt/aws/neuron)"
fi
black --version >/dev/null || die "black not callable"

echo "==> Done. venv=$VENV | python=$(python --version 2>&1) | black=$(black --version | head -1)"
echo "==> Verify end-to-end: PYTHONPATH=.:nkigym/src:autotune/src python examples/random_rollout.py --cache /home/ubuntu/cache"
