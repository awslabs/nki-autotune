#!/usr/bin/env bash
#
# Fresh-box setup for nki-autotune: build the Python venv (Neuron SDK stack)
# and install the host tools that KernelIR.dump() shells out to.
#
# dump() calls two host binaries that are NOT Python deps and are absent on a
# fresh box (each crashes the examples with FileNotFoundError / RuntimeError):
#   - black : formats the emitted kernel.py
#   - mmdc  : renders tree.png / dependency.png
#             (Mermaid CLI -> Node.js -> headless Chrome via Puppeteer)
#
# Every step is idempotent — safe to re-run against an existing venv.
#
# Usage:
#   ./install.sh                          # venv -> ~/venvs/kernel-env, python3
#   VENV=/path/to/venv ./install.sh
#   PYTHON=python3.12 ./install.sh
#
set -euo pipefail

VENV="${VENV:-$HOME/venvs/kernel-env}"
PYTHON="${PYTHON:-}"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

# apt / npm -g need root; use sudo only when we are not already root.
if [[ "$(id -u)" -eq 0 ]]; then
    SUDO=""
else
    SUDO="sudo"
fi

# --- [1/7] Create or reuse the venv ----------------------------------------
if [[ -f "$VENV/bin/activate" ]]; then
    echo "==> [1/7] Reusing existing venv at $VENV"
else
    echo "==> [1/7] Creating venv at $VENV"
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
        echo "==> [1/7] '$PYTHON' lacks venv/ensurepip — installing python3-venv via apt"
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

# --- [2/7] Upgrade pip + point at the Neuron package repository ------------
echo "==> [2/7] Upgrading pip + configuring the Neuron pip index"
python -m pip install --quiet --upgrade pip
python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

# --- [3/7] Neuron SDK: utilities + compiler + framework --------------------
# Loose specifiers (2.*, 2.9.*) are no-ops when an existing venv already
# satisfies them, so this re-affirms rather than downgrades a pinned env.
echo "==> [3/7] Installing wget, awscli + Neuron compiler/framework"
python -m pip install wget awscli
python -m pip install neuronx-cc==2.* torch-neuronx==2.9.* torchvision nki

# --- [4/7] Python formatters used by dump() --------------------------------
echo "==> [4/7] Installing black + isort"
python -m pip install --quiet black isort

# --- [5/7] Node.js + npm (apt) ---------------------------------------------
# Ubuntu ships Node 18; mermaid-cli 11.x prints a benign EBADENGINE warning
# (wants Node >=20) but renders correctly on 18.
if command -v node >/dev/null 2>&1; then
    echo "==> [5/7] Node.js present ($(node --version)) — skipping apt install"
else
    echo "==> [5/7] Installing Node.js + npm via apt"
    $SUDO apt-get update -qq
    $SUDO apt-get install -y -q nodejs npm
fi

# --- [6/7] Mermaid CLI (mmdc) ----------------------------------------------
if command -v mmdc >/dev/null 2>&1; then
    echo "==> [6/7] mmdc present ($(mmdc --version)) — skipping npm install"
else
    echo "==> [6/7] Installing @mermaid-js/mermaid-cli globally (provides mmdc)"
    $SUDO npm install -g @mermaid-js/mermaid-cli
fi

# --- [7/7] Headless Chrome for Puppeteer -----------------------------------
# mmdc drives headless Chrome via Puppeteer. The browser must live in the
# INVOKING user's cache (~/.cache/puppeteer) — running this under sudo puts it
# in /root/.cache where the examples (run as you) can't find it. So: NO sudo.
# The Chrome version is chosen by the puppeteer bundled with mermaid-cli, so we
# never pin it here.
echo "==> [7/7] Installing chrome-headless-shell for Puppeteer (user cache)"
MERMAID_DIR="$(npm root -g)/@mermaid-js/mermaid-cli"
[[ -d "$MERMAID_DIR" ]] || die "mermaid-cli not found at $MERMAID_DIR"
(cd "$MERMAID_DIR" && npx --yes puppeteer browsers install chrome-headless-shell)

# --- Smoke test: exercise the full black + mmdc render chain ----------------
echo "==> Verifying toolchain"
python -c "import numpy, nki" || die "venv broken — numpy/nki not importable"
black --version >/dev/null || die "black not callable"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
printf 'flowchart TB\n    a --> b\n' >"$TMP/t.mmd"
printf '{"args":["--no-sandbox"]}' >"$TMP/p.json"
mmdc -i "$TMP/t.mmd" -o "$TMP/t.png" --puppeteerConfigFile "$TMP/p.json" >/dev/null 2>&1 ||
    die "mmdc render failed — Node/Puppeteer/Chrome not working"
[[ -s "$TMP/t.png" ]] || die "mmdc produced an empty PNG"

echo "==> Done. venv=$VENV | python=$(python --version 2>&1) | node=$(node --version) | mmdc=$(mmdc --version)"
echo "==> Verify end-to-end: python examples/matmul_lhsT_rhs.py --cache-root-dir /home/ubuntu/cache"
