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

# apt / npm -g need root; use sudo only when we are not already root.
if [[ "$(id -u)" -eq 0 ]]; then
    SUDO=""
else
    SUDO="sudo"
fi

# --- [1/8] Create or reuse the venv ----------------------------------------
if [[ -f "$VENV/bin/activate" ]]; then
    echo "==> [1/8] Reusing existing venv at $VENV"
else
    echo "==> [1/8] Creating venv at $VENV"
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
        echo "==> [1/8] '$PYTHON' lacks venv/ensurepip — installing python3-venv via apt"
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

# --- [2/8] Upgrade pip + point at the Neuron package repository ------------
echo "==> [2/8] Upgrading pip + configuring the Neuron pip index"
python -m pip install --quiet --upgrade pip
python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

# --- [3/8] Neuron SDK: utilities + compiler + framework --------------------
# Loose specifiers (2.*, 2.9.*) are no-ops when an existing venv already
# satisfies them, so this re-affirms rather than downgrades a pinned env.
echo "==> [3/8] Installing wget, awscli + Neuron compiler/framework"
python -m pip install wget awscli
python -m pip install neuronx-cc==2.* torch-neuronx==2.9.* torchvision nki

# --- [4/8] NKIPy + Spike (editable, from source) ---------------------------
# nkipy + spike live in the aws-neuron/nkipy monorepo as subdirs and are NOT
# published to PyPI, so we clone the repo and pip-install each in editable mode
# (-e), per the project's pip install guide. They depend on the neuronx-cc / nki
# wheels installed in [3/8], and the Neuron pip index is already set in [2/8].
# `spike` is a native nanobind/CMake extension, so its editable build needs a
# C++ toolchain + CMake + Python headers; its CMakeLists gracefully skips the
# libnrt-linked runtime when the Neuron Runtime isn't found, so the install
# still succeeds off-box (the _spike native module is built on a Trn box).
echo "==> [4/8] Installing nkipy + spike (editable) from $NKIPY_SRC"
if ! command -v cmake >/dev/null 2>&1 || ! command -v g++ >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1; then
    echo "==> [4/8] Installing spike build deps (git, cmake, build-essential, python3-dev) via apt"
    $SUDO apt-get update -qq
    $SUDO apt-get install -y -q git cmake build-essential python3-dev
fi
if [[ -d "$NKIPY_SRC/.git" ]]; then
    echo "==> [4/8] Reusing existing nkipy clone at $NKIPY_SRC"
else
    echo "==> [4/8] Cloning $NKIPY_REPO -> $NKIPY_SRC"
    mkdir -p "$(dirname "$NKIPY_SRC")"
    git clone "$NKIPY_REPO" "$NKIPY_SRC"
fi
python -m pip install -e "$NKIPY_SRC/nkipy" -e "$NKIPY_SRC/spike"

# --- [5/8] Python project requirements + formatters ------------------------
# Third-party deps declared across autotune/ + nkigym/ pyproject.toml, plus two
# the code imports but neither manifest declares: networkx (the IR graph
# backbone in nkigym/ir) and ml_dtypes (bf16 dtype resolution in
# runner/types.py). The first-party autotune/nkigym packages are NOT installed
# here — the examples and tests put nkigym/src + the repo root on sys.path
# themselves, and an editable install of nkigym pins a worktree path that
# breaks subprocess tests. black + isort drive dump()'s kernel.py formatting.
echo "==> [5/8] Installing Python project requirements + formatters"
python -m pip install \
    numpy networkx ml_dtypes \
    matplotlib tabulate tqdm \
    hypothesis pytest pre-commit \
    black isort

# --- [6/8] Node.js + npm (apt) ---------------------------------------------
# Ubuntu ships Node 18; mermaid-cli 11.x prints a benign EBADENGINE warning
# (wants Node >=20) but renders correctly on 18.
if command -v node >/dev/null 2>&1; then
    echo "==> [6/8] Node.js present ($(node --version)) — skipping apt install"
else
    echo "==> [6/8] Installing Node.js + npm via apt"
    $SUDO apt-get update -qq
    $SUDO apt-get install -y -q nodejs npm
fi

# --- [7/8] Mermaid CLI (mmdc) ----------------------------------------------
# npm's cacache hardlinks from its cache dir into node_modules. On a Kaizen
# desktop $HOME is an S3-Files NFS mount that rejects hardlinks ("EMLINK: too
# many links"), so a default ~/.npm cache breaks `npm install`. Probe $HOME for
# hardlink support and, when it's missing, relocate the npm cache to local disk
# (/ustore/ssd on Kaizen, else a tmpdir). No-op on normal filesystems.
if command -v mmdc >/dev/null 2>&1; then
    echo "==> [7/8] mmdc present ($(mmdc --version)) — skipping npm install"
else
    npm_cache_args=()
    _lt="$HOME/.nki_hardlink_test"
    if touch "$_lt" 2>/dev/null && ln "$_lt" "$_lt.2" 2>/dev/null; then
        echo "==> [7/8] \$HOME supports hardlinks — using default npm cache"
    else
        if [[ -d /ustore/ssd && -w /ustore/ssd ]]; then
            npm_cache_dir=/ustore/ssd/.npm-cache
        else
            npm_cache_dir="$(mktemp -d)/npm-cache"
        fi
        echo "==> [7/8] \$HOME rejects hardlinks (S3-Files) — npm cache -> $npm_cache_dir"
        npm_cache_args=(--cache "$npm_cache_dir")
    fi
    rm -f "$_lt" "$_lt.2"
    echo "==> [7/8] Installing @mermaid-js/mermaid-cli globally (provides mmdc)"
    $SUDO npm install -g ${npm_cache_args[@]+"${npm_cache_args[@]}"} @mermaid-js/mermaid-cli
fi

# --- [8/8] Headless Chrome for Puppeteer -----------------------------------
# mmdc drives headless Chrome via Puppeteer. The browser must live in the
# INVOKING user's cache (~/.cache/puppeteer) — running this under sudo puts it
# in /root/.cache where the examples (run as you) can't find it. So: NO sudo.
# The Chrome version is chosen by the puppeteer bundled with mermaid-cli, so we
# never pin it here.
echo "==> [8/8] Installing chrome-headless-shell for Puppeteer (user cache)"
MERMAID_DIR="$(npm root -g)/@mermaid-js/mermaid-cli"
[[ -d "$MERMAID_DIR" ]] || die "mermaid-cli not found at $MERMAID_DIR"
(cd "$MERMAID_DIR" && npx --yes puppeteer browsers install chrome-headless-shell)

# --- Smoke test: exercise the full black + mmdc render chain ----------------
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
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
printf 'flowchart TB\n    a --> b\n' >"$TMP/t.mmd"
printf '{"args":["--no-sandbox"]}' >"$TMP/p.json"
mmdc -i "$TMP/t.mmd" -o "$TMP/t.png" --puppeteerConfigFile "$TMP/p.json" >/dev/null 2>&1 ||
    die "mmdc render failed — Node/Puppeteer/Chrome not working"
[[ -s "$TMP/t.png" ]] || die "mmdc produced an empty PNG"

echo "==> Done. venv=$VENV | python=$(python --version 2>&1) | node=$(node --version) | mmdc=$(mmdc --version)"
echo "==> Verify end-to-end: python examples/matmul_lhsT_rhs.py --cache-root-dir /home/ubuntu/cache"
