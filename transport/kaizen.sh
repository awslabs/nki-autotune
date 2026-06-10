#!/usr/bin/env bash
#
# Kaizen transport: sync repo -> verify script -> run --cmd -> reverse-sync cache.
#
# The 4 steps:
#   [1/4] s5cmd-sync the whole repo into the desktop's S3-backed $HOME. This
#         MUST include the nkigym/ and autotune/ source trees: scripts import
#         nkigym.* / autotune.* and resolve them from this synced source via
#         sys.path (they self-bootstrap repo_root + */src).
#   [2/4] Preflight: resolve the desktop's $HOME so we know where --cache lands
#         under the S3 export (activating the venv first also confirms it's set
#         up). The --cmd script's acceptance of --cache is checked LOCALLY,
#         before [1/4] (see below) — not on the desktop.
#   [3/4] Run the --cmd in the venv. If the script accepts --cache, append
#         `--cache <dir>` and (on success) write a completion manifest into
#         --cache as the run's LAST action; otherwise run it as-is.
#   [4/4] If --cache was appended, reverse-sync --cache to the SAME absolute path
#         locally, polling until every file the manifest lists has landed at its
#         stated size — a reliable done-signal across the ~60s export lag and S3
#         orphans. If not, there are no remote artifacts, so this is skipped.
#
# Before [1/4] the transport (a) decides LOCALLY whether the --cmd script accepts
# --cache (this dev box has no Python env to run `python <script> --help`, so it
# statically scans the source) — a BRANCH, not a gate: accepts → append --cache +
# reverse-sync; doesn't → run as-is, skip the sync — and (b) tees every line of
# its own stdout+stderr into <--cache>/output.log. So the local --cache ALWAYS
# gets the run's terminal output, in either branch, and even when an infra
# failure (expired creds, desktop down) aborts the run early.
#
# The venv is NOT built here. Set it up once yourself, on the desktop:
#   kaizen desktop connect --name <d> --cmd 'VENV=$HOME/venvs/kernel-env bash ~/nki-autotune/install.sh'
#
# Usage:
#   transport/kaizen.sh --name <desktop> --cmd "python path/to/xxx.py" --cache /abs/path
#
# Prerequisites (caller's responsibility — fails loud if missing):
#   - mwinit -o done; kaizen-access + cluster-role AWS profiles configured
#   - s5cmd on PATH; kaizen CLI installed
#   - the kernel-env venv already built on the desktop (see install.sh)
#   - --cache is an absolute path under the desktop's $HOME (S3-export visible)
#
# If --name's desktop is not RUNNING (e.g. it hit its --timeout and went
# FAILED), this starts a fresh one (see desktop_* in common.sh) and waits for
# RUNNING. $HOME is S3-backed and survives, so the kernel-env venv persists.
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

# Wipe the local --cache up front so it holds ONLY this run's artifacts, even if
# the run fails partway (a stale local cache from a prior run would otherwise
# masquerade as this run's output). The remote --cache is wiped symmetrically by
# the run command itself (see remote_run_cmd); any S3 orphans still lingering
# from a prior run's lagging deletes are dropped by the manifest prune in [4/4].
rm -rf "$CACHE_DIR"
mkdir -p "$CACHE_DIR"

# Tee every line of our own stdout+stderr from here on into <cache>/output.log
# while still printing to the terminal — so the cache carries a full record of
# the run, INCLUDING an infra failure (expired creds, desktop down) that aborts
# before the remote run ever happens. The process substitution runs tee
# concurrently; the log starts fresh since we just recreated $CACHE_DIR. It is
# written locally (not by the remote run) so it never enters transport_manifest;
# the reverse-sync completeness check ignores it and the orphan prune in [4/4]
# skips it explicitly.
exec > >(tee "$CACHE_DIR/$transport_log") 2>&1
echo "==> Wiped + recreated local cache $CACHE_DIR (run log: $CACHE_DIR/$transport_log)"

# Decide LOCALLY whether the --cmd script accepts --cache, before the round trip.
# The dev box has no Python env to run `python <script> --help` (that's why the
# old remote check moved here), so this is a static scan of the script source
# for a --cache reference — the faithful local equivalent. This is a BRANCH, not
# a gate: if the script accepts --cache we append it in [3/4] and reverse-sync
# its artifacts in [4/4]; if not, we run the command as-is and skip the reverse
# sync. EITHER WAY the run's terminal output is captured into $CACHE_DIR's
# output.log (tee'd above), so the local --cache always gets the output.
local_script="$(user_script)"
case "$local_script" in
    /*) ;;
    *) local_script="$repo_root_dir/$local_script" ;;
esac
[[ -f "$local_script" ]] || die "script not found locally: $local_script"
# Match --cache only as a QUOTED option token — "--cache" or '--cache', the form
# an arg parser registers it in (argparse add_argument / click.option). The
# closing quote right after rejects bare prose in a docstring/comment ("...NO
# --cache arg...") and unrelated longer options ("--cache-dir"), both of which a
# plain substring grep would false-match. This is the robust local stand-in for
# the desktop's `python <script> --help | grep --cache` (no local Python env).
SCRIPT_ACCEPTS_CACHE=0
grep -Eq "['\"]--cache['\"]" "$local_script" && SCRIPT_ACCEPTS_CACHE=1
if [[ "$SCRIPT_ACCEPTS_CACHE" == "1" ]]; then
    echo "==> $(user_script) accepts --cache (local scan of $local_script): will append --cache + reverse-sync artifacts"
else
    echo "==> $(user_script) does not reference --cache (local scan of $local_script): running as-is; only output.log returns"
fi

# Read a desktop's status (RUNNING / FAILED / CANCELLED / PENDING / "" if absent).
desktop_status() {
    AWS_PROFILE=kaizen-access kaizen desktop info --name "$NAME" --output json 2>&1 | tail -1 |
        python3 -c 'import json,sys
try: print(json.loads(sys.stdin.read()).get("desktopStatus",""))
except Exception: print("")'
}

echo "==> Resolving desktop $NAME"
STATUS="$(desktop_status)"
echo "    status: ${STATUS:-<none>}"
case "$STATUS" in
    RUNNING) ;;
    PENDING)
        # Already coming up — just wait below, don't start a second one.
        echo "==> Desktop is PENDING — waiting for RUNNING" ;;
    *)
        # FAILED / CANCELLED / absent — start a fresh one. No explicit stop:
        # per the Kaizen docs, `start` rejects only a same-name desktop that is
        # ALREADY RUNNING (or the 2-desktop cap); a terminal FAILED/CANCELLED
        # entry is auto-pruned, so `start` replaces it directly.
        echo "==> Desktop not usable (${STATUS:-<none>}) — starting $desktop_instance ($desktop_timeout s)"
        AWS_PROFILE=kaizen-access kaizen desktop start --name "$NAME" \
            --image "$desktop_image" --instanceType "$desktop_instance" --timeout "$desktop_timeout" 2>&1 | tail -3 ;;
esac
if [[ "$STATUS" != "RUNNING" ]]; then
    echo "==> Waiting for RUNNING (up to ~5min)"
    for attempt in $(seq 1 30); do
        STATUS="$(desktop_status)"
        echo "    [$attempt/30] status: ${STATUS:-<none>}"
        [[ "$STATUS" == "RUNNING" ]] && break
        case "$STATUS" in FAILED|CANCELLED) die "desktop $NAME entered terminal state $STATUS while starting" ;; esac
        sleep 10
    done
    [[ "$STATUS" == "RUNNING" ]] || die "desktop $NAME did not reach RUNNING (~5min); last status: ${STATUS:-<none>}"
fi

echo "==> Resolving s3SyncUri + region"
INFO="$(AWS_PROFILE=kaizen-access kaizen desktop info --name "$NAME" --output json 2>&1 | tail -1)"
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

echo "==> [2/4] Preflight: resolve remote \$HOME (and confirm the venv is set up)"
CHECK_OUT="$(AWS_PROFILE=kaizen-access kaizen desktop connect --name "$NAME" --cmd "$(remote_check_cmd)" 2>&1)" || {
    echo "$CHECK_OUT" | tail -5
    die "preflight failed — the venv is likely missing (run install.sh on the desktop)"
}
REMOTE_HOME="$(printf '%s\n' "$CHECK_OUT" | sed -n 's/^TRANSPORT_HOME=//p' | tail -1)"
[[ -n "$REMOTE_HOME" ]] || die "could not resolve remote \$HOME from preflight"
# The cache-under-$HOME requirement only matters when we reverse-sync remote
# --cache artifacts, i.e. when the script accepts --cache. (output.log is written
# locally, so it never needs the desktop's --cache to be S3-visible.)
cache_rel="${CACHE_DIR#"$REMOTE_HOME"/}"
if [[ "$SCRIPT_ACCEPTS_CACHE" == "1" ]]; then
    [[ "$cache_rel" != "$CACHE_DIR" ]] ||
        die "--cache ($CACHE_DIR) must be under the desktop's \$HOME ($REMOTE_HOME) so the reverse sync can see it"
fi

if [[ "$SCRIPT_ACCEPTS_CACHE" == "1" ]]; then
    echo "==> [3/4] Executing on desktop: $USER_CMD --cache $CACHE_DIR"
else
    echo "==> [3/4] Executing on desktop: $USER_CMD"
fi
AWS_PROFILE=kaizen-access kaizen desktop connect --name "$NAME" --cmd "$(remote_run_cmd "$SCRIPT_ACCEPTS_CACHE")"

# [4/4] Reverse-sync the desktop --cache artifacts back — ONLY when the script
# accepts --cache (otherwise it wrote nothing remotely, there's no manifest, and
# the only local artifact is output.log, which is already complete). The run's
# terminal output reached output.log in BOTH branches via the tee above, so the
# "download the terminal output back to --cache" guarantee holds either way.
if [[ "$SCRIPT_ACCEPTS_CACHE" != "1" ]]; then
    echo "==> [4/4] No --cache on the script — skipping artifact reverse-sync; output.log holds the run output"
else
    echo "==> [4/4] Reverse-syncing cache -> $CACHE_DIR/"
    # $CACHE_DIR was wiped + recreated before the run (see top), so it holds ONLY
    # this run's files (plus our local output.log); stale S3 orphans the prefix
    # sync re-pulls are dropped by the manifest prune after the loop.
    # The run wrote $transport_manifest LAST (see remote_run_cmd): every output
    # file as 'size<TAB>relpath', sizes from the lag-free desktop FS. We sync the
    # whole --cache prefix each pass, then check the manifest is present locally
    # AND every file it lists exists at its stated size. This is reliable against
    # the three things that broke the old count heuristic: the ~60s export lag
    # (we wait for real completion), S3 orphans from prior runs (only
    # manifest-listed files must match — extras are ignored), and partial trickle
    # (size-checked, not counted). A script that accepts --cache but writes
    # nothing yields an EMPTY manifest (0 files) — a valid "finished, no
    # artifacts" outcome, not a failure. s5cmd S3->local sync needs a trailing
    # '*' on the source prefix.
    manifest_local="$CACHE_DIR/$transport_manifest"
    done_ok=0
    for attempt in $(seq 1 18); do
        AWS_PROFILE=cluster-role AWS_REGION="$REGION" s5cmd sync \
            "$S3_URI$cache_rel/*" "$CACHE_DIR/" >/dev/null 2>&1 || true
        if [[ -f "$manifest_local" ]] &&
            missing="$(while IFS=$'\t' read -r sz rel; do
                [[ -z "$rel" ]] && continue
                [[ "$(stat -c %s "$CACHE_DIR/$rel" 2>/dev/null)" == "$sz" ]] || echo "$rel"
            done < "$manifest_local")" && [[ -z "$missing" ]]; then
            # `grep -c` exits 1 on an empty manifest (0 matches) — guard it under
            # set -e so a valid no-artifacts run doesn't abort the transport here.
            n="$(grep -c . "$manifest_local" || true)"
            echo "==> manifest complete: $n file(s) verified after $attempt attempt(s)"
            done_ok=1
            break
        fi
        pending="$( [[ -f "$manifest_local" ]] && echo "$(printf '%s\n' "$missing" | grep -c . || true) file(s) lagging" || echo "manifest not exported yet" )"
        echo "    waiting for reverse S3 export: $pending (attempt $attempt/18)..."
        sleep 10
    done
    [[ "$done_ok" -eq 1 ]] || die "reverse S3 export did not complete after 18 attempts (~3min); $CACHE_DIR/ may be incomplete (manifest: $manifest_local)"

    # Prune anything the prefix sync re-pulled that ISN'T in this run's manifest
    # (stale S3 orphans from prior runs whose deletes haven't exported yet), so
    # $CACHE_DIR holds ONLY this run's files. The manifest paths are read into a
    # variable ONCE and matched via a herestring — a `<(...)` process
    # substitution inside this null-delimited `while read` loop interferes with
    # the loop's own `< <(find)` and intermittently yields no match, which would
    # delete everything.
    manifest_paths="$(cut -f2- "$manifest_local")"
    pruned=0
    while IFS= read -r -d '' f; do
        rel="${f#"$CACHE_DIR"/}"
        [[ "$rel" == "$transport_manifest" ]] && continue
        [[ "$rel" == "$transport_log" ]] && continue
        grep -qxF "$rel" <<<"$manifest_paths" || { rm -f "$f"; pruned=$((pruned + 1)); }
    done < <(find "$CACHE_DIR" -type f -print0)
    [[ "$pruned" -gt 0 ]] && echo "    pruned $pruned stale orphan file(s) not in this run's manifest"
    find "$CACHE_DIR" -type d -empty -delete 2>/dev/null || true
fi
echo "==> Done. Cache in $CACHE_DIR/ (run log: $CACHE_DIR/$transport_log)"
