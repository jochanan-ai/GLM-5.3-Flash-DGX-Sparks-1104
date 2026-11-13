#!/usr/bin/env bash
# For a supervisor (systemd Type=simple, llama-swap) that runs a start script and stops it with the stop script:
#   --allowed-host NAME   the names rank 0's API is reached under through a proxy, for an API on a loopback HOST
#   PREPARE=0             a start builds, downloads and copies nothing: what is missing stops it (scripts/prepare.sh)
#   FOREGROUND=1          after the smoke test the start stays attached to rank 0 and exits when the server ends
# Sourced by start.sh and start-tp4-switchless.sh after scripts/prepare.sh (and scripts/tp4.sh); the stop scripts never
# read it. With none of the three a start behaves exactly as before.
[[ "${_GLM_SUPERVISE_LOADED:-0}" == 1 ]] && return 0
_GLM_SUPERVISE_LOADED=1

# PREPARE: auto (the default, as before: build, download and send whatever is missing), 1 (the same), 0 (check only).
PREPARE="${PREPARE:-auto}"
FOREGROUND="${FOREGROUND:-0}"
[[ "$PREPARE" =~ ^(auto|0|1)$ ]] || fail E_CONFIG "PREPARE must be auto, 0 or 1, not $PREPARE."
[[ "$FOREGROUND" =~ ^[01]$ ]] || fail E_CONFIG "FOREGROUND must be 0 or 1, not $FOREGROUND."
[[ "${FOREGROUND_LOCK_WAIT_S:-120}" =~ ^[1-9][0-9]{0,3}$ ]] || fail E_CONFIG "FOREGROUND_LOCK_WAIT_S is 1 to 9999 seconds, not $FOREGROUND_LOCK_WAIT_S."

# --- arguments -------------------------------------------------------------------------------------------------------
# start_args <launcher> "$@": MODE (start or restart) and ALLOWED_HOSTS from [restart] [--allowed-host NAME]...
# TensorFold's server on a loopback HOST answers 403 to a request whose Host header is not localhost or a loopback
# address (zig/src/server/request_guard.zig, DNS rebinding); a proxy in front of it (llama-swap) passes its own
# caller's Host on. Each NAME (a host name, an IPv4 address or a bracketed IPv6 one, no port; the engine checks it
# again) is added as --allowed-host to every rank's serve flags, so the ranks' flags stay identical. At most 64.
ALLOWED_HOSTS=()
# shellcheck disable=SC2034  # MODE is read by the start script that sources this file
start_args() {
  local usage="Usage: $1 [restart] [--allowed-host NAME]... Set serving options in the environment or scripts/local.sh."
  shift
  MODE=start
  ALLOWED_HOSTS=()
  if [[ $# -eq 1 && -z "$1" ]]; then shift; fi   # ./start.sh "" was a plain start before
  if [[ "${1:-}" == restart ]]; then MODE=restart; shift; fi
  while (( $# )); do
    case "$1" in
      --allowed-host) (( $# >= 2 )) || fail E_CONFIG "$usage"; allowed_host "$2"; shift 2 ;;
      --allowed-host=*) allowed_host "${1#*=}"; shift ;;
      *) fail E_CONFIG "$usage" ;;
    esac
  done
  (( ${#ALLOWED_HOSTS[@]} <= 64 )) || fail E_CONFIG "At most 64 --allowed-host names (the engine's limit), not ${#ALLOWED_HOSTS[@]}."
}
allowed_host() {
  [[ "$1" =~ ^[A-Za-z0-9]([A-Za-z0-9.-]{0,251}[A-Za-z0-9.])?$ || "$1" =~ ^\[[0-9A-Fa-f:.]{2,45}\]$ ]] \
    || fail E_CONFIG "--allowed-host takes a host name or an IP address without a port, not '$1'."
  ALLOWED_HOSTS+=("$1")
}
# The serve flags they add (build_serve_args appends them after the recipe's own)
allowed_host_args() { local h; for h in "${ALLOWED_HOSTS[@]}"; do SERVE_ARGS+=(--allowed-host "$h"); done; }

# --- PREPARE=0 -------------------------------------------------------------------------------------------------------
# A start that builds an image, downloads weights or copies them to a worker does what a supervisor's start must not
# (time, the network, hundreds of GB). With PREPARE=0 each such step only checks that scripts/prepare.sh did it, and a
# start that finds something missing stops before any rank runs. The NFS volumes on the workers are still made at a
# start (a Docker volume pointing at this Spark's export, no data moves).
check_only() { [[ "$PREPARE" == 0 ]]; }
# not_prepared <what is missing>: stops the start; DRY_RUN=1 only says so and goes on with the plan
not_prepared() {
  if [[ "$DRY_RUN" == 1 ]]; then warn W_PREPARE "$1 A start with PREPARE=0 stops here (run scripts/prepare.sh first)."; return 0; fi
  fail E_PREPARE "$1 PREPARE=0: a start builds, downloads and copies nothing. Run scripts/prepare.sh first."
}
if check_only && [[ "$WORKER_WEIGHTS" == rsync ]]; then
  # the same mounts as rsync, without the copy: the worker's own copy must be there already
  WORKER_WEIGHTS=check
fi
check_image() {
  if [[ "$DRY_RUN" == 1 ]]; then log "PREPARE=0: check that $IMAGE (tf.patches=$PATCHES_HASH) is on this Spark; nothing is built."; return 0; fi
  image_ready || not_prepared "The image $IMAGE (tf.patches=$PATCHES_HASH) is not on this Spark."
  log "Image is ready: $IMAGE (tf.patches=$PATCHES_HASH)"
}
# check_worker_images <ident function> <host function> <rank>...: every worker holds the very image this Spark does
check_worker_images() {
  local ident=$1 name=$2 here there i
  shift 2
  if [[ "$DRY_RUN" == 1 ]]; then log 'PREPARE=0: check that every worker holds the same image by content; nothing is sent.'; return 0; fi
  here=$(image_ident "$IMAGE")
  [[ "$here" != missing ]] || not_prepared "The image $IMAGE is not on this Spark."
  for i in "$@"; do
    there=$("$ident" "$i")
    [[ "$there" == "$here" ]] || not_prepared "$("$name" "$i") has no copy of $IMAGE (${there:0:12} against ${here:0:12} here)."
    log "[I_IMAGE] $("$name" "$i") already runs this image ($IMAGE, ${here:0:12})."
  done
}
check_weights() {
  if [[ "$MODEL_SOURCE" != download ]]; then
    log "Checkpoint: $MODEL_DESC (local, mounted read-only; no download)."
  elif weights_ready "$MODEL_DIR" "$MODEL_REVISION"; then
    log "Weights are ready: $MODEL_ID @ ${MODEL_REVISION:0:12} ($MODEL_DIR)"
  else
    not_prepared "The checkpoint $MODEL_ID @ ${MODEL_REVISION:0:12} is not complete in $MODEL_DIR (no .revision marker)."
  fi
  [[ "$DRAFTER" == dflash2 ]] || return 0
  if drafter_ready; then log "DFlash2 is ready: $DFLASH2_DIR"; else not_prepared "The DFlash2 drafter is not in $DFLASH2_DIR."; fi
}

# --- FOREGROUND=1 ----------------------------------------------------------------------------------------------------
# Without it a start ends after the smoke test and leaves the ranks running, which a supervisor reads as the server
# having ended (systemd's Type=simple then runs its ExecStop, the stop script, at once). With FOREGROUND=1:
#  - From the launch on, SIGINT and SIGTERM stop every rank (the stop script) and end the start (130, 143), and a start
#    that fails (a rank exits, WAIT_TIMEOUT, the smoke test, a context retry that does not fit) stops every rank too:
#    a supervisor's next start never finds one left behind.
#  - After the smoke test the start lets go of start.lock (so the supervisor's own stop script, or the memory guard's,
#    can run) and stays attached to rank 0: its log follows on standard output, and the start ends when rank 0's
#    container stops. Rank 0 stops by itself when any other rank goes (the engine's watch line), and its end takes the
#    other ranks down with the stop script. A stop at work (start.lock taken) is waited for, up to 300 s: if it
#    removed rank 0's container, the end was asked for and the start exits 0; if the memory guard stopped the server,
#    1; if rank 0 ended by itself, its exit code (1 when that is 0: a server that ends unasked is a failure).
#  - A server already running is attached to after its smoke test, instead of the start exiting. That smoke test
#    failing stops the server like any start that fails after the launch, and a stop by the memory guard of the start
#    that launched it reads as one that was asked for (its guard is not this start's child).
FG_STOP=''           # the stop script of this start (start.sh: stop.sh, the four-Spark starts: stop-tp4.sh)
# How long the end of an attached start waits for a stop at work (start.lock): below a supervisor's stop timeout
# (systemd's TimeoutStopSec, 240 s on the platform this was written for), so the start still stops what is left
FOREGROUND_LOCK_WAIT_S="${FOREGROUND_LOCK_WAIT_S:-120}"
FG_LOCK_HELD=0       # this start holds start.lock (fd 8)
FG_LAUNCHED=0        # from the launch on, a failure or a signal stops every rank
FG_FAILING=0
FG_LOGS=''; FG_WAIT=''; FG_JOB=''
fg_on() { [[ "$FOREGROUND" == 1 && "$DRY_RUN" != 1 ]]; }
# fg_init <stop script>: after the start script has its own functions; replaces prepare.sh's fail with one that first
# stops every rank once anything was launched
fg_init() {
  FG_STOP=$1
  fg_on || return 0
  eval "_fg_plain_$(declare -f fail)"
  # shellcheck disable=SC2329  # prepare.sh's fail, replaced: every caller of fail reaches this one
  fail() {
    if (( FG_LAUNCHED && ! FG_FAILING )); then
      FG_FAILING=1
      trap '' INT TERM
      printf '[%s] %s\n' "$1" "$2" >&2
      warn W_FOREGROUND 'FOREGROUND=1: the start failed after the launch, so every rank is stopped.'
      fg_end_jobs
      fg_stop_all >&2 || warn W_FOREGROUND "$FG_STOP could not stop every rank; see above."
      exit 1
    fi
    _fg_plain_fail "$@"
  }
}
fg_locked() { FG_LOCK_HELD=1; }   # called by the start script once it holds start.lock
# fg_launch: from here on a failure or SIGINT/SIGTERM stops every rank
fg_launch() {
  fg_on || return 0
  FG_LAUNCHED=1
  trap 'fg_signal SIGINT 130' INT
  trap 'fg_signal SIGTERM 143' TERM
}
fg_signal() {
  trap '' INT TERM
  FG_FAILING=1
  warn W_FOREGROUND "$1: stopping every rank ($FG_STOP)."
  fg_end_jobs
  fg_stop_all || warn W_FOREGROUND "$FG_STOP could not stop every rank; see above."
  exit "$2"
}
# Every helper this start runs in the background: the log follower, docker wait, the smoke test (and the curl it waits
# on), the memory guard (a stop script it started finishes: the stop that follows waits for its lock)
fg_end_jobs() {
  local p
  if [[ -n "$FG_JOB" ]] && command -v pkill >/dev/null; then pkill -TERM -P "$FG_JOB" 2>/dev/null || true; fi
  for p in "$FG_LOGS" "$FG_WAIT" "$FG_JOB" "${log_follower:-}" "${MEMGUARD_PID:-}"; do
    [[ -n "$p" ]] || continue
    kill "$p" 2>/dev/null || true
    wait "$p" 2>/dev/null || true
  done
  FG_LOGS=''; FG_WAIT=''; FG_JOB=''
}
# fg_lock: take start.lock again (after the start let it go), waiting up to FOREGROUND_LOCK_WAIT_S for a stop at work
fg_lock() {
  (( FG_LOCK_HELD )) && return 0
  mkdir -p "$STATE_DIR"
  exec 8>"$STATE_DIR/start.lock"
  flock -w "$FOREGROUND_LOCK_WAIT_S" 8 || {
    warn W_LOCK "start.lock is still held after $FOREGROUND_LOCK_WAIT_S s (a stop or a preparation at work?); stopping anyway."
    return 0; }
  FG_LOCK_HELD=1
}
fg_stop_all() { fg_lock; GLM_STOP_LOCK_HELD=1 MEMGUARD=0 "$FG_STOP"; }
# fg_run <function>: runs it in the background and waits, so a trapped signal acts at once and not after a curl of up
# to SMOKE_TIMEOUT; a fail inside it ends the background job only, and the start then fails here (stopping every rank)
fg_run() {
  if ! fg_on; then "$@"; return; fi
  ( FG_LAUNCHED=0; trap - INT TERM; "$@" ) &
  FG_JOB=$!
  if ! wait "$FG_JOB"; then FG_JOB=''; fail E_START "$1 failed (see above)."; fi
  FG_JOB=''
}
# fg_memguard_status: FG_MG, the memory guard's exit status once it ends by itself (3: it stopped the server); it
# notices within MEMGUARD_INTERVAL_S that rank 0 is gone. Run in this shell, never in $(...): only the shell that
# started the guard can wait for it.
FG_MG=0
fg_memguard_status() {
  local p=${MEMGUARD_PID:-} n
  FG_MG=0
  [[ -n "$p" ]] || return 0
  for (( n = 0; n < (MEMGUARD_INTERVAL_S + 5) * 10; n++ )); do kill -0 "$p" 2>/dev/null || break; sleep 0.1; done
  kill "$p" 2>/dev/null || true
  wait "$p" 2>/dev/null || FG_MG=$?
  MEMGUARD_PID=''
}
# fg_rank0_ended: 0 once rank 0's container has stopped or is gone, as a Docker that answers says; 1 while it runs or
# Docker does not answer (a daemon that restarts says nothing about the container)
fg_rank0_ended() {
  local st
  if st=$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null); then
    [[ "$st" != true ]]
  else
    ! docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1 && docker info >/dev/null 2>&1
  fi
}
fg_follow() { docker logs --follow "$@" "$CONTAINER_NAME" 2>&1 & FG_LOGS=$!; }
# fg_attach <docker logs options>: stay with rank 0 until it ends, then take every rank down and exit (see above)
fg_attach() {
  local code mg
  fg_on || return 0
  trap - EXIT
  exec 8>&-
  FG_LOCK_HELD=0
  log "[I_FOREGROUND] FOREGROUND=1: attached to rank 0 ($CONTAINER_NAME); its end, any rank's or SIGTERM stops every rank ($FG_STOP)."
  fg_follow "$@"
  while :; do
    docker wait "$CONTAINER_NAME" >/dev/null 2>&1 &
    FG_WAIT=$!
    wait "$FG_WAIT" || true
    FG_WAIT=''
    # docker wait also ends when the daemon goes away for a moment: only a stopped or removed container counts
    fg_rank0_ended && break
    sleep 5
    # the log follower ended with it: follow again from now
    kill -0 "$FG_LOGS" 2>/dev/null || { wait "$FG_LOGS" 2>/dev/null || true; fg_follow --since "$(date +%s)"; }
  done
  trap '' INT TERM
  FG_FAILING=1
  kill "$FG_LOGS" 2>/dev/null || true; wait "$FG_LOGS" 2>/dev/null || true; FG_LOGS=''
  fg_lock                                   # a stop at work finishes first: what it leaves tells the cases apart
  fg_memguard_status
  mg=$FG_MG
  if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    code=$(docker inspect -f '{{.State.ExitCode}}' "$CONTAINER_NAME" 2>/dev/null || true)
    [[ "$code" =~ ^[0-9]+$ ]] || code=1
    warn W_FOREGROUND "Rank 0 ended by itself (exit code $code): stopping every rank left ($FG_STOP)."
    (( code )) || code=1
  elif [[ "$mg" == 3 ]]; then
    code=1; warn W_FOREGROUND 'The memory guard stopped the server (memguard.log in the archive).'
  else
    code=0; log "[I_FOREGROUND] Rank 0's container was removed by a stop: the server ended as asked."
  fi
  GLM_STOP_LOCK_HELD=1 MEMGUARD=0 "$FG_STOP" || { warn W_FOREGROUND "$FG_STOP could not stop every rank; see above."; code=1; }
  exit "$code"
}
