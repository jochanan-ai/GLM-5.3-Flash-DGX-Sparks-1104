#!/usr/bin/env bash
# One command for four Sparks cabled as a ring (TP=4, no switch): check all four, build the engine image here and send it
# to the workers, check the weights, start one rank on each Spark (rank 0 here, the API), wait for the API and
# smoke-test it. The same image, patches, settings and checks as ./start.sh; the ring's NCCL settings and the workers'
# NFS weights are scripts/tp4.sh. Four Sparks on a switch: ./start-tp4.sh, which runs this script with
# FABRIC_EXPECT=mesh. README: "4 Sparks".
# Usage: ./start-tp4-switchless.sh [restart] [--allowed-host NAME]...   (./stop-tp4.sh stops it; DRY_RUN=1 prints the plan and
# the ring found, changes nothing). For a supervisor (scripts/supervise.sh): --allowed-host, PREPARE=0, FOREGROUND=1; it
# can run ./start.sh and ./stop.sh with FABRIC_EXPECT=ring (or mesh) in the environment, which come here.
# Configuration: scripts/config.sh defaults < .env < scripts/local.sh < exported environment, with WORKER, WORKER2 and
# WORKER3 in the ring's order from this Spark (its two neighbours are WORKER and WORKER3; WORKER2 is across the ring).
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# FABRIC_EXPECT: ring (this script, four Sparks cabled to each other) or mesh (./start-tp4.sh: every Spark's ports on one
# switch). Each mode stops on the other's cabling and names the script that fits it.
export FABRIC_EXPECT="${FABRIC_EXPECT:-ring}"
case "$FABRIC_EXPECT" in
  ring) TP4_START=./start-tp4-switchless.sh; FABRIC_DESC='a ring' ;;
  mesh) TP4_START=./start-tp4.sh; FABRIC_DESC='a switch' ;;
  *) printf '[E_CONFIG] FABRIC_EXPECT must be ring or mesh, not %s.\n' "$FABRIC_EXPECT" >&2; exit 1 ;;
esac
export TP=4 GLM_TP4=1
# shellcheck source=scripts/prepare.sh
source ./scripts/prepare.sh
# shellcheck source=scripts/tp4.sh
source ./scripts/tp4.sh
# shellcheck source=scripts/banner.sh
source ./scripts/banner.sh
case "${1:-}" in
  help|-h|--help) sed -n '2,11s/^# //p' "$0"; exit 0 ;;
esac
# shellcheck source=scripts/supervise.sh
source ./scripts/supervise.sh
start_args "$TP4_START" "$@"
validate_config
# KV_SHARD=1 (engine patch 0114 here) shards each stream's latent rows over two ranks and refuses any other world
# (KvShardNeedsTwoRanks): stop here, before anything runs, rather than after the four ranks start.
[[ "$KV_SHARD" != 1 ]] || fail E_CONFIG 'KV_SHARD=1 runs at TP=2 only (engine patch 0114): unset KV_SHARD for a four-Spark start.'
# PROMPT_CACHE_MIN_GIB is start.sh's check of the pair's startup lines; this start does not read it.
[[ -z "$PROMPT_CACHE_MIN_GIB" ]] || warn W_CONFIG "PROMPT_CACHE_MIN_GIB=$PROMPT_CACHE_MIN_GIB is not checked by $TP4_START; see rank 0's prompt cache line after LIVE."
set_image
inspect_model
# The ring carries every exchange on NCCL unless ROCE_RING=on (tp4_rank_env adds the engine's four-rank RoCE ring
# all-gather per rank); the engine's two-rank RoCE path and peer-store gathers do not apply at four. On a switch NCCL
# carries everything: the ring all-gather relays through neighbours, which a switch does not need, and this engine
# has no other RoCE path for four ranks.
ROCE=off
[[ "$FABRIC_EXPECT" == ring || "$ROCE_RING" == off ]] \
  || fail E_CONFIG 'ROCE_RING=on is the ring'"'"'s RoCE relay (engine patch 0108); on a switch NCCL carries every exchange: unset ROCE_RING.'
# NCCL_COMM_ID's address: this Spark's address on the network every Spark shares, the one its default route leaves
# with (the LAN). Every rank keeps NCCL_COMM_ID and NCCL finds its bootstrap interface by that address's subnet (see
# tp4_rank_env in scripts/tp4.sh); the ranks' watch lines go to its port plus one. MASTER_ADDR overrides it.
if [[ -z "$MASTER_ADDR" ]]; then
  MASTER_ADDR=$(ip -o -4 route show default 2>/dev/null | sed -n 's/.* src \([^ ]*\).*/\1/p' | head -1)
  [[ -n "$MASTER_ADDR" ]] || MASTER_ADDR=$(ip -o -4 addr show dev "$(ip -o -4 route show default 2>/dev/null | sed -n 's/.* dev \([^ ]*\).*/\1/p' | head -1)" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)
  [[ -n "$MASTER_ADDR" ]] || fail E_CONFIG 'MASTER_ADDR is empty and this Spark has no default route: set MASTER_ADDR to an address of this Spark every worker reaches.'
fi
[[ "$MASTER_ADDR" =~ ^[0-9]+(\.[0-9]+){3}$ ]] || fail E_CONFIG "MASTER_ADDR must be an IPv4 address (the ranks' watch lines take no host name), not $MASTER_ADDR."

step() { printf '\n[%s/6] %s\n' "$1" "$2"; }
bits() { case "$1" in bf16) echo 16 ;; q8) echo 8 ;; q6) echo 6 ;; q4) echo 4 ;; esac; }
API_HOST="$HOST"
[[ "$HOST" != 0.0.0.0 && "$HOST" != :: ]] || API_HOST=127.0.0.1
[[ "$API_HOST" != *:* ]] || API_HOST="[$API_HOST]"
URL="http://$API_HOST:$PORT"

# --- the context window (as start.sh, keyed with the three workers and the reduce mode) ---------------------------------
FIT_FILE="$STATE_DIR/context-fit-tp4.tsv"
fit_key() {
  printf '%s|' "$PATCHES_HASH" "$MODEL_DIR" "$WORKER" "$WORKER2" "$WORKER3" "$TP" "$PARALLEL" "$KV" "$KV_PAGES" "$DENSE" "$HEAD" "$VISION" "$VISION_MAX_IMAGES" \
    "$DRAFTER" "$DRAFTER_BITS" "$MEMORY_RESERVE_GIB" "$PROMPT_CACHE_GIB" "$CACHE_FIRST" "$HOST_CACHE_GIB" "$PREFILL_REDUCE" "$CONTEXT_MAX" "$SERVE_EXTRA_ARGS" \
    | sha256sum | cut -c1-16
}
remembered_context() {
  [[ -f "$FIT_FILE" ]] || return 0
  awk -F'\t' -v k="$1" '$1 == k { v = $2 } END { if (v != "") print v }' "$FIT_FILE"
}
remember_context() {
  local tmp="$FIT_FILE.tmp"
  mkdir -p "$STATE_DIR"
  { if [[ -f "$FIT_FILE" ]]; then awk -F'\t' -v k="$1" '$1 != k' "$FIT_FILE"; fi
    printf '%s\t%s\t%s\n' "$1" "$2" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"; } > "$tmp"
  mv -- "$tmp" "$FIT_FILE"
}
# A smaller window the engine's memory refusals in the ranks' logs allow (start.sh's rule; every rank's figures count
# and the tightest wins).
fit_context() {
  python3 - "$1" "$2" "$CONTEXT_FIT_MARGIN" "$CONTEXT" "${CONTEXT_FIT_FIXED_GIB:-0.4}" <<'PY'
import re, sys
text = open(sys.argv[1], errors="replace").read()
asked, margin, auto, fixed = int(sys.argv[2]), float(sys.argv[3]), sys.argv[4] == "auto", float(sys.argv[5])
num = r"([0-9]+(?:\.[0-9]+)?)"
w = None
named = re.findall(r"largest fitting prompt-plus-reply window: ([0-9]+) tokens", text)
if named:
    w = min(int(x) for x in named)
if w is None:
    m = re.findall(r"--parallel ([0-9]+) needs " + num + r" GiB for its streams at a ([0-9]+)-token window, "
                   r"and the CUDA memory budget leaves " + num + " GiB", text)
    fits = []
    for g in m:
        n, x, w0, r = int(g[0]), float(g[1]), int(g[2]), float(g[3])
        s = x / max(n - 1, 1)
        f = min(fixed, s / 2)
        if s > f > 0 or s > 0:
            per_token = (s - f) / w0
            fits.append(((r + s) / n - f) / per_token * margin)
    if fits:
        w = min(fits)
if w is None:
    m = re.findall(r"one at a ([0-9]+)-token window takes " + num + r" GiB of KV pages, and the budget leaves " + num + " GiB", text)
    if m:
        w0, p, r = int(m[-1][0]), float(m[-1][1]), float(m[-1][2])
        if p > 0:
            w = w0 * r / p * margin * 0.9
if w is None:
    m = re.findall(r"one at a ([0-9]+)-token window takes " + num + r" GiB, and " + num + " GiB is left after", text)
    if m:
        w0, s, r = int(m[-1][0]), float(m[-1][1]), float(m[-1][2])
        if s > 0:
            w = w0 * r / s * margin
if w is None and auto and re.search(r"\(OutOfMemory\)|CUDA_ERROR_OUT_OF_MEMORY", text):
    w = asked / 2
if w is not None:
    w = int(w) // 1024 * 1024
    if 8192 <= w < asked:
        print(w)
PY
}
FIT_KEY=$(fit_key)
if [[ "$CONTEXT" == auto ]]; then
  CTX=$(remembered_context "$FIT_KEY")
  if [[ -n "$CTX" ]]; then CTX_NOTE="auto: the window that served these settings last time ($FIT_FILE)"
  else CTX=$CONTEXT_MAX; CTX_NOTE="auto: try $CONTEXT_MAX, then once the largest window the engine's memory refusal allows"; fi
else
  CTX=$CONTEXT; CTX_NOTE=explicit
fi

# --- what every rank is given --------------------------------------------------------------------------------------
case "$CHAT_TEMPLATE" in
  auto) TEMPLATE_FILE=; [[ "$VISION" != 1 ]] || TEMPLATE_FILE="$PWD/templates/glm-5.3-flash-vision.jinja" ;;
  none) TEMPLATE_FILE= ;;
  /*) TEMPLATE_FILE=$CHAT_TEMPLATE ;;
  *) fail E_CONFIG 'CHAT_TEMPLATE must be auto, none or an absolute file path.' ;;
esac
[[ -z "$TEMPLATE_FILE" || -f "$TEMPLATE_FILE" ]] || fail E_CONFIG "Chat template not found: $TEMPLATE_FILE"
# EFFORT_LINE=thinking: the served template is the chosen one (the checkpoint's own with VISION=0) with its effort line
# set from reasoning_effort, which the engine passes with thinking on only (null with thinking off: prompt.zig), so a
# thinking-off prompt no longer carries `<|system|>Reasoning Effort: Max`. Written to STATE_DIR every start; the base
# line must be the one the checkpoint ships, else the start stops rather than serve a template it did not understand.
# With VISION=0 the base is the checkpoint's own file, which the default MODEL_SOURCE=download fetches in step 3: until
# that download is complete (weights_ready), step 3 writes the template after it (EFFORT_AFTER_DOWNLOAD). Every other
# start checks and writes it here, before anything runs or a serving pair is stopped.
effort_template() {
  if [[ "$DRY_RUN" == 1 && "$EFFORT_AFTER_DOWNLOAD" == 1 && ! -f "$EFFORT_BASE" ]]; then
    log "Then write $STATE_DIR/chat_template.jinja: the downloaded $EFFORT_BASE with its effort line taken from reasoning_effort (EFFORT_LINE=thinking)."
    return 0
  fi
  [[ -f "$EFFORT_BASE" ]] || fail E_CONFIG "Chat template not found: $EFFORT_BASE (EFFORT_LINE=thinking rewrites its effort line)."
  [[ "$DRY_RUN" == 1 ]] || mkdir -p "$STATE_DIR"
  python3 - "$EFFORT_BASE" "$STATE_DIR/chat_template.jinja" "$DRY_RUN" <<'PY' || fail E_CONFIG "EFFORT_LINE=thinking: $EFFORT_BASE has no effort line this recipe knows; set EFFORT_LINE=always or CHAT_TEMPLATE to a file."
import sys
base = "{%- set effective_reasoning_effort = reasoning_effort if reasoning_effort is defined and reasoning_effort in ['low', 'high'] else 'max' -%}"
line = "{%- set effective_reasoning_effort = (reasoning_effort if reasoning_effort in ['low', 'high'] else 'max') if (reasoning_effort is defined and reasoning_effort is not none) else none -%}"
src = open(sys.argv[1], encoding="utf-8").read()
if src.count(base) != 1:
    sys.exit(1)
if sys.argv[3] != "1":  # DRY_RUN=1 checks the base and writes nothing
    open(sys.argv[2], "w", encoding="utf-8").write(src.replace(base, line))
PY
}
EFFORT_AFTER_DOWNLOAD=0
if [[ "$CHAT_TEMPLATE" == auto && "$EFFORT_LINE" == thinking ]]; then
  EFFORT_BASE=${TEMPLATE_FILE:-$MODEL_DIR/chat_template.jinja}
  if [[ -z "$TEMPLATE_FILE" && "$MODEL_SOURCE" == download ]] && ! weights_ready "$MODEL_DIR" "$MODEL_REVISION"; then
    EFFORT_AFTER_DOWNLOAD=1
  else
    effort_template
  fi
  TEMPLATE_FILE="$STATE_DIR/chat_template.jinja"
fi

build_serve_args() {
  local -a extra=()
  SERVE_ARGS=(--name "$SERVED_NAME" --host "$HOST" --port "$PORT" --context "$1" --parallel "$PARALLEL"
    --max-tokens "$MAX_TOKENS" --kv-cache "$KV" --no-update-check)
  if [[ "$THINKING" == 1 ]]; then SERVE_ARGS+=(--thinking); else SERVE_ARGS+=(--no-thinking); fi
  [[ -z "$REASONING_EFFORT" ]] || SERVE_ARGS+=(--reasoning-effort "$REASONING_EFFORT")
  [[ "$VISION" != 1 ]] || SERVE_ARGS+=(--vision --vision-max-images "$VISION_MAX_IMAGES")
  [[ -z "$TEMPLATE_FILE" ]] || SERVE_ARGS+=(--chat-template /templates/chat_template.jinja)
  case "$DRAFTER" in
    dflash2) SERVE_ARGS+=(--drafter /dflash2 --drafter-bits "$DRAFTER_BITS") ;;
    none) SERVE_ARGS+=(--no-drafts) ;;
  esac
  if [[ "$DRAFTER" != none ]]; then
    [[ -z "$MTP_DRAFTS" ]] || SERVE_ARGS+=(--mtp-drafts "$MTP_DRAFTS")
    [[ -z "$MTP_CONFIDENCE" ]] || SERVE_ARGS+=(--mtp-confidence "$MTP_CONFIDENCE")
  fi
  [[ -z "$PROMPT_CACHE_GIB" ]] || SERVE_ARGS+=(--prompt-cache-gib "$PROMPT_CACHE_GIB")
  # LEARN_GIB: engine patch 0113 keeps the prompt states on disk too (rank 0's CACHE_DIR/learn holds every rank's part)
  [[ "$LEARN_GIB" == 0 ]] || SERVE_ARGS+=(--learn-conversations --learn-dir /state/learn --learn-gib "$LEARN_GIB"
    --learn-min-free-gib "$LEARN_MIN_FREE_GIB")
  allowed_host_args
  read -r -a extra <<< "$SERVE_EXTRA_ARGS"
  SERVE_ARGS+=("${extra[@]}")
}

# Recipe-managed names (start.sh's, plus the ring's); any other exported TF_*, TENSORFOLD_* or NCCL_* variable reaches
# every rank by name (values never printed). The per-rank ones are added by tp4_rank_env.
declare -a PASS_ENV=()
MANAGED_ENV=' HOME HF_HUB_OFFLINE CUDA_DEVICE_ORDER CUDA_VISIBLE_DEVICES TENSORFOLD_NO_UPDATE_CHECK TENSORFOLD_NO_LIVE
 TENSORFOLD_MEMORY_RESERVE_GIB TF_GLM_DENSE_BITS TF_GLM_HEAD_BITS TF_GLM_CLEAR_THINKING TF_GLM_RANK TF_GLM_WORLD TF_CUDA_DEVICE
 TF_KV_PAGES TF_GLM_CE_EXCHANGE TF_GLM_ABSORB_MMA TF_GLM_W8A8 TF_GLM_HOST_CACHE_GIB TENSORFOLD_PREFILL_SHARE TENSORFOLD_PREFILL_BUDGET TF_PROMPT_CACHE_FIRST
 TF_GLM_KV_SHARD TF_GLM_L2PF TF_EXTRA_EOS TF_THINK_OFF_SPLIT TF_ROCE TF_ROCE_RING TF_ROCE_MAX_KB TF_ROCE_HCA TF_BIN TF_REPO TF_REF TF_GLM_PREFILL_REDUCE
 NCCL_COMM_ID NCCL_SOCKET_IFNAME NCCL_IB_HCA NCCL_IB_GID_INDEX NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS NCCL_ALGO NCCL_GRAPH_FILE
 NCCL_CROSS_NIC NCCL_IB_SUBNET_AWARE_ROUTING NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_IB_MERGE_NICS NCCL_IB_SUBNET_PREFIX_LEN
 NCCL_RUNTIME_CONNECT NCCL_CUMEM_ENABLE NCCL_CUMEM_HOST_ENABLE '
build_env_args() {
  local key
  ENV_ARGS=(-e HOME=/state/home -e HF_HUB_OFFLINE=1 -e TENSORFOLD_NO_UPDATE_CHECK=1 -e TENSORFOLD_NO_LIVE=1
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e "TENSORFOLD_MEMORY_RESERVE_GIB=$MEMORY_RESERVE_GIB"
    -e "TF_GLM_DENSE_BITS=$(bits "$DENSE")" -e "TF_GLM_HEAD_BITS=$(bits "$HEAD")" -e "TF_GLM_CLEAR_THINKING=$CLEAR_THINKING"
    -e "TF_KV_PAGES=$KV_PAGES" -e "TF_GLM_CE_EXCHANGE=$CE_EXCHANGE" -e "TF_GLM_ABSORB_MMA=$ABSORB_MMA" -e "TF_GLM_W8A8=$W8A8"
    -e "TF_CUDA_DEVICE=0" -e "TF_GLM_WORLD=$TP" -e "NCCL_COMM_ID=$MASTER_ADDR:$NCCL_PORT"
    -e "TF_GLM_PREFILL_REDUCE=$PREFILL_REDUCE" -e "TF_ROCE=$([[ "$ROCE_RING" == on ]] && echo 1 || echo 0)" -e "TF_ROCE_MAX_KB=$ROCE_MAX_KB"
    -e "TF_BIN=$TF_BIN" -e "TENSORFOLD_PREFILL_SHARE=$PREFILL_SHARE")
  [[ -z "$HOST_CACHE_GIB" ]] || ENV_ARGS+=(-e "TF_GLM_HOST_CACHE_GIB=$HOST_CACHE_GIB")
  [[ -z "$PREFILL_BUDGET" ]] || ENV_ARGS+=(-e "TENSORFOLD_PREFILL_BUDGET=$PREFILL_BUDGET")
  # CACHE_FIRST=1: engine patch 0112 sets PROMPT_CACHE_GIB aside before the streams' windows (every rank alike)
  [[ "$CACHE_FIRST" != 1 ]] || ENV_ARGS+=(-e "TF_PROMPT_CACHE_FIRST=1")
  # L2PF=1: engine patch 0117 prefetches each decode window's next weights into L2 (every rank must agree)
  [[ "$L2PF" != 1 ]] || ENV_ARGS+=(-e "TF_GLM_L2PF=1")
  # EXTRA_EOS: engine patch 0120 ends a reply at these tokens too (every rank alike)
  [[ "$EXTRA_EOS" == none ]] || ENV_ARGS+=(-e "TF_EXTRA_EOS=$EXTRA_EOS")
  # THINK_OFF_SPLIT: engine patch 0121 moves a thinking-off draft closed by </think> into reasoning_content
  [[ "$THINK_OFF_SPLIT" == 0 ]] || ENV_ARGS+=(-e "TF_THINK_OFF_SPLIT=$THINK_OFF_SPLIT")
  PASS_ENV=()
  while IFS= read -r key; do
    [[ "$MANAGED_ENV" != *" $key "* ]] || continue
    case "$key" in TF_*|TENSORFOLD_*|NCCL_*) ENV_ARGS+=(-e "$key"); PASS_ENV+=("$key") ;; esac
  done < <(compgen -e | LC_ALL=C sort)
}
# Worker i's copy of the exported variables, as a root-readable-only env file there; sets WENV_FILE (or empty).
write_worker_env() {
  local i=$1 key
  WENV_FILE=''
  (( ${#PASS_ENV[@]} )) || return 0
  WENV_FILE="${WORKER_STATE[i]}/rank.env"
  if [[ "$DRY_RUN" == 1 ]]; then
    log "  (${#PASS_ENV[@]} exported variable(s) — ${PASS_ENV[*]} — are sent to $(whost "$i") in $WENV_FILE, values not shown)"
    return 0
  fi
  { for key in "${PASS_ENV[@]}"; do printf '%s=%s\n' "$key" "${!key}"; done; } \
    | wssh_n "$i" sh -c "umask 077; cat > $(printf '%q' "$WENV_FILE")" \
    || fail E_START "Could not place the environment file on $(whost "$i")."
}

RUN_ARGS=(--gpus all --ipc=host --network host --shm-size 16g --cap-add IPC_LOCK
          --ulimit memlock=-1 --ulimit stack=67108864)
[[ ! -e /dev/infiniband ]] || RUN_ARGS+=(--device /dev/infiniband)

show_ranks() {
  local r
  for (( r = 3; r >= 0; r-- )); do
    if (( r == 0 )); then printf '  rank 0 here (%s), HTTP server and vision tower:\n    ' "$(hostname)"
    else printf '  rank %s on %s (follows rank 0'"'"'s steps, binds no port):\n    ' "$r" "$(whost "$r")"; fi
    shq env "TF_GLM_RANK=$r" "TF_GLM_WORLD=$TP" TF_CUDA_DEVICE=0 "NCCL_COMM_ID=$MASTER_ADDR:$NCCL_PORT" "TF_GLM_PREFILL_REDUCE=$PREFILL_REDUCE" \
      "$TF_BIN" serve /model "${SERVE_ARGS[@]}"
    printf '\n'
  done
}

api_curl() {
  if [[ -n "${TENSORFOLD_API_KEY:-}" ]]; then
    curl --header @<(printf 'Authorization: Bearer %s\n' "${TENSORFOLD_API_KEY%%,*}") "$@"
  else
    curl "$@"
  fi
}

# --- launch: ranks 3, 2, 1 on the workers (each retries its line to rank 0 for up to 300 s), then rank 0 here ----------
launch() {
  local ctx=$1 i a
  local -a user_args=() mounts=() cmd=() wcmd=() worker_env=()
  build_serve_args "$ctx"
  build_env_args
  RUN_LOG_DIR="$LOG_DIR/run-tp4-$(date -u +%Y%m%dT%H%M%SZ)-$$-$ATTEMPT"
  log "Ranks:"
  show_ranks
  if [[ "$DRY_RUN" != 1 ]]; then
    mkdir -p "$RUN_LOG_DIR" "$CACHE_DIR/home" "$CACHE_DIR/learn"
    ln -sfn -- "$RUN_LOG_DIR" "$LOG_DIR/current"
  fi
  for i in 3 2 1; do
    [[ "$DRY_RUN" == 1 ]] || wssh_n "$i" mkdir -p "${WORKER_CACHE[i]}/home" "${WORKER_STATE[i]}" \
      || fail E_START "Could not create ${WORKER_CACHE[i]} on $(whost "$i")."
    tp4_rank_env "$i"
    tp4_worker_mounts "$i"
    write_worker_env "$i"
    worker_env=()
    for a in "${ENV_ARGS[@]}"; do
      [[ "$a" != -e && "$a" == *=* ]] && worker_env+=(-e "$a")
    done
    [[ -z "$WENV_FILE" ]] || worker_env+=(--env-file "$WENV_FILE")
    user_args=(); [[ -z "$CONTAINER_USER" ]] || user_args=(--user "$(tp4_container_user "$i")")
    wcmd=(docker run -d --name "$CONTAINER_NAME" --label tf.recipe=glm53-spark --label "tf.patches=$PATCHES_HASH"
      --init --restart=no "${RUN_ARGS[@]}" "${user_args[@]}" "${worker_env[@]}" "${RANK_ENV[@]}" "${WMOUNTS[@]}"
      --entrypoint "$TF_BIN" "$IMAGE" serve /model "${SERVE_ARGS[@]}")
    if [[ "$DRY_RUN" == 1 ]]; then log "On $(whost "$i"), over ssh:"; plan "${wcmd[@]}"; continue; fi
    wssh_n "$i" "${wcmd[@]}" >/dev/null || { stop_all; fail E_START "Could not start rank $i on $(whost "$i"). Check its Docker and the image."; }
    log "Started rank $i on $(whost "$i")."
  done
  tp4_rank_env 0
  user_args=(); [[ -z "$CONTAINER_USER" ]] || user_args=(--user "$CONTAINER_USER")
  mounts=(-v "$MODEL_DIR:/model:ro" "${OVERLAY_MOUNT[@]}")
  [[ "$DRAFTER" != dflash2 ]] || mounts+=(-v "$DFLASH2_DIR:/dflash2:ro")
  [[ -z "$TEMPLATE_FILE" ]] || mounts+=(-v "$TEMPLATE_FILE:/templates/chat_template.jinja:ro")
  mounts+=(-v "$CACHE_DIR:/state" "${RANK_MOUNTS[@]}")
  cmd=(docker run -d --name "$CONTAINER_NAME" --label tf.recipe=glm53-spark --label "tf.patches=$PATCHES_HASH"
    --init --restart=no "${RUN_ARGS[@]}" "${user_args[@]}" "${ENV_ARGS[@]}" "${RANK_ENV[@]}" "${mounts[@]}"
    --entrypoint "$TF_BIN" "$IMAGE" serve /model "${SERVE_ARGS[@]}")
  if [[ "$DRY_RUN" == 1 ]]; then log 'Here:'; plan "${cmd[@]}"; return 0; fi
  "${cmd[@]}" >/dev/null || { stop_all; fail E_START 'Docker could not launch rank 0. Check its error and rerun.'; }
  log "Started rank 0 here (logs: $RUN_LOG_DIR)."
}

# Every container down, rank 0 first so it ends the followers' steps; used between fit attempts and on failure.
stop_all() {
  GLM_STOP_LOCK_HELD=1 MEMGUARD=0 "$REPO_ROOT/stop-tp4.sh" >/dev/null 2>&1 \
    || fail E_STOP 'The ranks would not stop (stop, then kill, verified absent). Clean up by hand: docker ps on every Spark.'
}

stop_follower() { kill "${log_follower:-}" 2>/dev/null || true; wait "${log_follower:-}" 2>/dev/null || true; trap - EXIT; }
# 0 once /health answers (rank 0 listens only after every rank loaded and agreed); 1 when a container stopped.
wait_ready() {
  local started=$SECONDS heartbeat=$SECONDS i
  docker logs --follow --tail 5 "$CONTAINER_NAME" 8>&- &
  log_follower=$!
  trap stop_follower EXIT
  until curl --fail --silent --max-time 5 "$URL/health" >/dev/null 2>&1; do
    if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null || true)" != true ]]; then
      stop_follower; return 1
    fi
    if (( (SECONDS - started) % 30 < 5 )); then
      for i in $(worker_ids); do
        if ! wssh_n "$i" docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null | grep -qx true; then
          stop_follower
          warn W_RANK "Rank $i on $(whost "$i") is no longer running."
          return 1
        fi
      done
    fi
    if (( SECONDS - started >= WAIT_TIMEOUT )); then
      stop_follower
      tp4_collect_logs
      fail E_API "The API did not come up within $WAIT_TIMEOUT s. Inspect $LOG_DIR/current/rank*.log; raise WAIT_TIMEOUT for a first start (each rank hashes the checkpoint once, three of them over NFS)."
    fi
    if (( SECONDS - heartbeat >= 30 )); then
      log "Waiting for the API: $((SECONDS - started)) s (first start: each rank hashes the checkpoint once, the workers over NFS; then the load and graphs)."
      heartbeat=$SECONDS
    fi
    sleep 5
  done
  stop_follower
}
show_failure() {
  local f
  tp4_collect_logs
  for f in "$RUN_LOG_DIR"/rank*.log; do
    [[ -f "$f" ]] || continue
    printf '\n--- %s (last 15 lines) ---\n' "$f" >&2
    tail -n 15 -- "$f" >&2
  done
}

live_message() {
  brand_style
  local weights=EXL3; [[ "$FP8" != 1 ]] || weights=FP8
  printf '%sGLM-5.3-Flash-%s is now LIVE! on port %s (four Sparks, TP=4)%s\n%sEndpoint: %s/v1%s\n' \
    "$AE_SILVER" "$weights" "$PORT" "$AE_RESET" "$AE_WHITE" "$URL" "$AE_RESET"
}

SMOKE_JSON=$(python3 - "$SERVED_NAME" <<'PY'
import json, sys
print(json.dumps({"model": sys.argv[1], "messages": [{"role": "user", "content": "Reply with exactly: ready"}],
                  "temperature": 0, "max_tokens": 32, "chat_template_kwargs": {"enable_thinking": False}}))
PY
)

# Rank 0's startup line: the pool's pages, the prompt cache and any stream the plan could not keep (README "Memory").
startup_report() {
  local f="$RUN_LOG_DIR/rank0.log" line
  [[ -f "$f" ]] || return 0
  line=$(grep -h -m1 -E 'taking [0-9]+ pages of 512 rows' "$f" 2>/dev/null || true)
  [[ -z "$line" ]] || log "Startup: ${line#*] }"
  line=$(grep -h -m1 -i -E 'prompt cache' "$f" 2>/dev/null || true)
  [[ -z "$line" ]] || log "Startup: ${line#*] }"
  if grep -q -i 'cannot keep one state' "$f" 2>/dev/null; then
    warn W_CACHE 'Rank 0 says it cannot keep one prompt state: the prompt cache is empty at this reserve (README "Memory"). Raise MEMORY_RESERVE_GIB a little or lower CONTEXT.'
  fi
}

finish_start() {
  if [[ "$DRY_RUN" == 1 ]]; then
    log "Poll every 5 s for up to $WAIT_TIMEOUT s (rank 0 listens once all four ranks have loaded and agreed):"
    plan curl --fail --silent --max-time 5 "$URL/health"
    [[ -z "${TENSORFOLD_API_KEY:-}" ]] || log '  (requests below carry Authorization: Bearer from TENSORFOLD_API_KEY; not shown)'
    plan curl --fail --silent --max-time 10 "$URL/v1/models"
    plan curl --fail --silent --show-error --max-time "$SMOKE_TIMEOUT" "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d "$SMOKE_JSON"
    log 'Would check that /v1/models lists the served name and the reply has nonempty content, then print:'
    live_message
    return 0
  fi
  tp4_collect_logs
  api_curl --fail --silent --show-error --max-time 10 "$URL/v1/models" > "$RUN_LOG_DIR/models.json" \
    || fail E_API "GET /v1/models failed (an API key mismatch?). Inspect $LOG_DIR/current/rank0.log."
  python3 -c 'import json,sys; ids=[m.get("id") for m in json.load(open(sys.argv[1])).get("data", [])]; sys.exit(0 if sys.argv[2] in ids else 1)' \
    "$RUN_LOG_DIR/models.json" "$SERVED_NAME" || fail E_API "/v1/models does not list $SERVED_NAME (see $RUN_LOG_DIR/models.json)."
  api_curl --fail --silent --show-error --max-time "$SMOKE_TIMEOUT" "$URL/v1/chat/completions" \
    -H 'Content-Type: application/json' -d "$SMOKE_JSON" > "$RUN_LOG_DIR/smoke.json" \
    || fail E_SMOKE "The smoke request failed. Inspect $LOG_DIR/current/rank0.log and retry."
  python3 - "$RUN_LOG_DIR/smoke.json" <<'PY' || fail E_SMOKE "The smoke test returned no answer, or one character repeated. Inspect $LOG_DIR/current/smoke.json and rank0.log."
import json, sys
reply = json.load(open(sys.argv[1], encoding="utf-8"))
if reply.get("error") is not None: sys.exit("an error reply")
message = reply["choices"][0]["message"]
content = message.get("content")
if message.get("role") != "assistant" or not isinstance(content, str) or not content.strip(): sys.exit("no answer from the assistant")
text = content.strip()
if len(text) >= 8 and len(set(text)) <= 2: sys.exit("one character repeated")
print("Smoke test passed: " + text[:120])
PY
  printf '\n'
  grep -h -m1 -E 'CUDA sm_[0-9]+ device' "$RUN_LOG_DIR/rank0.log" 2>/dev/null || true
  grep -h -m1 -E '^.*serving .* at http' "$RUN_LOG_DIR/rank0.log" 2>/dev/null || true
  startup_report
  printf 'Health: %s/health\nLogs: %s/current/rank*.log\n' "$URL" "$LOG_DIR"
  live_message
}

# PREPARE=0's check of the workers' images and FOREGROUND=1's stop script
tp4_image_ident_rank() { tp4_image_ident "$1" "$IMAGE"; }
fg_init "$REPO_ROOT/stop-tp4.sh"

banner
printf '\n%s\n%s at once | context %s (%s)\n%s dense, %s head, %s KV | %s drafts | vision %s | reserve %s GiB | port %s\nTP=4 on %s: rank 0 here, ranks 1-3 on %s, %s, %s | NCCL only, prefill reduce %s\n' \
  "$MODEL_DESC" "$PARALLEL" "$CTX" "$CTX_NOTE" "$DENSE" "$HEAD" "$KV$([[ "$KV_PAGES" != on ]] || echo ' (one shared page pool; context = one request'"'"'s cap)')" \
  "$DRAFTER" "$VISION" "$MEMORY_RESERVE_GIB" "$PORT" "$FABRIC_DESC" "$(whost 1)" "$(whost 2)" "$(whost 3)" "$PREFILL_REDUCE"

step 1 'Check all four Sparks: Linux aarch64, Docker, the NVIDIA runtime, memory, disk and ports'
if [[ "$DRY_RUN" == 1 ]]; then
  log '[I_DRY_RUN] Plan only: nothing is built, started, downloaded, sent or changed. The links below are probed read-only.'
  log "Check Linux aarch64, docker group access and the NVIDIA runtime here and on the three workers over ssh; a GB10 on each;"
  log "$MIN_AVAIL_GIB GiB available on each; disk for what is missing; port $PORT on $HOST and $NCCL_PORT-$((NCCL_PORT + 1)) here."
  if [[ "$MODE" == restart ]]; then DRY_RUN=1 "$REPO_ROOT/stop-tp4.sh"; fi
  tp4_probe_workers
else
  [[ "$(uname -s)" == Linux && "$(uname -m)" == aarch64 ]] || fail E_PLATFORM 'This recipe requires Linux aarch64 (DGX Spark, GB10).'
  for cmd in docker curl python3 flock gzip tar sha256sum nvidia-smi ssh ip awk getent; do
    command -v "$cmd" >/dev/null || fail E_DEPENDENCY "Missing command: $cmd. See README requirements."
  done
  docker info >/dev/null 2>&1 || fail E_DOCKER 'Cannot talk to Docker. Start the daemon and check socket permissions.'
  if (( EUID != 0 )) && ! id -nG | tr ' ' '\n' | grep -qx docker; then
    fail E_DOCKER_GROUP 'Your login is not in the docker group. Add it and start a new login session.'
  fi
  nvidia_docker_ready '' || fail E_NVIDIA_RUNTIME 'Docker cannot give a container the GPU: install NVIDIA Container Toolkit.'
  log '[I_PLATFORM] Linux aarch64, Docker access and GPU passthrough are ready here.'
  tp4_probe_workers
  mkdir -p "$STATE_DIR"
  exec 8>"$STATE_DIR/start.lock"
  flock -n 8 || fail E_LOCK 'Another start, stop or preparation is in progress.'
  fg_locked
  running=$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null || true)
  if [[ "$running" == true && "$MODE" == start ]]; then
    log "$CONTAINER_NAME is already running. Endpoint: $URL/v1"
    log "Use $TP4_START restart to apply changed settings, or ./stop-tp4.sh to stop it."
    RUN_LOG_DIR=$(readlink -f "$LOG_DIR/current" 2>/dev/null || true)
    [[ -d "$RUN_LOG_DIR" ]] || fail E_LOG 'The running container has no accessible log directory. Inspect its mounts.'
    fg_launch
    wait_ready || { show_failure; fail E_START "A rank exited. Inspect $LOG_DIR/current/rank*.log; ./stop-tp4.sh archives them."; }
    fg_run finish_start
    fg_attach --tail 50
    exit 0
  fi
  any=0
  docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1 && any=1
  for i in $(worker_ids); do wssh_n "$i" docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1 && any=1; done
  (( any == 0 )) || GLM_STOP_LOCK_HELD=1 "$REPO_ROOT/stop-tp4.sh"
  names=$(nvidia-smi -L 2>/dev/null) || fail E_GPU 'Cannot list GPUs. Check the driver.'
  grep -q 'GB10' <<< "$names" || warn W_GPU "This Spark's GPU is not a GB10: $(head -1 <<< "$names")."
  log "[I_GPU] rank 0: $(sed -n '1s/ (UUID.*//p' <<< "$names")."
  tp4_worker_gpu
  read -r mine parked <<< "$(mem_avail_gb)"
  total=$(mem_total_gb)
  if [[ -n "$mine" ]]; then
    log "[I_RAM] rank 0: $mine GiB available of $total GiB${parked:+ (${parked} GiB of it parked on the per-CPU page lists)}."
    at_least "$mine" "$MIN_AVAIL_GIB" || fail E_MEMORY "This Spark has $mine GiB available; MIN_AVAIL_GIB=$MIN_AVAIL_GIB is required. Stop whatever else holds its memory and try again."
  else warn W_RAM 'Cannot read MemAvailable on this Spark.'; fi
  tp4_worker_memory
  if ! image_ready; then
    docker_root=$(docker info --format '{{.DockerRootDir}}') || fail E_DISK 'Cannot locate Docker storage.'
    free=$(df -Pk -- "$docker_root" | awk 'END {printf "%d\n", $4 / 1048576}')
    (( free >= IMAGE_FREE_GIB )) || fail E_DISK "Need $IMAGE_FREE_GIB GiB free under $docker_root for the image build; only $free GiB available."
  fi
  log '[I_DISK] Enough disk space on this Spark for what is still missing.'
  tp4_worker_disk
  python3 - "$HOST" "$PORT" "$NCCL_PORT" <<'PY' || fail E_PORT "Port $PORT or $NCCL_PORT-$((NCCL_PORT + 1)) is in use (another server?). Stop its owner or change PORT / NCCL_PORT."
import socket, sys
host, port, nccl = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
for h, p in ((host, port), ("0.0.0.0", nccl), ("0.0.0.0", nccl + 1)):
    family = socket.AF_INET6 if ":" in h else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((h, p))
PY
  log "[I_PORT] Port $PORT and $NCCL_PORT-$((NCCL_PORT + 1)) are free."
fi

check_only || check_weights_access

if check_only; then
  step 2 'Check the TensorFold native image here and on the three workers (PREPARE=0)'
  check_image
  # shellcheck disable=SC2046
  check_worker_images tp4_image_ident_rank whost $(worker_ids)
else
  step 2 "Build the TensorFold native image with $PATCH_COUNT engine patches for sm_$SM, and send it to the three workers"
  prepare_image
  tp4_ship_image
fi

step 3 'Download or check the checkpoint and the drafter weights'
if check_only; then check_weights; else prepare_weights; fi
model_overlay
[[ "$EFFORT_AFTER_DOWNLOAD" == 0 ]] || effort_template

if [[ "$FABRIC_EXPECT" == ring ]]; then
  step 4 'Find the ring, write the NCCL graphs, and mount the weights on the workers'
else
  step 4 'Find the links on the switch, and mount the weights on the workers'
fi
detect_ring
[[ "$FABRIC_EXPECT" != ring ]] || ring_graphs
tp4_prepare_worker_weights

step 5 "Start ranks 3, 2 and 1 on $(whost 3), $(whost 2) and $(whost 1), then rank 0 here"
ATTEMPT=1
fg_launch
launch "$CTX"

step 6 'Wait for the API and run a greedy smoke test'
if [[ "$DRY_RUN" != 1 ]] && ! wait_ready; then
  while :; do
    tp4_collect_logs
    new=''
    [[ "$CONTEXT_RETRY" != 1 ]] || new=$(fit_context <(cat "$RUN_LOG_DIR"/rank*.log 2>/dev/null) "$CTX" || true)
    if [[ -z "$new" && "$CONTEXT_RETRY" == 1 && "$CONTEXT" == auto && "$ATTEMPT" == 1 && -f "$FIT_FILE" ]]; then
      new=$(sort -t$'\t' -k3,3 "$FIT_FILE" | awk -F'\t' -v asked="$CTX" 'END { if ($2 + 0 >= 8192 && $2 + 0 < asked) print $2 }')
    fi
    if grep -qh "for its other streams' fixed parts" "$RUN_LOG_DIR"/rank*.log 2>/dev/null; then
      fits=$(grep -ho 'fits: .*' "$RUN_LOG_DIR"/rank*.log 2>/dev/null | tail -1)
      t=$(grep -o -- '--context [0-9]*' <<< "$fits" | grep -o '[0-9]*$' || true)
      if [[ "$CONTEXT" == auto && -n "$t" ]] && (( t >= 8192 && t < CTX )); then
        new=$(( t * 99 / 100 / 1024 * 1024 ))
      else
        show_failure
        fail E_START "The shared KV page pool can't hold $PARALLEL streams' fixed parts beside one $CTX-token window. ${fits:+The engine says $fits. }Lower CONTEXT or MEMORY_RESERVE_GIB (now $MEMORY_RESERVE_GIB), or PARALLEL."
      fi
    fi
    if [[ -z "$new" ]] || (( ATTEMPT > ${CONTEXT_RETRIES:-3} )); then
      show_failure
      (( ATTEMPT == 1 )) && fail E_START "A rank exited during startup. Inspect $LOG_DIR/current/rank*.log; ./stop-tp4.sh archives them."
      fail E_START "Start $ATTEMPT (--context $CTX) failed too. Inspect $LOG_DIR/current/rank*.log; lower CONTEXT or PARALLEL, or raise MEMORY_RESERVE_GIB."
    fi
    warn W_CONTEXT "This start's memory holds less than a $CTX-token window at --parallel $PARALLEL; starting once more with --context $new."
    stop_all
    ATTEMPT=$(( ATTEMPT + 1 ))
    CTX=$new
    launch "$CTX"
    wait_ready && break
  done
fi
FG_SINCE=$(date +%s)                    # FOREGROUND=1 follows rank 0's log from here
tp4_start_memguard
fg_run finish_start
if [[ "$DRY_RUN" != 1 ]]; then
  if [[ "$CONTEXT" == auto ]]; then
    remember_context "$FIT_KEY" "$CTX"
    log "Context: $CTX tokens (kept for the next start with these settings; delete $FIT_FILE to probe again)."
    # Windows (KV_PAGES=off, or auto with KV=fp8) have no pages line, so no pool to report and no pipefail exit.
    pool=$(grep -ho 'taking [0-9]* pages of 512 rows' "$RUN_LOG_DIR/rank0.log" 2>/dev/null | tail -1 | grep -o '[0-9]*' | head -1) || true
    [[ -z "$pool" ]] || log "KV pool: $pool pages = $(( pool * 512 )) tokens shared by $PARALLEL streams (one request up to $CTX)."
    # One stream reads "1 stream at once"; auto names no count.
    [[ "$PARALLEL" == auto ]] || grep -qE "(^|[^0-9])$PARALLEL streams? at once" "$RUN_LOG_DIR/rank0.log" 2>/dev/null ||
      warn W_STREAMS "The engine serves fewer than PARALLEL=$PARALLEL streams at once; see the startup line in $RUN_LOG_DIR/rank0.log."
  elif (( ATTEMPT == 2 )); then
    log "Context: $CTX tokens, not the $CONTEXT asked; set CONTEXT=$CTX in scripts/local.sh to skip the refused start."
  fi
fi
if fg_on; then
  fg_attach --since "$FG_SINCE"
elif [[ "$FOREGROUND" == 1 ]]; then
  log 'FOREGROUND=1: after the smoke test the start would stay attached to rank 0 until the server ends (DRY_RUN=1: not here).'
fi
