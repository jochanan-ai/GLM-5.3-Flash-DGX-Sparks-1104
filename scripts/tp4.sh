# Four Sparks at TP=4 (./start-tp4-switchless.sh on a ring of cables, ./start-tp4.sh on a switch, ./stop-tp4.sh for
# both): reaching the three workers, finding the links and the RDMA devices each runs on (on a ring, NCCL's ring graphs
# too), and the image and the weights on every Spark. FABRIC_EXPECT (ring or mesh) is set by the start script.
# Sourced after scripts/prepare.sh (its helpers, config.sh, nodes.sh); start.sh and stop.sh never read it.
#
# The ring discovery (node_inventory, pair_links), the per-rank NCCL graph files (ring_graphs, graph_xml) and the NCCL
# settings of a ring are adapted from the lab's 2x DGX Sparks recipe's scripts/nodes.sh
# (github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold, branch zig-engine, start-tp4-switchless.sh), which
# measured them on this very ring. The lessons are that recipe's: with one GPU a node NCCL takes one RoCE device a
# channel for both its receive from the previous rank and its send to the next, and on a ring cabled port to like port
# a node's two neighbours sit on different ports, so NCCL's own pick (or its fused two-port device) sends part of the
# traffic to a port without that peer; a graph file per rank names, channel by channel, a device toward the previous
# rank (where the node listens), and subnet-aware routing moves each send to the device on the next rank's subnet.
# NCCL_ALGO=Ring keeps every collective between neighbours (PAT and Tree send two ranks away, which no cable reaches).
[[ "${_GLM_TP4_LOADED:-0}" == 1 ]] && return 0
_GLM_TP4_LOADED=1

# --- the three workers -------------------------------------------------------------------------------------------------
worker_ids() { echo 1 2 3; }
wvar() { if (( $2 == 1 )); then printf '%s' "$1"; else printf '%s%s' "$1" "$2"; fi; }   # WORKER, WORKER2, WORKER3
wtarget() { local v; v=$(wvar WORKER "$1"); printf '%s' "${!v}"; }
whost() { local t; t=$(wtarget "$1"); printf '%s' "${t#*@}"; }
# wssh_n <rank> <words...>: a command on worker <rank>'s host, the words quoted for its shell ($HOME left to it).
wssh_n() {
  local i=$1 a remote=''
  shift
  for a in "$@"; do
    case "$a" in '$HOME'|'$HOME'/*) remote+=" \"$a\"" ;; *) remote+=" $(printf '%q' "$a")" ;; esac
  done
  # shellcheck disable=SC2086
  ssh $SSH_OPTS -- "$(wtarget "$i")" "$remote"
}
# shellcheck disable=SC2086
wsh_n() { ssh $SSH_OPTS -- "$(wtarget "$1")" bash -s <<< "$2"; }
wfn_n() {
  local i=$1
  shift
  local -a names=()
  while (( $# )) && [[ "$1" != -- ]]; do names+=("$1"); shift; done
  shift || true
  wsh_n "$i" "$(declare -f "${names[@]}"); $*"
}

# Filled by tp4_probe_workers: each worker's home (its cache and state live under it: the workers may be other logins
# than this Spark's), uid:gid and host name.
declare -a WORKER_HOME=() WORKER_UG=() WORKER_NAME=() WORKER_CACHE=() WORKER_STATE=()
tp4_nvidia_ready() {
  local runtimes
  runtimes=$(wssh_n "$1" docker info --format '{{json .Runtimes}}' 2>/dev/null || true)
  [[ "$runtimes" != *'"nvidia"'* ]] || return 0
  wssh_n "$1" test -x /usr/bin/nvidia-container-runtime-hook >/dev/null 2>&1 && return 0
  wssh_n "$1" test -x /usr/bin/nvidia-ctk >/dev/null 2>&1 && return 0
  return 1
}
tp4_probe_workers() {
  local i out m ug home name
  for i in $(worker_ids); do
    out=$(wssh_n "$i" sh -c 'printf "%s %s:%s %s %s\n" "$(uname -m)" "$(id -u)" "$(id -g)" "$HOME" "$(hostname)"' 2>/dev/null) \
      || fail E_WORKER "Cannot reach $(wtarget "$i") (rank $i) over ssh with $SSH_OPTS. Set up key-based login (ssh-copy-id) and check $(wvar WORKER "$i") in scripts/local.sh."
    read -r m ug home name <<< "$out"
    [[ "$m" == aarch64 ]] || fail E_WORKER "$(whost "$i") reports $m, not aarch64: every rank runs the same aarch64 image."
    [[ "$home" == /* ]] || fail E_WORKER "Could not read the login's home on $(whost "$i")."
    wssh_n "$i" docker info >/dev/null 2>&1 || fail E_WORKER "$(whost "$i") cannot talk to Docker (is the login there in the docker group?)."
    tp4_nvidia_ready "$i" || fail E_WORKER "$(whost "$i") cannot give a container the GPU: no nvidia runtime registered and no NVIDIA Container Toolkit hook. Install NVIDIA Container Toolkit there."
    WORKER_HOME[i]=$home; WORKER_UG[i]=$ug; WORKER_NAME[i]=$name
    WORKER_CACHE[i]="$home/.cache/aevonix-glm53/zig-home"; WORKER_STATE[i]="${WORKER_CACHE[i]}/state"
    log "[I_WORKER] rank $i: $name ($(whost "$i")) reachable over ssh, aarch64, Docker with the NVIDIA runtime."
  done
}
# The user the ranks run as on worker i: the worker's own login when CONTAINER_USER is this Spark's login (the default),
# else CONTAINER_USER as given (root, or a fixed UID:GID).
tp4_container_user() {
  if [[ "$CONTAINER_USER" == "$(id -u):$(id -g)" ]]; then printf '%s' "${WORKER_UG[$1]}"; else printf '%s' "$CONTAINER_USER"; fi
}
tp4_worker_gpu() {
  local i names
  for i in $(worker_ids); do
    names=$(wssh_n "$i" nvidia-smi -L 2>/dev/null) || fail E_GPU "Cannot list GPUs on $(whost "$i"). Check its driver."
    grep -q 'GB10' <<< "$names" || warn W_GPU "$(whost "$i")'s GPU is not a GB10: $(head -1 <<< "$names")."
    log "[I_GPU] rank $i: $(sed -n '1s/ (UUID.*//p' <<< "$names") on $(whost "$i")."
  done
}
tp4_worker_avail_gib() { wfn_n "$1" mem_counting mem_avail_gb -- 'mem_avail_gb' 2>/dev/null | awk '{print $1}'; }
tp4_worker_memory() {
  local i theirs
  for i in $(worker_ids); do
    theirs=$(tp4_worker_avail_gib "$i")
    if [[ -z "$theirs" ]]; then warn W_RAM "Cannot read $(whost "$i")'s available memory; the start goes on."; continue; fi
    log "[I_RAM] rank $i: $theirs GiB available on $(whost "$i")."
    at_least "$theirs" "$MIN_AVAIL_GIB" \
      || fail E_MEMORY "$(whost "$i") has $theirs GiB available; MIN_AVAIL_GIB=$MIN_AVAIL_GIB is required. Stop whatever else holds its memory and try again."
  done
}
tp4_worker_disk() {
  local i root free
  for i in $(worker_ids); do
    if wssh_n "$i" docker image inspect "$IMAGE" >/dev/null 2>&1; then continue; fi
    root=$(wssh_n "$i" docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)
    free=$(wssh_n "$i" df -Pk -- "$root" 2>/dev/null | awk 'END {printf "%d\n", $4 / 1048576}')
    [[ "$free" =~ ^[0-9]+$ ]] || { warn W_DISK "Cannot read $(whost "$i")'s free disk space; the start goes on."; continue; }
    (( free >= IMAGE_FREE_GIB )) || fail E_DISK "$(whost "$i") needs about $IMAGE_FREE_GIB GiB free for the image; it has $free GiB on $root."
    log "[I_DISK] $(whost "$i") has $free GiB free for the image."
  done
}

# --- the image on every Spark --------------------------------------------------------------------------------------------
tp4_image_ident() { local s; s=$(wssh_n "$1" docker image inspect -f "$IMAGE_IDENT" "$2" 2>/dev/null) && sha256sum <<<"$s" | cut -c1-64 || echo missing; }
# Every worker runs the very image this Spark holds; a missing or different copy is streamed there (docker save | ssh
# docker load), the Spark across the ring through a neighbour.
tp4_ship_image() {
  local i here there
  if [[ "$DRY_RUN" == 1 ]]; then
    log 'Compare the image on every Spark by content; where a worker'"'"'s copy differs or is missing:'
    for i in $(worker_ids); do plan docker save "$IMAGE" '|' ssh "$(wtarget "$i")" docker load; done
    return 0
  fi
  here=$(image_ident "$IMAGE")
  [[ "$here" != missing ]] || fail E_IMAGE "The image $IMAGE is not on this Spark. Run scripts/prepare.sh --image first."
  for i in $(worker_ids); do
    there=$(tp4_image_ident "$i" "$IMAGE")
    if [[ "$here" == "$there" ]]; then log "[I_IMAGE] $(whost "$i") already runs this image ($IMAGE, ${here:0:12})."; continue; fi
    log "Sending $IMAGE to $(whost "$i") (its copy is ${there:0:12}); this takes a few minutes."
    # shellcheck disable=SC2086
    docker save "$IMAGE" | ssh $SSH_OPTS -o Compression=no -- "$(wtarget "$i")" docker load >/dev/null \
      || fail E_IMAGE "Could not load $IMAGE on $(whost "$i"). Check its disk space and Docker access."
    there=$(tp4_image_ident "$i" "$IMAGE")
    [[ "$here" == "$there" ]] || fail E_IMAGE "$(whost "$i") still reports another image for $IMAGE (${there:0:12} against ${here:0:12})."
    log "[I_IMAGE] $(whost "$i") now runs this image ($IMAGE, ${here:0:12})."
  done
}

# --- the ring --------------------------------------------------------------------------------------------------------------
# node_inventory [master]: this node's default-route netdev ("default <netdev>"), whether it routes to <master>
# ("master ok|none"), and one line per RoCE netdev that is up with an IPv4 address: "<netdev> <rdma device>
# <address/prefix> <RoCE v2 GID index of that address, or ->". Read-only (ip and sysfs). The netdevs come in uverbs
# order (rocep1s0f0, rocep1s0f1, then the PCIe twins), which is the order NCCL 2.30 numbers the devices in on DGX
# Sparks, so a graph file's "net dev N" names the device meant. Self-contained: it is sent to each worker as text.
node_inventory() {
  local n dev hca cidr want idx f u
  echo "default $(ip -o -4 route show default 2>/dev/null | sed -n 's/.* dev \([^ ]*\).*/\1/p' | head -1)"
  [[ -z "${1:-}" ]] || { ip -o -4 route get "$1" >/dev/null 2>&1 && echo "master ok" || echo "master none"; }
  for n in $(for n in /sys/class/net/*; do
               [[ -d $n/device/infiniband ]] || continue
               u=$(ls "$n/device/infiniband_verbs" 2>/dev/null | head -1); u=${u#uverbs}
               [[ "$u" =~ ^[0-9]+$ ]] || u=9999; echo "$u $n"
             done | sort -n | cut -d' ' -f2); do
    dev=${n##*/}
    [[ -d $n/device/infiniband && "$(cat "$n/operstate" 2>/dev/null)" == up ]] || continue
    hca=$(ls "$n/device/infiniband" | head -1)
    for cidr in $(ip -o -4 addr show dev "$dev" | awk '{print $4}'); do
      # shellcheck disable=SC2046
      want=$(printf '0000:0000:0000:0000:0000:ffff:%02x%02x:%02x%02x' $(tr '.' ' ' <<<"${cidr%/*}"))
      idx=-
      for f in /sys/class/infiniband/"$hca"/ports/1/gids/*; do
        [[ "$(cat "$f" 2>/dev/null)" == "$want" ]] || continue
        [[ "$(cat /sys/class/infiniband/"$hca"/ports/1/gid_attrs/types/"${f##*/}" 2>/dev/null)" == *v2* ]] && { idx=${f##*/}; break; }
      done
      echo "$dev $hca $cidr $idx"
    done
  done
}

# pair_links <ranks...> (stdin: node_inventory lines, each prefixed with its rank): two nodes are linked where they
# have addresses in one subnet (a direct cable). Prints "rank <r> <default netdev> <hcas> <gid or ->" (a node's RoCE
# devices toward all its peers, in inventory order; the GID index when it is one for all of them), "peer <i> <worker
# i's lowest address on its link to rank 0>", "missing <a> <b>" for two ranks without a common subnet, "ring
# <ranks...>" when the links form one ring through all the ranks (every rank linked to two), in its order from rank 0
# toward its lower-numbered neighbour, "toward <r> <s> <hcas>" for every linked pair (r's devices on the subnets it
# shares with s), and "prefix <len>": the shortest prefix length (longer than the links' own) at which, on every link,
# each device's address and exactly one of the peer's share a subnet and no two of a node's devices do (a port and its
# PCIe twin addressed .N and .10N in one /24: 26), so NCCL's subnet-aware routing can tell the twins apart.
pair_links() {
  python3 -c '
import ipaddress, sys
nodes, default = {}, {}
for line in sys.stdin:
    f = line.split()
    if len(f) == 3 and f[1] == "default":
        default[int(f[0])] = f[2]
    elif len(f) == 5:
        nodes.setdefault(int(f[0]), []).append((f[1], f[2], ipaddress.ip_interface(f[3]), f[4]))
ranks = [int(x) for x in sys.argv[1:]]
linked = lambda e, others: any(e[2].network == o[2].network for o in others)
links = {r: set() for r in ranks}
for r in ranks:
    used = set()
    for s in ranks:
        if s == r:
            continue
        mine = [e for e in nodes.get(r, []) if linked(e, nodes.get(s, []))]
        if not mine:
            if r < s:
                print("missing", r, s)
            continue
        links[r].add(s)
        used.update(id(e) for e in mine)
        print("toward", r, s, ",".join(dict.fromkeys(e[1] for e in mine)))
        if r == 0:
            print("peer", s, min(o[2].ip for o in nodes[s] if linked(o, mine)))
    hcas, gids = [], set()
    for e in nodes.get(r, []):
        if id(e) in used and e[1] not in hcas:
            hcas.append(e[1]); gids.add(e[3])
    print("rank", r, default.get(r) or "-", ",".join(hcas) or "-", gids.pop() if len(gids) == 1 else "-")
def splits(n):
    for r in ranks:
        for s in links[r]:
            mine = [e for e in nodes.get(r, []) if linked(e, nodes.get(s, []))]
            theirs = [o for o in nodes.get(s, []) if linked(o, mine)]
            nets = [ipaddress.ip_interface(f"{e[2].ip}/{n}").network for e in mine]
            if len(set(nets)) != len(nets) or any(sum(o[2].ip in net for o in theirs) != 1 for net in nets):
                return False
    return True
own = max((e[2].network.prefixlen for r in ranks for e in nodes.get(r, [])), default=32)
pre = next((n for n in range(own, 31) if splits(n)), None)
if pre is not None:
    print("prefix", pre)
if len(ranks) > 3 and all(len(links[r]) == 2 for r in ranks):
    order = [ranks[0], min(links[ranks[0]])]
    while len(order) <= len(ranks):
        nxt = (links[order[-1]] - {order[-2]}).pop()
        if nxt == order[0]:
            break
        order.append(nxt)
    if len(order) == len(ranks):
        print("ring", *order)
' "$@"
}

# Filled by detect_ring: each rank's bootstrap netdev, RoCE devices and GID index; the head's address toward each
# worker (a cabled address for a neighbour, the route's source for the Spark across the ring) and the worker's.
declare -a NODE_DEV=() NODE_HCAS=() NODE_GID=() NODE_GRAPH=() LINK_HEAD_ADDR=() LINK_WORKER_ADDR=()
declare -A TOWARD=()
RING_PREFIX=''
detect_ring() {
  local i r k kind a b c d pairs missing='' ring='' order='' peer
  local -a inv=()
  inv[0]=$(node_inventory)
  for i in $(worker_ids); do
    inv[i]=$(wfn_n "$i" node_inventory -- "node_inventory $(printf '%q' "$MASTER_ADDR")") \
      || fail E_FABRIC "Could not read rank $i's ($(whost "$i")) ConnectX-7 ports over ssh."
    [[ "${inv[i]}" != *'master none'* ]] \
      || fail E_FABRIC "$(whost "$i") has no route to MASTER_ADDR $MASTER_ADDR. Set MASTER_ADDR to an address of this Spark every worker reaches."
  done
  pairs=$(for r in 0 1 2 3; do sed "s/^/$r /" <<<"${inv[r]}"; done | pair_links 0 1 2 3) \
    || fail E_FABRIC 'Could not pair the Sparks'"'"' links (python3 and ip are needed on every Spark).'
  while read -r kind a b c d; do
    case "$kind" in
      rank) NODE_DEV[a]=${SOCKET_IFNAME:-$b}; NODE_HCAS[a]=$c; NODE_GID[a]=$d; [[ "$d" != - ]] || NODE_GID[a]='' ;;
      peer) LINK_WORKER_ADDR[a]=$b ;;
      missing) missing+=" $a-$b" ;;
      ring) ring="$a $b $c $d" ;;
      toward) TOWARD["$a $b"]=$c ;;
      prefix) RING_PREFIX=$a ;;
    esac
  done <<<"$pairs"
  ring=$(xargs <<<"$ring")
  if [[ "${FABRIC_EXPECT:-ring}" == mesh ]]; then
    # a switch: every pair of Sparks on one RoCE subnet, so NCCL reaches every rank directly, in any order
    if [[ -n "$missing" && -n "$ring" ]]; then
      fail E_FABRIC "These four Sparks are cabled as a ring without a switch (no RoCE subnet between ranks$missing): use ./start-tp4-switchless.sh."
    fi
    [[ -z "$missing" ]] || fail E_FABRIC "Some of these four Sparks share no RoCE subnet (ranks$missing): on a switch every Spark's ConnectX-7 ports need an address on the switch's subnet (ip -4 addr)."
  else
    [[ -n "$missing" ]] || fail E_FABRIC 'Every pair of these four Sparks shares a RoCE subnet (a switch), not a ring: use ./start-tp4.sh.'
  fi
  if [[ "${FABRIC_EXPECT:-ring}" == ring ]]; then
    [[ -n "$ring" ]] || fail E_FABRIC "The four Sparks' links do not form one ring (no common subnet between ranks$missing): cable each Spark to two others, one subnet per cable, and give both ends addresses on it."
    if [[ "$ring" != '0 1 2 3' ]]; then
      for k in $(cut -d' ' -f2- <<<"$ring"); do order+=", $(whost "$k")"; done
      fail E_FABRIC "The Sparks are cabled as a ring, but WORKER, WORKER2 and WORKER3 are not in its order (from this Spark$order, and back): set them in that order in scripts/local.sh."
    fi
  fi
  for i in $(worker_ids); do
    peer=${LINK_WORKER_ADDR[i]:-}
    if [[ -z "$peer" ]]; then
      peer=$(whost "$i")
      if [[ ! "$peer" =~ ^[0-9]+(\.[0-9]+){3}$ ]]; then
        c=$(getent ahostsv4 "$peer" | awk 'NR == 1 {print $1}') || true
        [[ -n "$c" ]] || fail E_FABRIC "Cannot resolve $peer ($(wvar WORKER "$i")) to an IPv4 address."
        peer=$c
      fi
      LINK_WORKER_ADDR[i]=$peer
    fi
    read -r LINK_HEAD_ADDR[i] _ <<<"$(link_info "$peer")" || true
    [[ -n "${LINK_HEAD_ADDR[i]:-}" ]] || fail E_FABRIC "No route from this Spark to $peer (rank $i, $(whost "$i"))."
  done
  for r in 0 1 2 3; do
    [[ -n "${NODE_DEV[r]:-}" && "${NODE_DEV[r]}" != - ]] \
      || fail E_FABRIC "rank $r has no default route: set SOCKET_IFNAME to the netdev of the network all four Sparks share (NCCL's bootstrap socket)."
    [[ -n "${NODE_HCAS[r]:-}" && "${NODE_HCAS[r]}" != - ]] \
      || fail E_FABRIC "rank $r shares no RoCE subnet with any other rank: check the ConnectX-7 cabling and addresses (ip -4 addr)."
  done
  if [[ "${FABRIC_EXPECT:-ring}" == mesh ]]; then
    log "[I_FABRIC] A switch: every pair of the four Sparks shares a RoCE subnet; RoCE v2 GID index ${NODE_GID[0]:-per device}/${NODE_GID[1]:-per device}/${NODE_GID[2]:-per device}/${NODE_GID[3]:-per device}."
    for r in 0 1 2 3; do log "[I_FABRIC] rank $r: NCCL on ${NODE_HCAS[r]} (bootstrap ${NODE_DEV[r]})."; done
    return 0
  fi
  log "[I_FABRIC] A ring: this Spark -> $(whost 1) -> $(whost 2) -> $(whost 3) -> back here; RoCE v2 GID index ${NODE_GID[0]:-per device}/${NODE_GID[1]:-per device}/${NODE_GID[2]:-per device}/${NODE_GID[3]:-per device}; subnet prefix ${RING_PREFIX:-of the links}."
  for r in 0 1 2 3; do
    a=$(( (r + 1) % 4 )); b=$(( (r + 3) % 4 ))
    log "[I_FABRIC] rank $r: NCCL on ${NODE_HCAS[r]} (bootstrap ${NODE_DEV[r]}); toward the next rank ${TOWARD[$r $a]:-?}, toward the previous ${TOWARD[$r $b]:-?}."
  done
  log "[I_FABRIC] Rank 2 ($(whost 2)) is across the ring from this Spark: its line to rank 0 and its NFS go to ${LINK_HEAD_ADDR[2]} through a neighbour."
}

# ring_graphs: an NCCL graph file for each rank, so that every ring channel runs over a cable: channel by channel a
# device toward the previous rank (where the node listens), a port and its PCIe twin in turn. Written under this
# Spark's STATE_DIR for rank 0 and under each worker's state directory for the others; mounted read-only.
ring_graphs() {
  local r prev path
  for r in 0 1 2 3; do
    prev=$(( (r + 3) % 4 ))
    if (( r == 0 )); then
      path="$STATE_DIR/nccl-graph-rank0.xml"
      if [[ "$DRY_RUN" == 1 ]]; then plan "write $path"; else mkdir -p "$STATE_DIR"; graph_xml "$r" "$prev" > "$path"; fi
    else
      path="${WORKER_STATE[r]}/nccl-graph-rank$r.xml"
      if [[ "$DRY_RUN" == 1 ]]; then plan "write $path on $(whost "$r")"
      else
        wssh_n "$r" mkdir -p "${WORKER_STATE[r]}" || fail E_FABRIC "Could not create ${WORKER_STATE[r]} on $(whost "$r")."
        graph_xml "$r" "$prev" | wssh_n "$r" sh -c "cat > $(printf '%q' "$path")" || fail E_FABRIC "Could not write $path on $(whost "$r")."
      fi
    fi
    NODE_GRAPH[r]=$path
  done
  log "[I_FABRIC] NCCL ring graphs written: $NCCL_CHANNELS channels a rank, each on a device toward the previous rank (NCCL_ALGO=$RING_NCCL_ALGO)."
}
# graph_xml <rank> <previous rank>: an NCCL graph whose channels listen on the rank's devices toward the previous rank.
graph_xml() {
  local r=$1 peer=$2 k n d idx ch='' xml
  local -a all=() toward=()
  IFS=, read -r -a all <<<"${NODE_HCAS[r]}"
  IFS=, read -r -a toward <<<"${TOWARD[$r $peer]:-}"
  (( ${#toward[@]} )) || fail E_FABRIC "rank $r has no RoCE device toward rank $peer."
  for (( k = 0; k < NCCL_CHANNELS; k++ )); do
    d=${toward[k % ${#toward[@]}]}; idx=-1
    for (( n = 0; n < ${#all[@]}; n++ )); do [[ "${all[n]}" == "$d" ]] && idx=$n; done
    ch+="    <channel><net dev=\"$idx\"/><gpu dev=\"0\" rank=\"$r\"/><net dev=\"$idx\"/></channel>"$'\n'
  done
  xml="<graphs version=\"1\">"$'\n'
  for k in "0 4" "1 3"; do
    xml+="  <graph id=\"${k% *}\" pattern=\"${k#* }\" crossnic=\"1\" nchannels=\"$NCCL_CHANNELS\" speedintra=\"12\" speedinter=\"12\" latencyinter=\"0\" typeintra=\"LOC\" typeinter=\"P2C\" samechannels=\"1\">"$'\n'"$ch  </graph>"$'\n'
  done
  for k in "4 6" "2 3" "3 5"; do
    xml+="  <graph id=\"${k% *}\" pattern=\"${k#* }\" crossnic=\"0\" nchannels=\"0\" speedintra=\"0\" speedinter=\"0\" latencyinter=\"0\" typeintra=\"LOC\" typeinter=\"LOC\" samechannels=\"0\"/>"$'\n'
  done
  printf '%s</graphs>\n' "$xml"
}

# Rank r's placement and NCCL settings (RANK_ENV) and its graph mount (RANK_MOUNTS). The bootstrap socket on the
# network every Spark shares; every RoCE device toward a neighbour; NCCL_ALGO=Ring with the graph file; crossnic and
# subnet-aware routing (each cable joins port 0 of one Spark to port 1 of the next); no NIC fusion, no P2P or SHM
# transport (one GPU a node); the prefix that keeps a port and its PCIe twin apart so both carry traffic.
# NCCL_RUNTIME_CONNECT=1: the engine otherwise asks NCCL to connect every algorithm's channels at init (so a full GB10
# never registers them later), and NCCL's tree for four ranks joins rank 0 to rank 2, which no cable reaches: measured
# here 2026-11-04, ncclCommInitRank timed out in ibv_modify_qp toward the Spark across the ring. Connected at first use,
# only the ring's channels are ever made (NCCL_ALGO=Ring), at the ranks' first all-gather, before the weights load.
# NCCL needs cuMem for that (NCCL_CUMEM_ENABLE=1); the host-side cuMem buffers stay off as the engine sets them.
# NCCL_SOCKET_IFNAME goes to rank 0 only: a follower that names one makes the engine unset NCCL_COMM_ID before NCCL
# (cuda_native.zig), and the engine's std library then stops at the first stderr lock after that ("thread N panic:
# attempt to use null value", measured here 2026-11-04 on the spark3+spark4 pair under NCCL_RUNTIME_CONNECT=1: libc's
# unsetenv shortens the environment under Zig's fixed-length view of it). Without it a follower keeps NCCL_COMM_ID,
# and NCCL finds its bootstrap interface by that address's subnet: MASTER_ADDR is therefore this Spark's address on
# the network every Spark shares (the LAN), not an alias only this Spark holds.
tp4_rank_env() {
  local r=$1
  RANK_ENV=(-e "TF_GLM_RANK=$r" -e "NCCL_IB_HCA=${NODE_HCAS[r]}")
  (( r != 0 )) || RANK_ENV+=(-e "NCCL_SOCKET_IFNAME=${NODE_DEV[r]}")
  [[ -z "${NODE_GID[r]}" ]] || RANK_ENV+=(-e "NCCL_IB_GID_INDEX=${NODE_GID[r]}")
  if [[ "${FABRIC_EXPECT:-ring}" == mesh ]]; then
    # a switch: NCCL's own algorithms and device choice over every RoCE device the rank has on the switch's subnet; one
    # GPU a node, so no P2P or SHM transport. NCCL_RUNTIME_CONNECT=1 as on the ring (channels made at first use), with
    # the cuMem setting it needs; not measured on a switch with this engine.
    RANK_ENV+=(-e NCCL_P2P_DISABLE=1 -e NCCL_SHM_DISABLE=1 -e NCCL_RUNTIME_CONNECT=1 -e NCCL_CUMEM_ENABLE=1 -e NCCL_CUMEM_HOST_ENABLE=0
               -e "NCCL_MIN_NCHANNELS=$NCCL_CHANNELS" -e "NCCL_MAX_NCHANNELS=$NCCL_CHANNELS")
    RANK_MOUNTS=()
    return 0
  fi
  RANK_ENV+=(-e "NCCL_ALGO=$RING_NCCL_ALGO" -e NCCL_GRAPH_FILE=/nccl-graph.xml -e NCCL_CROSS_NIC=1 -e NCCL_IB_SUBNET_AWARE_ROUTING=1
             -e NCCL_P2P_DISABLE=1 -e NCCL_SHM_DISABLE=1 -e NCCL_IB_MERGE_NICS=0
             -e NCCL_RUNTIME_CONNECT=1 -e NCCL_CUMEM_ENABLE=1 -e NCCL_CUMEM_HOST_ENABLE=0
             -e "NCCL_MIN_NCHANNELS=$NCCL_CHANNELS" -e "NCCL_MAX_NCHANNELS=$NCCL_CHANNELS")
  [[ -z "$RING_PREFIX" ]] || RANK_ENV+=(-e "NCCL_IB_SUBNET_PREFIX_LEN=$RING_PREFIX")
  if [[ "$ROCE_RING" == on ]]; then
    # the engine's ring all-gather: TF_ROCE_HCA names the device toward the previous rank, then the one toward the next
    local -a tprev=() tnext=()
    IFS=, read -r -a tprev <<<"${TOWARD[$r $(( (r + 3) % 4 ))]:-}"
    IFS=, read -r -a tnext <<<"${TOWARD[$r $(( (r + 1) % 4 ))]:-}"
    (( ${#tprev[@]} && ${#tnext[@]} )) || fail E_FABRIC "rank $r has no RoCE device toward both neighbours; set ROCE_RING=off."
    RANK_ENV+=(-e TF_ROCE=1 -e TF_ROCE_RING=1 -e "TF_ROCE_HCA=${tprev[0]},${tnext[0]}")
  fi
  RANK_MOUNTS=(-v "${NODE_GRAPH[r]}:/nccl-graph.xml:ro")
}

# --- the workers' weights ------------------------------------------------------------------------------------------------
# NFS_SERVER<i> names the address worker i mounts from (NFS_SERVER for all); else the head's address on its route to
# that worker: the cabled address for a neighbour, and for the Spark across the ring the address the route there leaves
# with (10.0.0.1-style, through a neighbour).
tp4_nfs_server() { local v="NFS_SERVER$1"; printf '%s' "${!v:-${NFS_SERVER:-${LINK_HEAD_ADDR[$1]}}}"; }
# tp4_nfs_volume <rank> <volume> <exported path> <container path>: a read-only NFS volume on that worker, created when
# missing and refused when it points somewhere else; proven by a read before any rank starts (a hard mount that does
# not reach the server would wedge a load).
tp4_nfs_volume() {
  local i=$1 vol=$2 path=$3 at=$4 server have
  server=$(tp4_nfs_server "$i")
  [[ -n "$server" ]] || fail E_WEIGHTS "No NFS server address for rank $i: set NFS_SERVER$i to this Spark's address as $(whost "$i") reaches it."
  if [[ "$DRY_RUN" == 1 ]]; then
    plan ssh "$(wtarget "$i")" docker volume create --driver local --opt type=nfs --opt "device=:$path" --opt "o=addr=$server,$NFS_OPTS" "$vol"
    return 0
  fi
  have=$(wssh_n "$i" docker volume inspect -f '{{index .Options "device"}} {{index .Options "o"}}' "$vol" 2>/dev/null || true)
  if [[ -z "$have" ]]; then
    wssh_n "$i" docker volume create --driver local --opt type=nfs --opt "device=:$path" --opt "o=addr=$server,$NFS_OPTS" "$vol" >/dev/null \
      || fail E_WEIGHTS "Could not create the Docker volume $vol on $(whost "$i")."
  elif [[ "${have%% *}" != ":$path" || "$have" != *"addr=$server,"* ]]; then
    fail E_WEIGHTS "$(whost "$i")'s Docker volume $vol is ${have%% *} from ${have#* }, not :$path from $server. Remove it there (docker volume rm $vol), or set NFS_MODEL_VOLUME / NFS_DFLASH2_VOLUME / NFS_SERVER$i."
  fi
  wssh_n "$i" docker run --rm --network host -v "$vol:/w:ro" --entrypoint /bin/sh "$IMAGE" -c 'test -f /w/config.json' >/dev/null 2>&1 \
    || fail E_WEIGHTS "$(whost "$i") cannot read $path over NFS from $server (volume $vol). The NFS server on this Spark must export that path, read-only, to $(whost "$i")'s address on its route here (${LINK_WORKER_ADDR[i]:-its link address}); see README \"4 Sparks\"."
  log "[I_WEIGHTS] rank $i ($(whost "$i")) reads $at over NFS from $server:$path (volume $vol)."
}
# tp4_prepare_worker_weights: the volumes on every worker (WORKER_WEIGHTS=nfs), or a check that each worker holds its
# own copy (check). rsync is not offered at four: the Spark across the ring would take the checkpoint through a
# neighbour, and three copies of 164 GiB is what NFS is for.
tp4_prepare_worker_weights() {
  local i mpath dpath
  mpath=${NFS_MODEL_PATH:-$MODEL_DIR}; dpath=${NFS_DFLASH2_PATH:-$DFLASH2_DIR}
  for i in $(worker_ids); do
    case "$WORKER_WEIGHTS" in
      nfs)
        tp4_nfs_volume "$i" "$NFS_MODEL_VOLUME" "$mpath" /model
        [[ "$DRAFTER" != dflash2 ]] || tp4_nfs_volume "$i" "$NFS_DFLASH2_VOLUME" "$dpath" /dflash2 ;;
      check)
        [[ "$DRY_RUN" == 1 ]] && continue
        wssh_n "$i" test -f "$WORKER_MODEL_DIR/config.json" >/dev/null 2>&1 \
          || fail E_WEIGHTS "$(whost "$i") has no checkpoint in $WORKER_MODEL_DIR (WORKER_WEIGHTS=check). Set WORKER_WEIGHTS=nfs to mount this Spark's copy."
        [[ -z "$MODEL_REVISION" || "$(wssh_n "$i" cat "$WORKER_MODEL_DIR/.revision" 2>/dev/null || true)" == "$MODEL_REVISION" ]] \
          || fail E_WEIGHTS "$(whost "$i")'s checkpoint in $WORKER_MODEL_DIR is not revision ${MODEL_REVISION:0:12} (WORKER_WEIGHTS=check)."
        [[ "$DRAFTER" != dflash2 ]] || wssh_n "$i" test -f "$WORKER_DFLASH2_DIR/config.json" >/dev/null 2>&1 \
          || fail E_WEIGHTS "$(whost "$i") has no DFlash2 weights in $WORKER_DFLASH2_DIR (WORKER_WEIGHTS=check)."
        log "[I_WEIGHTS] rank $i ($(whost "$i")) holds the checkpoint and the drafter." ;;
      *) fail E_CONFIG "WORKER_WEIGHTS=$WORKER_WEIGHTS: at four Sparks the workers read the checkpoint over NFS (nfs) or hold their own copy (check)." ;;
    esac
  done
}
# tp4_worker_mounts <rank>: WMOUNTS, that worker's -v arguments: the weights, the config.json overlay (placed there),
# the chat template (placed there), the rank's HOME (its digest cache) and its NCCL graph.
tp4_worker_mounts() {
  local i=$1
  WMOUNTS=()
  if [[ "$WORKER_WEIGHTS" == nfs ]]; then
    WMOUNTS+=(-v "$NFS_MODEL_VOLUME:/model:ro")
    [[ "$DRAFTER" != dflash2 ]] || WMOUNTS+=(-v "$NFS_DFLASH2_VOLUME:/dflash2:ro")
  else
    WMOUNTS+=(-v "$WORKER_MODEL_DIR:/model:ro")
    [[ "$DRAFTER" != dflash2 ]] || WMOUNTS+=(-v "$WORKER_DFLASH2_DIR:/dflash2:ro")
  fi
  if (( ${#OVERLAY_MOUNT[@]} )); then
    [[ "$DRY_RUN" == 1 ]] || wssh_n "$i" tee "${WORKER_STATE[i]}/model-config.json" < "$STATE_DIR/model-config.json" >/dev/null \
      || fail E_WEIGHTS "Could not place the config.json overlay on $(whost "$i")."
    WMOUNTS+=(-v "${WORKER_STATE[i]}/model-config.json:/model/config.json:ro")
  fi
  if [[ -n "$TEMPLATE_FILE" ]]; then
    [[ "$DRY_RUN" == 1 ]] || wssh_n "$i" tee "${WORKER_STATE[i]}/chat_template.jinja" < "$TEMPLATE_FILE" >/dev/null \
      || fail E_START "Could not place the chat template on $(whost "$i")."
    WMOUNTS+=(-v "${WORKER_STATE[i]}/chat_template.jinja:/templates/chat_template.jinja:ro")
  fi
  WMOUNTS+=(-v "${WORKER_CACHE[i]}:/state" "${RANK_MOUNTS[@]}")
}

# --- logs and the memory guard over four Sparks ------------------------------------------------------------------------------
tp4_collect_logs() {
  local i
  [[ -n "${RUN_LOG_DIR:-}" && -d "$RUN_LOG_DIR" ]] || return 0
  docker logs "$CONTAINER_NAME" > "$RUN_LOG_DIR/rank0.log" 2>&1 || true
  for i in $(worker_ids); do wssh_n "$i" docker logs "$CONTAINER_NAME" > "$RUN_LOG_DIR/rank$i.log" 2>&1 || true; done
}
# Every Spark's weights, KV pool and everything else it runs share one pool, and a box driven into reclaim can stop
# answering long before the OOM killer picks a victim. The guard samples this Spark every tick and one worker a tick
# in turn (an ssh round trip each), and stops the server when any Spark stays under the floor.
tp4_start_memguard() {
  [[ "$MEMGUARD" == 1 ]] || return 0
  if [[ "$DRY_RUN" == 1 ]]; then
    log "Memory guard: sample MemAvailable (plus parked pages) on every Spark (this one each ${MEMGUARD_INTERVAL_S} s, a worker"
    log "in turn) and run ./stop-tp4.sh when one stays under ${MEMGUARD_FLOOR_GIB} GiB for ${MEMGUARD_HOLD_S} s."
    return 0
  fi
  ( exec 8>&-
    declare -a under=(0 0 0 0) sample=('' '' '' '')
    tick=0
    while sleep "$MEMGUARD_INTERVAL_S"; do
      [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null || true)" == true ]] || exit 0
      sample[0]=$(mem_avail_gb | awk '{print $1}')
      w=$(( tick % 3 + 1 )); sample[w]=$(tp4_worker_avail_gib "$w" || true)
      tick=$(( tick + 1 ))
      for r in 0 1 2 3; do
        if [[ -n "${sample[r]}" ]] && ! at_least "${sample[r]}" "$MEMGUARD_FLOOR_GIB"; then
          under[r]=$(( under[r] + MEMGUARD_INTERVAL_S ))
        else under[r]=0; fi
        if (( under[r] >= MEMGUARD_HOLD_S )); then
          who=$(hostname); (( r == 0 )) || who=$(whost "$r")
          printf '[memguard] %s: %s GiB available, under the %s GiB floor for %s s; stopping the server\n' \
            "$who" "${sample[r]}" "$MEMGUARD_FLOOR_GIB" "$MEMGUARD_HOLD_S" | tee -a "$RUN_LOG_DIR/memguard.log" >&2
          MEMGUARD=0 "$REPO_ROOT/stop-tp4.sh" >> "$RUN_LOG_DIR/memguard.log" 2>&1 || true
          exit 3                  # it stopped the server (FOREGROUND=1 reads this)
        fi
      done
    done ) &
  MEMGUARD_PID=$!
  log "[I_MEMGUARD] Watching all four Sparks: a stop below ${MEMGUARD_FLOOR_GIB} GiB held ${MEMGUARD_HOLD_S} s (pid $MEMGUARD_PID; MEMGUARD=0 turns it off)."
}
