<h1 align="center">GLM-5.3-Flash on DGX Sparks with TensorFold</h1>

<img src=".github/image.jpg" alt="GLM-5.3-Flash on DGX Sparks, now with TP4 support" width="100%">

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia's AI Lab</a></sub>
  <br><br>
  <a href="https://github.com/sponsors/MiaAI-Lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Sponsor%20me%20on%20GitHub-181717?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor me on GitHub" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

> [!IMPORTANT]
> **This is a brand-new recipe, version 1104.** It is not an update of
> [GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)
> (the Python-engine recipe and its `zig-engine` branch, which stay as they are). It descends from the lab's
> [2x RTX PRO 6000 recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-2x-RTX-PRO-6000-TensorFold): the same engine pin
> and 93 of its patches unchanged, plus a two-Spark launch over the ConnectX-7 link, unified-memory budgeting and twelve
> GB10 patches, every default measured on the pair. Different engine lineage, different patch stack, different numbers.

**Engine: TensorFold 1.1.9** (`tensorfold-native --version`), upstream commit
[`f350cd3e`](https://github.com/ashhart/TensorFold/commit/f350cd3ea253a042e205746928271f9d8f71a82d) plus this recipe's
121 engine patches (`patches/engine/`).

This recipe serves GLM-5.3-Flash from an EXL3 checkpoint on **two NVIDIA DGX Sparks** (GB10, 128 GB unified memory
each) at tensor parallel 2, one rank on each Spark, over their ConnectX-7 link. It uses TensorFold's native Zig/CUDA
server (`tensorfold-native`) built in Docker for `sm_121`. The default checkpoint is the pinned 4 bpw EXL3 quant,
downloaded on the first start and mounted read-only. The pair's KV pool holds about 2M tokens in total (fp8 on shared
pages, 12.5 GiB on each rank). Rank 0 serves the OpenAI-compatible API; rank 1 follows its steps.

```sh
./start.sh            # check both Sparks, build the image once, send it to the worker, check the weights, start both ranks, smoke-test
./stop.sh             # stop both ranks, archive the logs (logs/glm53-spark-*.tar.gz)
DRY_RUN=1 ./start.sh  # print the whole plan: the link it finds, the docker build, both ranks' command lines, the health wait
```

The engine is TensorFold 1.1.9 at commit `f350cd3e` ([ashhart/TensorFold](https://github.com/ashhart/TensorFold)), plus the
`patches/engine/*.patch` files, compiled with Zig 0.17.0 for `sm_121` by `docker/Dockerfile`.


## Why this engine on a Spark

- `sm_121` is the engine's default `-Dsm` target, GLM at tensor parallel over hosts is an upstream-measured
  configuration, and the engine's **RoCE all-gather** (`zig/src/cuda/roce.zig`) carries the decode and verify
  exchanges as one-shot RDMA writes over the Sparks' cable: same bytes, same bits as NCCL, less time.
- **Unified-memory budgeting** is upstream: the engine budgets from `/proc/meminfo` plus the pages parked on the
  kernel's per-CPU lists, not from a card's free bytes.
- The two single-host optimisations in the stack (peer-store all-gather, copy-engine prompt exchange) step aside across
  hosts instead of failing.

## What is checked

121 patches apply to `f350cd3e` with no rejects; the image builds on aarch64 for `sm_121` (about 6 minutes) and
reaches the worker by content; link discovery finds both PCIe twins of the cabled port and the RoCE v2 GID that
holds each port's own IPv4; NCCL and the engine's RDMA all-gather run over the cable; the full 1,048,576-token window
serves 8 streams beside a 3.6 GiB prompt cache; greedy replies are byte-stable across every change
([CHANGELOG.md](CHANGELOG.md)); sparkDash's quality suite scores the served template's effort line
([The effort line](#the-effort-line)). Image input was checked by a user (issue 8: PNG, and four frames in order); video input
needs the `WITH_VIDEO=1` image (checked here: a three-colour clip described in order). `W8A8` is stashed on quality.

## Performance

Served configuration: `PARALLEL=8`, `CONTEXT=1048576`, `KV=fp8` on shared pages, `DENSE=q8`, `HEAD=q8`,
`ABSORB_MMA=1`, DFlash2 drafts at 4 bits, `MEMORY_RESERVE_GIB=9.5`, image `8c4ca7bd8c39` (105 patches). The complete
measured record, traps and retractions included, is [docs/speed/RESULTS.md](docs/speed/RESULTS.md).

**Official numbers** (sparkDash, this pair, 2026-11-04). Decode is prose at temperature 0.7 with **thinking on**;
prefill is a cold prompt of the given size:

| Decode, prose, T=0.7, thinking on | 1 stream | 2 streams | 3 streams | 4 streams |
| --- | ---: | ---: | ---: | ---: |
| Total tok/s | **47.8** | **66.5** | **77.0** | **87.3** |
| tok/s per stream | 47.8 | 33.6 | 27.3 | 22.6 |
| TTFT | 200 ms | 324 ms | 437 ms | 557 ms |

| Prefill, prompt tokens | 4K | 8K | 16K | 32K | 64K | 128K | 256K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tok/s | 1,601 | 1,688 | 1,767 | 1,801 | 1,812 | 1,778 | 1,708 |
| TTFT | 2.8 s | 5.1 s | 9.5 s | 18.4 s | 36.4 s | 74.0 s | 153.7 s |

**The lab's harness** (`docs/speed/harness/`, greedy, thinking off, fixed prompts, the client on a third Spark), the
shape every engine change is gated on:

| Decode, greedy, thinking off | 1 stream | 4 streams | 6 streams | 8 streams |
| --- | ---: | ---: | ---: | ---: |
| Prose, total tok/s | 38.4 | 79-81 | 94-95 | 105-106 |
| Code, 1 stream | 56.6 | | | |
| Sampled prose, T=0.7, total | 39.1-39.3 | 75-78 | | |

A chat decoding beside a 50K-token prefill keeps 12.9 tok/s with a worst pause of 2.7 s (one 4,096-row chunk). On the
same checkpoint, prompts and day, prose at T=0.7 reads 39.1 / 75-78 tok/s (1 / 4 streams) here against 29.6 / 68-71
on TensorFold's Python engine v0.6.0 and 21.3 / 30-31 on the Zig v1.0.7 branch; under greedy decoding the v1.0.7
branch leads at 4 streams (89-95 against 79-81).

> [!WARNING]
> **Measure cold prefill with fresh prompt content.** The prompt cache resumes a shared prefix and reports a rate
> that is not prefill; `prompt_tokens_details.cached_tokens` is 0 when nothing was resumed.

Prefill was tuned on 2026-11-02 (docs/speed/RESULTS.md): `ABSORB_MMA` is +8.0% with identical answers and is on; the
4,096-row prompt chunk (0094/0095) is +7% with byte-identical replies; `W8A8` reaches +10.5% but rounds activations
(one of ten answers wrong) and stays off; `q4` dense is rejected on quality; the cable is about 3% of a long prompt's
time. At equal dense width the engine is at parity with the v1.0.7 build's prefill.

## Requirements

- **Two DGX Sparks** (GB10, 128 GB, CUDA 13 driver) cabled to each other on their ConnectX-7 ports, with IPv4
  addresses on one subnet. One cable is two netdevs and two RDMA devices (the port's two PCIe links); both are used.
- Key-based ssh from this Spark to the worker; your login in `docker` on both; Docker with the NVIDIA Container
  Toolkit and BuildKit; `rdma-core` (without it every exchange stays on NCCL, more slowly).
- `curl`, `python3`, `flock`, `tar`, `gzip`, `sha256sum`, `nvidia-smi`, `ssh`, `rsync`, `ip`, `awk`; `hf` for downloads.
- The checkpoint `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold@76c0b517` (about 164 GiB, downloaded when missing);
  both Sparks read every shard, so the worker gets it over the link (`WORKER_WEIGHTS=rsync`) or by NFS. The default
  drafter `incoai/GLM-5.3-Flash-DFlash2@bf582e4e` is CC BY-NC-ND 4.0: for commercial use set `DRAFTER=mtp` or `none`.
- Disk: about 40 GiB for the image on each Spark, 180 GiB for the checkpoint on each that holds it.

## How it starts

1. **Checks** on both Sparks: Linux aarch64, Docker, the NVIDIA runtime, a GB10, memory as the engine counts it
   (`scripts/memavail.sh`), disk, the ports.
2. **Image**: `docker build` tagged `tensorfold-glm53-spark:<hash>` over the Dockerfile, the patches and the pins;
   skipped when that label exists; the worker gets the same image by content over the link.
3. **Weights**: downloaded when missing, placed on the worker by `WORKER_WEIGHTS`; both ranks mount `/model` and
   `/dflash2`, so their startup fingerprints agree.
4. **The link**: the route to the worker gives the netdev, sysfs the port's RDMA devices and the RoCE v2 GID holding
   each side's IPv4; a second rail is probed with a full-size ping before use.
5. **Launch**: rank 1 on the worker first, then rank 0 here, each PID 1 in its own container with `--network host
   --ipc=host --device /dev/infiniband`; rank 0 serves the API and watches the follower (a rank that dies stops the
   other).
6. **Ready**: `/health`, then `/v1/models`, then a greedy smoke test.

Every rank carries the recipe's `TF_*`/`TENSORFOLD_*`/`NCCL_*` settings; any other such variable you export reaches
both ranks by name (`TENSORFOLD_API_KEY` included, never printed).

### Prefill beside decoding

A long prompt is read in atomic 4,096-row chunks (about 2.5 s each); other streams decode between chunks.
`PREFILL_SHARE=p` makes each chunk owe the decoders `(1 - p) / p` of its own time in decode rounds first. At the
default `0.68` a chat beside a long prefill keeps about 12-13 tok/s with a worst pause of one chunk, and the long
prompt's first token comes about 47% later than at `1`. Replies are byte-identical at any value.

## Memory on a 128 GB unified Spark

- VRAM and system RAM are one pool: the budget is `MemAvailable` plus parked pages, minus `MEMORY_RESERVE_GIB`.
- **Defaults since 2026-11-11, two and four Sparks alike: `MEMORY_RESERVE_GIB=6.5 CACHE_FIRST=1`**, with
  `PROMPT_CACHE_GIB` 5 on the pair, 8 with EXL3 at four Sparks and 4 with FP8. The lower reserve funds the prompt
  cache; cache-first keeps it from being lost to a third window (below). At four Sparks measured: see the
  [memory table](#4-sparks-tp4-experimental). On the pair the 5 GiB is predicted to sit beside the same two 1M
  windows (4,096 pages) and is not measured yet; check rank 0's startup line. Before, at 9.5 and windows first, per
  Spark on the served pair: model 83.3 GiB, KV pool 12.5 GiB (4,098 pages, 2.1M tokens), prompt cache 3.6 GiB.
- **With `CACHE_FIRST=0`, the page pool takes whole 1M windows first; the prompt cache gets the rest**, so `PROMPT_CACHE_GIB` only caps
  it, and each stream past the first costs about 0.42 GiB. Room differs by a few GiB between Sparks. After every
  start or change, check rank 0's startup line: `4098 pages`, `prompt cache: N GiB`, no `cannot keep one state`.
- **Cache hits land on a 4,096-token grid** (up to 4,095 trailing tokens re-prefilled, about 2.5 s), and the cache
  lives in device memory only unless `LEARN_GIB` is set.
- **Long agent sessions: `CACHE_FIRST=1 PROMPT_CACHE_GIB=12`** (patch 0112). The cache is set aside first; the eight
  streams share one 1M window (2,056 pages) and rank 0 funds a 9.3-9.7 GiB cache. A 380k-token conversation resumes
  99% cached (3.0 s instead of 234 s) alongside eight ~31k ones; decode and hashes unchanged. Trade: a second request
  past ~1M tokens of live context waits.
- **Conversations on disk: `LEARN_GIB=30`** (patch 0113, off by default). Prompt states are also written to
  `CACHE_DIR/learn`, so a conversation resumes after eviction or restart: 102.5k tokens in 1.8 s instead of 58.4 s
  cold. About 1% decode cost while writing; nothing is written under `LEARN_MIN_FREE_GIB` (15) free.
- **KV sharding: `KV_SHARD=1`** (patch 0114, off by default; TP=2 only). Each stream's latent rows live on one Spark, doubling
  the pool (8,196 pages, 4.2M tokens) for 1-4% decode; greedy replies identical. Pair it with
  `CACHE_FIRST=1 PROMPT_CACHE_GIB=6` (4.1 GiB cache, 6,147 pages). Toggling it starts an empty `LEARN_GIB` store.
- `HOST_CACHE_GIB=0` on GB10. The memory guard stops the pair when a Spark stays under 2.5 GiB available
  (`logs/current/memguard.log`). To free memory: lower `PARALLEL`, `DENSE`/`HEAD` bits or `VISION=0`; raise
  `MEMORY_RESERVE_GIB` on out-of-memory errors.

## CONTEXT=auto

The engine refuses a window that does not fit after loading, naming bytes. `CONTEXT=auto` asks for `CONTEXT_MAX`
(1,048,576) and, on a refusal with figures, restarts both ranks at the window those figures allow (times
`CONTEXT_FIT_MARGIN`, the tighter rank winning), up to `CONTEXT_RETRIES` (3) times; a window that served is kept in
`.state/context-fit.tsv` for the same settings. `CONTEXT_RETRY=0` turns it off.

## Settings

Layered: `scripts/config.sh` defaults, then `.env`, then `scripts/local.sh` (trusted Bash, gitignored), then exported
variables. **The served pair's `local.sh`** sets `PORT=8888`, `PARALLEL=8`, `CONTEXT=auto`, `KV=fp8`, `DENSE=q8`,
`HEAD=q8`, `ABSORB_MMA=1`, `HOST_CACHE_GIB=0`, `MODEL_SOURCE=local`, `WORKER_WEIGHTS=check`, and the reserve at the
default 6.5.

### The pair

| Setting | Default | Meaning |
| --- | --- | --- |
| `WORKER` | **required** | The second Spark's ssh target on the direct link, e.g. `zurih@10.0.34.4` |
| `TP` | `2` | Two Sparks with `./start.sh` |
| `FABRIC_PEER`, `FABRIC_SELF`, `NCCL_SOCKET_IFNAME`, `TF_ROCE_HCA`, `NCCL_IB_GID_INDEX` | auto | Found from the route to `WORKER`, the port's RDMA devices and its RoCE v2 GID table |
| `NCCL_CHANNELS` | `4` | `NCCL_MIN_NCHANNELS` and `NCCL_MAX_NCHANNELS` |
| `ROCE`, `ROCE_MAX_KB` | `on`, `4096` | The engine's one-shot RDMA all-gathers, and the largest shard they carry; `off` puts everything on NCCL |
| `PORT`, `HOST`, `NCCL_PORT` | `8020`, `0.0.0.0`, `29583` | The API address; rank 0's bootstrap port on the link (the watch line uses the next one) |
| `WORKER_WEIGHTS` | `rsync` | `rsync` the checkpoint over the link, `nfs` (the worker mounts it; `NFS_*` settings), or `check` (refuse unless present) |
| `WORKER_MODEL_DIR`, `WORKER_DFLASH2_DIR`, `WORKER_CACHE_DIR` | the same paths | Where the worker keeps them; the home-based defaults re-root on the worker's own home when the two logins differ |
| `SSH_OPTS` | `-o BatchMode=yes -o ConnectTimeout=10` | For every call to the worker |

### Weights and serving

| Setting | Default | Meaning |
| --- | --- | --- |
| `MODEL_SOURCE`, `MODEL_DIR` | `download`, `DATA_DIR/model` | Fetch the pinned 4 bpw checkpoint when missing, or `local` to serve `MODEL_DIR` as it is (a tree with no `.revision` marker is checked by its config.json and shards) |
| `ABLIT` | `0` | `1`: the gated [Ablit weights](#ablit-weights); needs `HF_TOKEN` |
| `DRAFTER`, `DRAFTER_BITS` | `dflash2`, `4` | `dflash2`, `mtp` (the checkpoint's own head) or `none` |
| `PARALLEL` | `4` | `--parallel`; admitted only if it fits, `auto` serves up to 8. The pair runs 8: free at 1-4 streams, 94-95 tok/s at 6, 105-106 at 8 |
| `PREFILL_SHARE`, `PREFILL_BUDGET` | `0.68`, empty | See [Prefill beside decoding](#prefill-beside-decoding) |
| `CONTEXT`, `CONTEXT_MAX` | `auto`, `1048576` | See [CONTEXT=auto](#contextauto) |
| `KV`, `KV_PAGES` | `fp8`, `on` | `--kv-cache`, one shared pool of 512-row pages with `CONTEXT` as one request's cap |
| `DENSE`, `HEAD` | `q8`, `bf16` | `bf16`, `q8`, `q6` or `q4` for the dense linears and the LM head; the pair's `local.sh` sets `HEAD=q8` |
| `MEMORY_RESERVE_GIB`, `HOST_CACHE_GIB`, `PROMPT_CACHE_GIB` | `6.5`, `0`, `5` | `PROMPT_CACHE_GIB` is `8` with EXL3 and `4` with `FP8=1` at four Sparks. See [Memory](#memory-on-a-128-gb-unified-spark); the reserve was 12 (the pair ran 9.5) before 2026-11-11 |
| `LEARN_GIB`, `LEARN_MIN_FREE_GIB` | `0`, `15` | `LEARN_GIB` > 0 keeps the prompt states on disk too, up to that many GiB (`--learn-conversations`, patch 0113); see [Memory](#memory-on-a-128-gb-unified-spark) |
| `KV_SHARD` | `0` | `1` keeps each stream's latent rows on one Spark (`TF_GLM_KV_SHARD`, patch 0114): about twice the tokens, 1-4% decode; use with `CACHE_FIRST=1 PROMPT_CACHE_GIB=6`, see [Memory](#memory-on-a-128-gb-unified-spark). Needs `KV=fp8`, `KV_PAGES=on`, DFlash2 or no drafter, `PARALLEL` auto or above 1. TP=2 only: `./start-tp4.sh` refuses it |
| `L2PF` | `0` | `1` prefetches each decode window's next weights into L2 on a side stream (`TF_GLM_L2PF`, patch 0117, from upstream): the next layer's mixer input, this layer's router and KDA's `o_proj`, not the EXL3 routed experts. Upstream turns it on at TP>=2 on GB10; here it measured +2% at one stream and flat at four and eight (exact), so it stays opt-in. `TF_GLM_L2PF_MB`, `_CTAS`, `_PACE_GBPS` (12, 2, 150) pass through as exported `TF_*` variables |
| `CACHE_FIRST` | `1` | `1` sets `PROMPT_CACHE_GIB` (required) aside before the streams' windows (`TF_PROMPT_CACHE_FIRST`, patch 0112), so the cap becomes the cache; `0` counts whole windows first and gives the cache what they leave. See [Memory](#memory-on-a-128-gb-unified-spark) |
| `PROMPT_CACHE_MIN_GIB` | empty | The smallest prompt cache a start may serve with. `start.sh` reads each rank's startup line and stops both ranks before LIVE (`E_CACHE`) when one funds less; the cache is what that start's room leaves, so `PROMPT_CACHE_GIB` caps it but does not guarantee it, even with `CACHE_FIRST=1` (issue 16). Empty: no check. `./start-tp4.sh` does not check it |
| `WITH_VIDEO` | `0` | `1` builds `libtfvideo.so` (FFmpeg 9.0.2, static, LGPL; h264/hevc/vp8/vp9/mpeg4 in mp4/mkv) into its own image tag, so `VISION=1` serves `video_url` parts; a few minutes more at build |
| `VISION`, `VISION_MAX_IMAGES`, `CHAT_TEMPLATE` | `1`, `50`, `auto` | Image input (`image_url` parts) on rank 0's tower, with `templates/glm-5.3-flash-vision.jinja`. `video_url` parts too with `WITH_VIDEO=1` (otherwise HTTP 500, issue 8); audio parts are unsupported and an mp4's audio track is ignored |
| `THINKING`, `REASONING_EFFORT`, `MAX_TOKENS`, `CLEAR_THINKING` | `0`, `xhigh`, `32768`, `1` | Request defaults (`xhigh` renders the checkpoint's trained "Reasoning Effort: Max" line); earlier turns' reasoning is dropped from the prompt |
| `EFFORT_LINE` | `thinking` | The template's `<|system|>Reasoning Effort: Max` line with thinking on only, as TensorFold v0.6.0's recipe wrote it (the served template is written to `.state/chat_template.jinja`); `always` keeps the checkpoint's template, which writes it into every prompt and makes a thinking-off reply reason in its answer. See [The effort line](#the-effort-line) |
| `EXTRA_EOS` | `<\|assistant\|>` | Tokens that end a reply besides the checkpoint's EOS ids, comma-separated (`TF_EXTRA_EOS`, patch 0120). GLM-5.3 can close a reply with `<\|assistant\|>`, which its `generation_config.json` leaves out; without it the token reaches the reply as text and the model writes a second turn until `MAX_TOKENS`. `none`: the checkpoint's list only |
| `THINK_OFF_SPLIT` | `0` | With thinking off, a chat reply that writes `</think>` within its first this-many tokens gets the text before it as `reasoning_content` and the rest as `content` (`TF_THINK_OFF_SPLIT`, patch 0121). GLM-5.3 sometimes drafts in its reasoning voice with thinking off; without the split the draft and the marker are the answer. A streamed reply's content waits until the marker, the end or this many tokens. `0`: off |
| `ABSORB_MMA`, `W8A8` | `1`, `0` | Prompt numerics: `ABSORB_MMA` is free (+8.0%, identical answers); `W8A8` rounds activations and stays off |
| `CE_EXCHANGE` | `0` | Needs CUDA IPC between ranks, which two Sparks do not have |
| `SERVED_NAME`, `SERVE_EXTRA_ARGS`, `TENSORFOLD_API_KEY` | `GLM-5.3-Flash`, empty, unset | The model id clients send; more `serve` flags; a bearer key for every route but `/health`. Requests that send no sampling fields get the checkpoint's `generation_config.json` (temperature 1.0, top_p 0.95) and no `top_k`; the pair's `local.sh` sets `SERVE_EXTRA_ARGS='--top-k 20'`, the old deployment's default (greedy replies unchanged) |

### Build, checks and timeouts

| Setting | Default | Meaning |
| --- | --- | --- |
| `SM`, `ZIG_CPU` | `121`, `native` | GB10's target; `baseline` builds an image that runs elsewhere |
| `TF_REPO`, `TF_REF`, `ZIG_VERSION`, `ZIG_SHA256`, `NCCL_WHEEL` | pinned | All hashed into the image tag |
| `IMAGE`, `IMAGE_REPO`, `DOCKER_BUILD_FLAGS`, `CONTAINER_NAME` | `tensorfold-glm53-spark:<hash>`, `glm53-spark` | |
| `MEMGUARD`, `MEMGUARD_FLOOR_GIB`, `MEMGUARD_HOLD_S`, `MEMGUARD_INTERVAL_S`, `MIN_AVAIL_GIB` | `1`, `2.5`, `4`, `2`, `100` | See [Memory](#memory-on-a-128-gb-unified-spark) |
| `DATA_DIR`, `CACHE_DIR`, `LOG_DIR`, `LOG_KEEP` | `~/.cache/aevonix-glm53`, `DATA_DIR/zig-home`, `logs`, `10` | Downloads, the ranks' `HOME`, run logs |
| `WAIT_TIMEOUT`, `SMOKE_TIMEOUT`, `STOP_TIMEOUT` | `2400`, `180`, `45` | Seconds |
| `DRY_RUN` | `0` | `1` prints the plan and changes nothing |

### Under a supervisor (systemd, llama-swap)

`scripts/supervise.sh` adds what a supervisor that starts the server with one command and stops it with another needs.
None of it changes a start that does not use it.

| Setting | Default | Meaning |
| --- | --- | --- |
| `--allowed-host NAME` (an argument, repeatable) | none | With `HOST=127.0.0.1` behind a proxy: TensorFold answers 403 to a request whose `Host` is not a loopback name (DNS rebinding guard), and a proxy passes its caller's `Host` on. Each name (a host name or an IP address, no port; at most 64) goes to every rank's `serve` flags as `--allowed-host`. `./start.sh [restart] [--allowed-host NAME]...` |
| `PREPARE` | `auto` | `0`: a start builds, downloads and copies nothing. The image here, the same image on each worker, the checkpoint (its `.revision`) and DFlash2 are checked, and a start that misses one stops with `E_PREPARE` before any rank runs; `WORKER_WEIGHTS=rsync` is checked as `check`. `scripts/prepare.sh` does the work. `auto` and `1`: as before |
| `FOREGROUND` | `0` | `1`: from the launch on, SIGINT and SIGTERM stop every rank (exit 130, 143), and a start that fails after the launch stops every rank. After the smoke test the start lets go of `start.lock` (so the supervisor's `./stop.sh` can run), follows rank 0's log and ends when rank 0's container stops, taking the other ranks down: exit 0 when a `./stop.sh` removed it (a stop that was asked for), 1 when the memory guard stopped the server, else rank 0's own code (1 for 0). A stop at work is waited for up to `FOREGROUND_LOCK_WAIT_S` (120 s, below a supervisor's stop timeout); a Docker daemon that does not answer for a while ends nothing. A server already running is attached to after its smoke test (which, failing, stops it) |
| `FABRIC_EXPECT` (environment only) | unset | `ring` or `mesh`: `./start.sh` runs `./start-tp4-switchless.sh` and `./stop.sh` runs `./stop-tp4.sh`, so a supervisor that knows one pair of scripts serves four Sparks from the same commands |

Offline tests: `python3 -m pytest tests/` (fake `docker` and `ssh`, no hardware).

## Hermes and other agents: the tool-call guard

The server turns the `<tool_call>` markup GLM writes into OpenAI `tool_calls` only for tools the request offers, as
vLLM does. Agents sometimes send a turn without the tool the model then calls, and that call reaches the agent as raw
text. With [Hermes Agent](https://github.com/NousResearch/hermes-agent) it looks like this in the reply:

```
<tool_call>session_search<arg_key>after</arg_key><arg_value>7d</arg_value><arg_key>query</arg_key><arg_value>…</arg_value></tool_call>
```

It happens on a turn sent with no `tools`, with `tool_choice: "none"`, or with a tool list that lacks the called tool:
typically the final-summary turn after the agent reaches its iteration limit, or a toolset that is switched off.
Calls to offered tools are not affected, and parallel calls parse.

**First, on the Hermes side:** raise its iteration limit if long tasks hit it, and keep the toolset that holds the tool
(for `session_search`, session search) enabled for the agent you run.

**Then, to keep the markup out of replies,** turn on the guard (engine patch 0111, off by default):

```bash
TF_TOOL_CALL_GUARD=1 ./start.sh restart
```

or, to keep it on for every start, add this line to `scripts/local.sh` (it must be exported to reach the ranks):

```bash
export TF_TOOL_CALL_GUARD=1
```

The first start after pulling this version builds the new image (the patch set changed). Check it took effect with
`docker exec glm53-spark env | grep TF_TOOL_CALL_GUARD`.

What it does, streamed or not:

- A request that offers no tools (none sent, or `tool_choice: "none"`) whose history called tools gets a closing system
  note that tools are unavailable for this reply, and any `<tool_call>` block the model writes anyway is removed. The
  reply is then prose (the model answers from what it has) or empty (it only wanted to call).
- A request that offers tools has well-formed calls to tools it did not offer removed; calls to offered tools come back
  as `tool_calls` as before. A malformed call (no name, a name over two lines, broken JSON) stays in the reply as
  text, as with the guard off, so the client and the log show what the model wrote. When removing calls would leave
  the reply with no text and no call, they stay too: an empty reply reads to an agent as a failed generation, and
  Hermes retries it unchanged until its timeout (patch 0116).
- Each removal is logged: `docker logs glm53-spark 2>&1 | grep "tool-call guard"`.

Guard on or off, every call the server does not turn into `tool_calls` gets one log line with the request id, the
reason (`unknown_tool_name`, `malformed`, or `unclosed` for a call cut before its closer), the tool name, the
request's thinking mode, and what became of it:

```
tool call rejected from chatcmpl-…: malformed [''] (23 tools offered) thinking=True effort=max; kept as text
```

`docker logs glm53-spark 2>&1 | grep "tool call rejected"` lists them. To see the envelope the model wrote, add
`export TF_TOOL_CALL_DEBUG=1` to `scripts/local.sh` and restart: each line is then followed by the envelope's text
(escaped to one line and cut at about 1 KB). The text is the model's output, so leave it off where logs are shared.

An agent error such as Hermes' `tool_call calls[0] requires a 'name'` is not the server's: Hermes offers a bridge
tool named `tool_call` for its deferred tools, the server passed the model's call to it as `tool_calls`, and Hermes
refused the arguments. No `tool call rejected` line goes with it.
- Off (the default), every reply is byte-identical to a build without the patch. On, replies to ordinary requests are
  unchanged too (same greedy hashes, same speed): the guard only acts on the requests above.

Leave it off for a client that reads tool calls out of the reply text itself (tools described in its own system prompt,
no `tools` field): the guard would remove the calls that client is waiting for.

## The effort line

The checkpoint's chat template opens every prompt with `<|system|>Reasoning Effort: Max` (unless `reasoning_effort`
is `low` or `high`), thinking off included, and a thinking-off reply then reasons in its answer. TensorFold v0.6.0's
recipe wrote the line with thinking on only.

`EFFORT_LINE=thinking` (default) does the same: `start.sh` serves a copy of the template
(`.state/chat_template.jinja`) that writes the line only when thinking is on. `EFFORT_LINE=always` serves the
template as it is; a `CHAT_TEMPLATE` of your own is always served as it is.

Measured with sparkDash's quality suite (greedy, 755 items): with the line in every prompt, QA 79.3% against
v0.6.0's 84.0% and MMLU 91.2%; with `EFFORT_LINE=thinking`, QA 86.0% (140 of 150 replies byte-identical to v0.6.0's)
and MMLU 88.4%, thinking-on categories unchanged.

## Engine patches

`patches/engine/` holds 121 patches on TensorFold `f350cd3e`, applied in filename order at image build: 0001-0093 are
the [RTX PRO 6000 recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-2x-RTX-PRO-6000-TensorFold)'s, unchanged;
0094-0106 and 0111-0121 are this pair's own or ported from that recipe or from upstream. 0107-0110 are the four-Spark patches. None is GB10-specific: the arch-gated kernels test
`__CUDA_ARCH__ >= 1200`, which `sm_121` satisfies.

| Patches | What they do |
| --- | --- |
| 0001–0012 | EXL3 experts at mixed bit widths, 3-bit prompt kernels; FP8 attention loads, the shared expert on a side stream; one expert call a prompt pass, no MTP pass beside DFlash2 |
| 0013–0016 | DFlash2 reads complete prompt taps at TP>1 (upstream #1623); no target MTP layer beside DFlash2; a pinned host prompt-state tier (off here) |
| 0017–0033 | FP8 KV on shared pages; EXL3 load-ahead, fused MoE glue, KDA in one launch, one broadcast a follower step |
| 0034–0035 | Peer-store all-gather and copy-engine prompt exchange, both stepping aside across hosts |
| 0036–0048 | DFlash2 copy rounds, `ABSORB_MMA` (opt-in), "fits:" refusals, the device prompt-cache share |
| 0049–0066 | Paired prompt chunks (upstream #1423) extended to DFlash2; `W8A8` (opt-in) |
| 0067–0091 | DSA select on many CTAs, audited programmatic dependent launches along the decode chain, glue fusions, EXL3 prompt and decode kernels; each keeps an off switch (`TF_GLM_DSEL=0`, `TF_GLM_PDL_CHAIN=0`, `TF_GLM_MM_TWO=0`, `TF_GLM_TOPK_SUM=0`, `TF_GLM_SHARED_SEGS=0`, `TF_EXL3_SHFL=0`, `TF_EXL3_PDL=0`, `TF_GLM_KDA_STEP_WARPS=8`, `TF_GLM_HC_SPLIT=0`, `TF_GLM_SHARED_WRITES=0`, `TF_GLM_HC_PART=0`, `TF_GLM_DSCORES=0`). Never ship 0086 or 0089 without 0090 |
| 0092–0093 | A JSON answer stays JSON beside offered tools on chat, Responses and Messages; a reply grammar is freed when its stream cannot start |
| 0094–0097 | **This pair's.** A 4,096-row prompt chunk with DFlash2's `MAX_CTX` to match (+7% prefill, byte-identical replies); the GB10's measured cost tables and draft-rule seed |
| 0098–0102 | **This pair's,** env-gated, defaults behaviour-identical: `TF_GLM_DFLASH2_EDGE`, `TF_GLM_PROF` on shared windows, `TF_GLM_MM_STAGES`, `TF_GLM_DFLASH2_TCONF`, `TF_GLM_DFLASH2_Q32` (the decode campaign's exact knobs, each measured neutral) |
| 0103–0104 | From the RTX recipe: a tool call after prose that quotes a tool-call opener is the call (with a guard so a call quoted inside a call's value is never invoked); GLM warm turns, so a user message after an agent loop resumes past the loop (8,192 of 9.3K tokens cached, was 4,096) |
| 0105 | An untyped tool parameter takes the JSON object or array the model wrote; text stays text (sibling recipe issue #180) |
| 0107-0110 | Four Sparks: block FP8 checkpoints, the four-rank RoCE ring all-gather (opt-in), the DFlash2 drafter MLP split across every rank, `TF_GLM_CHUNK_SOLO` (opt-in); see [4 Sparks](#4-sparks-tp4-on-a-ring-experimental) |
| 0111 | `TF_TOOL_CALL_GUARD=1` (opt-in, off by default): tool-call markup never reaches the client as text. A request that offers no tools (none sent, or `tool_choice: "none"`) although its history called them gets a closing system note that tools are unavailable, and any `<tool_call>` the model writes anyway is hidden; a request that offers tools has calls to tools it did not offer, and malformed calls, hidden. Streamed or whole; each hide is logged (`tool-call guard:`). Off: replies byte-identical |
| 0112 | `TF_PROMPT_CACHE_FIRST=1` (`CACHE_FIRST=1`, opt-in): the prompt cache is set aside before the room is counted in whole windows; off, the plan is as before. The startup log says when the room funds less cache than `--prompt-cache-gib` asks, and what frees more |
| 0113 | `--learn-conversations` for GLM on CUDA (`LEARN_GIB`, off by default), a backport of upstream [#1411](https://github.com/ashhart/TensorFold/pull/1411): kept prompt states on disk, every rank's part, written behind the request and read back (ahead of a queued turn when it can) after an eviction or a restart. On this stack the disk image copies 0015's two-tier lease page by page and rank 1's part travels the follow link of 0031. `--spill-gib` and `--checkpoint-slots` are refused for GLM. Off: replies byte-identical |
| 0114 | KV sharding (`KV_SHARD=1`, `TF_GLM_KV_SHARD`, off by default) at TP=2, toward upstream [#1431](https://github.com/ashhart/TensorFold/issues/1431): each stream's FP8 latent rows on its owner rank, a second page table pointing off-owner at a discard page, owner-computed decode and verify (no new kernel), prompts on a borrowed twin, kept states carried to another owner in 32 MiB pieces, and 0113's disk image under it. Refused by name with the MTP drafter, `--parallel 1`, other KV caches, or a part under 131k tokens. `tf-cuda-test glm-kv-shard-check` checks it against an unsharded twin byte for byte. Off: replies byte-identical |
| 0115 | **This pair's.** The clean image build prints no compiler warning: two fp8 expert kernels' unread kernel-local `RW`, `prefill_attention`'s readerless `SMEM_MAX`, `glm_mm`'s `Tile` `SMEM` member no instantiation uses, and `flashnext_fact` / `flashnext_attn` unroll loops that sat behind an `if constexpr` they could not reach (now behind the inverted guard, so each instantiation compiles only its reachable side). No arithmetic, launch shape or output changes: all six files' fatbins compile byte-identical |
| 0116 | **This pair's.** Each tool envelope the parser rejects is logged on its own line: request id, reason (`unknown_tool_name`, `malformed`, `unclosed`), tool name, thinking and effort, and whether it was kept or hidden; `TF_TOOL_CALL_DEBUG=1` adds the envelope text. With `TF_TOOL_CALL_GUARD=1` and tools offered, only well-formed calls to unoffered tools are hidden; malformed calls stay as text, and a reply is never emptied by the guard. Host tests: rejection reasons, the guard's hiding whole and streamed. Guard off: replies byte-identical |
| 0117 | From upstream (8cc6ac420, a0ddae24d; opt-in): `TF_GLM_L2PF=1` (`L2PF=1`) reads a decode window's next weights into L2 on a side stream while the gathers and chains run, with DeepSeek V4.1's `l2pf` / `l2pace` kernels and their nvcc 13.0 contract records. Three sites a layer: the next mixer's input before `gather_b`, the router (or a dense layer's gate and up) before `gather_a`, KDA's `o_proj` before its chain, including the shared window's batched chains. Only reads; the side stream joins once a window and drains before it is freed. The ranks' fingerprint carries it (layout 16). Off: no side stream, no change. On the pair (2026-11-11, greedy, token hashes identical): one stream 37.6 to 38.4 tok/s (round 55.1 to 53.9 ms), four and eight streams within noise (72.7 to 73.1, 121.1 to 121.4) |
| 0118 | From upstream (b10fda39f, 1b5ac18bb, 7faa962a0): a reply `max_tokens` cuts inside a JSON `<tool_call>` that has named an offered tool returns that call with its arguments as far as written, as XML and GLM calls already did; a call cut before its name, or naming a tool not offered, stays text (so 0111/0116's guard sees the same offered-only calls). A whole `/v1/messages` reply closes the cut call's input where it parses instead of answering 500. Two local fixes over upstream: a key with no value (`{"name":,`) no longer indexes an empty slice, and the cut call's name matches offered tools trimmed and in any case, as a closed call's does. Changes only replies cut by `max_tokens` in a JSON call. Not yet measured on the pair |
| 0119 | From upstream (6d4191a95, #2257, adapted): decode windows of 32 rows (0039 had 16), so a context copy verifies up to 31 tokens a round; DFlash2 still drafts 7. Fixes what the wider window exposes on this stack: shared-window scratch is said outright (a 32-row segment no longer shrinks the KDA projection buffer to 4 bytes or drops 0075/0088's one-launch segment attention), the follower's draws hold 32 rows, and the window width joins `--learn-conversations`' store identity. Not taken: load-timed pricing, 15-deep MTP chains, and #1779's shared reservation (upstream turned it off for DFlash2, cc2fa8d23). On the pair (2026-11-11, greedy, token hashes identical): a 1,195-token file rewrite at one stream 151 to 201-205 tok/s decode (14.9 to 26.6 tokens a round); prose at 1/4/8 streams and four-stream rewrites unchanged; 34 MiB more fixed memory a stream |
| 0120 | `TF_EXTRA_EOS=<token>[,<token>...]` (`EXTRA_EOS`, on by default with `<\|assistant\|>`): the named tokens end a reply like the checkpoint's EOS ids and leave the reply as they do. A name the tokenizer does not hold refuses the start; unset, the EOS list is the checkpoint's |
| 0121 | `TF_THINK_OFF_SPLIT=<tokens>` (`THINK_OFF_SPLIT`, opt-in, off by default (`THINK_OFF_SPLIT=0`; 1024 is the hold the author measured)): a thinking-off chat reply that writes the think close token within the hold is split like a thinking reply, whole and streamed alike (a stream holds its content until the token, the hold or the end). A close after the hold stays text; raw completions, the JSON channel and logprobs requests are never split. Tests: `zig build test-think-off-split-host` |

Every default-on patch was A/B tested against the image before it: greedy replies identical plus a measured gain.

## 4 Sparks (TP=4, experimental)

Two launchers run the same image and settings as tensor parallel over four Sparks; each checks the cabling and stops
on the other's, naming the script that fits:

- **`./start-tp4-switchless.sh`: four Sparks cabled to each other without a switch** (the one run on this kit). Each
  Spark's two ConnectX-7 ports go to its two neighbours, one subnet per cable (a port and its PCIe twin on it); the
  two Sparks across the ring from each other are reachable only through a neighbour (an address routed over the
  cables, or the LAN). `WORKER`, `WORKER2` and `WORKER3` are the other three **in the ring's order** from rank 0
  (`WORKER` and `WORKER3` cabled to it, `WORKER2` across).
- **`./start-tp4.sh`: four Sparks on a switch** (written from the earlier Python-engine recipe's switch launcher, not
  yet run on a switch with this engine). Every Spark's ports share the switch's subnet, so the workers may be in any
  order; NCCL picks its own algorithms and devices (no ring graph files, no `NCCL_ALGO=Ring`, no subnet-aware routing)
  and carries every exchange: `ROCE_RING`, the ring's relay, is refused there, and this engine has no other RoCE path
  for four ranks. Everything else (the image, the weights over NFS, the settings, the checks) is the ring's.

Rank 0 is the Spark that runs the script (the API). `./stop-tp4.sh` stops the four ranks of either. `DRY_RUN=1` on
either prints the links it finds and every rank's command, and changes nothing.

**Weights: EXL3 (default) or FP8.** Both launchers serve either checkpoint; `FP8` picks it:

| | On a switch | On a ring (no switch) |
| --- | --- | --- |
| EXL3 4-bit (the pair's checkpoint, the default) | `./start-tp4.sh` | `./start-tp4-switchless.sh` |
| Block FP8 (Z.AI's own release, `FP8=1`) | `FP8=1 ./start-tp4.sh` | `FP8=1 ./start-tp4-switchless.sh` |

- **EXL3 4-bit** (`Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold`, ~164 GiB): the faster of the two (on the ring, 77.2 tok/s
  at one stream and 172.7 total at eight; [Performance at four Sparks](#performance-at-four-sparks)) and the one to
  start with. Its defaults at four Sparks, `MEMORY_RESERVE_GIB=6.5 CACHE_FIRST=1 PROMPT_CACHE_GIB=8`, keep all eight
  1M-token windows (16,392 pages, 8.39M tokens) and an 8.0 GiB prompt cache a rank.
- **Block FP8** (`zai-org/GLM-5.3-Flash`, ~306 GiB, engine patch 0107): the model as Z.AI publishes it, a little
  closer to the original (4-passage NLL 0.7425 against EXL3's 0.7487) and 31-52% slower in decode (52.2 tok/s at
  one stream, 119.0 total at eight), with prefill close to EXL3's (1.8-1.9K tok/s from 16K up). It fits only at four Sparks (~81 GiB of weights a rank); rank 0 downloads it into
  `DATA_DIR/model-fp8` (or set `MODEL_DIR`, and `NFS_MODEL_PATH` for the workers, to a copy you have), and
  its defaults at four Sparks, `MEMORY_RESERVE_GIB=6.5 CACHE_FIRST=1 PROMPT_CACHE_GIB=4`, leave three 1M-token windows
  (6,147 pages, 3.15M tokens) and a 3.9 GiB prompt cache a rank. Details in [Block FP8 weights](#block-fp8-weights-fp81-zais-own-release-four-sparks-only).

Both are measured on the ring only; on a switch neither is run yet. Switching between them is a restart with or
without `FP8=1` (`./start-tp4.sh restart` or `./start-tp4-switchless.sh restart`); the image is the same.

Memory at four Sparks, at the defaults (`MEMORY_RESERVE_GIB=6.5 CACHE_FIRST=1`, `PARALLEL=8`, `CONTEXT=auto`), measured
2026-11-11 on the ring with each rank's startup line:

| Weights | Reserve | Cache-first | `PROMPT_CACHE_GIB` | Weights a rank | KV pool | Prompt cache a rank | Least available on a Spark |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EXL3 4-bit | 6.5 GiB | on | 8 | 45.1 GiB | 16,392 pages, 8.39M tokens (eight 1M windows) | 8.0 GiB | 7.4 GiB |
| Block FP8 | 6.5 GiB | on | 4 | 80.2 GiB | 6,147 pages, 3.15M tokens (three 1M windows) | 3.9 GiB | 7.6 GiB |

The pool is shared: any stream takes up to one 1M window, and the eight streams' contexts together fit in the pool.
The lower reserve buys the prompt cache, not KV: FP8 at the earlier 9.5 had the same 6,147 pages and no prompt cache
(the pages left 0.08 GiB), at 8 a 2.4 GiB one; EXL3 at 12 had the same 16,392 pages and a 3.6 GiB cache. The least
available Spark (one also running other tools) is the one to watch; the memory guard stops all four ranks when any
Spark stays under 2.5 GiB for 4 s. A value in `local.sh` or the environment overrides these defaults.

### Performance at four Sparks

Measured 2026-11-11 on the ring (`./start-tp4-switchless.sh`, spark1 as rank 0, image `4835fc852d34`, 121 patches),
8 streams at a 1,048,576-token window, FP8 KV on pages, q8 dense and head, DFlash2; EXL3 at `MEMORY_RESERVE_GIB=12`,
FP8 at 9.5. Decode is prose at temperature 0.7 with thinking on and `top_k` 20, so these are served-traffic numbers,
not the greedy ones above.

Decode (tok/s; time to first token):

| Streams | EXL3 4-bit, total | EXL3 a stream | EXL3 TTFT | Block FP8, total | FP8 a stream | FP8 TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 77.2 | 77.2 | 128 ms | 52.2 | 52.2 | 204 ms |
| 2 | 98.0 | 49.6 | 246 ms | 69.5 | 35.7 | 367 ms |
| 3 | 116.3 | 40.1 | 309 ms | 88.6 | 30.6 | 466 ms |
| 4 | 134.3 | 35.5 | 382 ms | 88.6 | 23.6 | 563 ms |
| 8 | 172.7 | 23.0 | 686 ms | 119.0 | 15.8 | 1.08 s |

Prefill (one prompt; tok/s and time to first token):

| Prompt | EXL3 4-bit | EXL3 TTFT | Block FP8 | FP8 TTFT |
| ---: | ---: | ---: | ---: | ---: |
| 4K (~4,120 tokens) | 1,582.0 | 2.76 s | 1,534.9 | 2.92 s |
| 8K (~8,217) | 1,630.0 | 5.20 s | 1,806.6 | 4.78 s |
| 16K (~16,409) | 1,793.7 | 9.30 s | 1,833.1 | 9.19 s |
| 32K (~32,791) | 1,879.9 | 17.60 s | 1,843.4 | 18.03 s |
| 64K (~65,561) | 1,918.7 | 34.33 s | 1,849.3 | 35.69 s |
| 128K (~131,094) | 1,908.2 | 68.86 s | 1,839.2 | 71.52 s |
| 256K (~262,173) | 1,902.6 | 137.95 s | 1,802.0 | 145.72 s |

EXL3 decodes 31-52% faster than FP8 at every load (it reads half the expert bytes a round). Prefill is close: FP8 is
about 10% faster at 8K, and from 16K up the two are within 2-6% (EXL3 ahead from 32K), both holding 1.8-1.9K tok/s
through 256K.

### Guide: from the pair to four Sparks and back

1. **Cable the four Sparks.** On a switch: every Spark's ConnectX-7 ports to the switch, all on one subnet (use
   `./start-tp4.sh`). Without a switch, a ring: Spark A to B, B to C, C to D, D to A, each Spark's two ports to its two
   neighbours and each cable its own subnet (a port and its PCIe twin both carry an address on it); A and C, and B and
   D, have no cable between them and reach each other over the LAN (use `./start-tp4-switchless.sh`).
2. **Pick rank 0.** It serves the API and holds the weights the others mount. On a ring the workers go in its order:
   on this kit rank 0 is spark1 and ranks 1-3 are spark2 (cabled), spark3 (across) and spark4 (cabled), so
   `WORKER=spark2`, `WORKER2=spark3`, `WORKER3=spark4`. On a switch any order works.
   From rank 0, `ssh -o BatchMode=yes <worker> true` must succeed for all three (key-based login; the workers' user
   may differ from rank 0's).
3. **Stop everything else on the four Sparks.** Each of them serves one rank and needs its GPU and memory free, so
   stop any pair first (`./stop.sh` on each pair's rank 0) and anything else holding GPU memory. Step 1 of the start
   checks every Spark's memory, disk and ports and stops with a reason when one falls short.
4. **Weights.** Rank 0 needs the checkpoint (and DFlash2) locally, at its pinned revision. The workers either mount
   it over NFS from rank 0 (`WORKER_WEIGHTS=nfs`, the default here: rank 0 must export the folders, read-only, to the
   subnets the workers reach it on) or hold their own copy (`WORKER_WEIGHTS=check`). If the export is a different view of the path (a
   container exporting `/huggingface` for `~/.cache/huggingface`), name the exported paths in `NFS_MODEL_PATH` and
   `NFS_DFLASH2_PATH`. `FP8=1` needs about 306 GiB more on rank 0 (see below).
5. **Write rank 0's `scripts/local.sh`.** The workers (in ring order on a ring), plus the weights; everything else is
   the pair's settings. For a ring (your users, addresses and paths):
   ```sh
   WORKER=user@spark2           # cabled to this Spark
   WORKER2=user@spark3          # across the ring
   WORKER3=user@spark4          # cabled to this Spark
   WORKER_WEIGHTS=nfs
   NFS_MODEL_PATH=/exported/path/to/model        # only when the NFS export is another view of MODEL_DIR
   NFS_DFLASH2_PATH=/exported/path/to/dflash2
   PORT=8888
   PARALLEL=8
   CONTEXT=auto
   # MEMORY_RESERVE_GIB, CACHE_FIRST, PROMPT_CACHE_GIB: leave them unset (6.5, 1, and 8 for EXL3 or 4 for FP8=1)
   ```
   A two-Spark `local.sh` (one `WORKER`, no `WORKER2`/`WORKER3`) starts nothing here: the start stops and names them.
6. **Check the plan.** `DRY_RUN=1 ./start-tp4-switchless.sh` (a ring) or `DRY_RUN=1 ./start-tp4.sh` (a switch) finds
   the links, writes nothing and prints every rank's command. It stops with `E_FABRIC` when the cabling is the other
   launcher's (and names it), when the cables do not form one ring or the workers are not in its order, or when a
   Spark on a switch shares no subnet with the others; and with `E_CONFIG` for a setting this mode does not take.
7. **Start.** The same command without `DRY_RUN`: EXL3 4-bit by default, or with `FP8=1`. It builds the image on rank 0
   when its patches changed (about 6 minutes) and sends it to the three workers, mounts the weights, starts ranks 3,
   2 and 1, then rank 0, and waits for a greedy smoke reply. A first EXL3 load over NFS takes a few minutes; FP8 about
   4 (245 s measured with the loader's read-ahead at its default; `TF_GLM_LOAD_PREFETCH=0` made it 12 minutes).
8. **Check the start.** Rank 0's startup line in `logs/current/rank0.log`: the streams, the KV pages, the prompt
   cache. The ranks' logs are `logs/current/rank0..3.log`; `curl http://127.0.0.1:8888/health` answers on rank 0.
9. **Back to the pair.** `./stop-tp4.sh` stops all four ranks (rank 0 first) and archives their logs; then restore
   the pair's `local.sh` and `./start.sh` on each pair's rank 0.

Settings at four Sparks:

| Setting | At TP=4 |
| --- | --- |
| `ROCE_RING` | Ring only (refused on a switch). `off`; `on` sends the decode and verify all-gathers as RDMA writes around the ring (patch 0108): +1-2% at one stream and +5-7% at four on FP8, +2-3% on EXL3, greedy replies and NLL unchanged. Opt-in |
| `FP8` | `1` serves Z.AI's block FP8 release (patch 0107); four Sparks only |
| `LEARN_GIB` | Conversations on disk (patch 0113): rank 0's `CACHE_DIR/learn` holds every rank's part |
| `MEMORY_RESERVE_GIB`, `CACHE_FIRST`, `PROMPT_CACHE_GIB` | Four Sparks: `6.5`, `1`, and `4` with `FP8=1` or `8` with EXL3 (the pair: `6.5`, `1`, `5`). `FP8=1`, measured 2026-11-11: 6,147 pages (3.15M tokens) and a 3.9 GiB prompt cache a rank, with 7.6-10.5 GiB still available on each Spark; at 9.5 the same start had no prompt cache (the pages left 0.08 GiB), at 8, 2.4 GiB. EXL3 at 6.5: the same 16,392 pages it held at 12 (all eight 1M windows) and an 8.0 GiB prompt cache instead of 3.6. See the memory table under [Weights](#4-sparks-tp4-experimental). A value set in `local.sh` or the environment wins |
| `L2PF` | `1` prefetches the next weights into L2 during decode (patch 0117); unmeasured at four, opt-in |
| `EXTRA_EOS` | `<\|assistant\|>` by default (patch 0120), as on the pair |
| `THINK_OFF_SPLIT` | Opt-in (patch 0121), as on the pair |
| `EFFORT_LINE`, `CHAT_TEMPLATE` | As on the pair; the written template goes to every worker |
| `KV_SHARD` | Refused: patch 0114 shards over exactly two ranks |
| `PROMPT_CACHE_MIN_GIB` | Not checked here (a warning says so); read rank 0's prompt cache line instead |
| `PREFILL_REDUCE` | `gather`; `ring` does not run on the ring fabric (see the list below), and neither is measured on a switch |

Things that break a TP=4 start, and why:

- **`NCCL_SOCKET_IFNAME` exported to every rank.** A follower given one makes this engine unset `NCCL_COMM_ID` before
  NCCL starts, and it panics ("attempt to use null value"). `scripts/tp4.sh` gives it to rank 0 only; do not export
  it yourself. `SOCKET_IFNAME` is the setting for the bootstrap netdev.
- **`NCCL_RUNTIME_CONNECT=0` on a ring.** NCCL's tree for four ranks joins rank 0 to rank 2, which no cable reaches; connecting
  every channel at init then times out in `ibv_modify_qp`. The launcher sets `NCCL_RUNTIME_CONNECT=1` and
  `NCCL_CUMEM_ENABLE=1` itself.
- **Workers out of ring order (a ring only).** `E_FABRIC` names the order it found; swap `WORKER` and `WORKER3`, or fix `WORKER2`.
- **A worker still serving a pair rank.** Its GPU and memory are taken; stop that pair first (`./stop.sh` on its rank 0).

What differs from the pair (`scripts/tp4.sh`):

- **The ring.** Each Spark's RoCE addresses are read over ssh and paired by subnet; the launcher refuses a switch, a
  ring that is not in `WORKER` order, or a missing cable. NCCL gets `NCCL_ALGO=Ring` and a graph file per rank whose
  channels sit on the devices toward the previous rank, with subnet-aware routing (`NCCL_IB_SUBNET_AWARE_ROUTING=1`,
  `NCCL_IB_SUBNET_PREFIX_LEN` set so a port and its twin both carry traffic) moving each send onto the next rank's
  cable; NIC fusion, P2P and SHM transports are off. NCCL connects channels at first use
  (`NCCL_RUNTIME_CONNECT=1`): its tree for four ranks joins rank 0 to rank 2, which no cable reaches, so connecting
  everything at init, as the engine asks by default, times out in `ibv_modify_qp`. The engine's own exchanges run as
  all-gathers between neighbours (`PREFILL_REDUCE=gather`; `ring` passes the split prefill's rows along the ring and
  is the next thing to measure); the two-rank RoCE path does not apply and `ROCE` is off. Engine patch 0106 makes the
  start's peer warm-up follow that mode instead of posting to every rank.
- **Bootstrap.** `NCCL_COMM_ID` and the ranks' watch lines go to `MASTER_ADDR` (what this Spark's name resolves to; an
  address every worker reaches, through a neighbour for the Spark across); NCCL's bootstrap socket on each Spark's
  default-route netdev (`SOCKET_IFNAME`).
- **Weights.** Rank 0 reads its own `MODEL_DIR`; the workers mount it over NFS from this Spark's address on each
  worker's route (`WORKER_WEIGHTS=nfs`, the default here; `check` for workers with their own copy). When the NFS
  server exports another view of the path (a container exporting `/huggingface` for `~/.cache/huggingface`), set
  `NFS_MODEL_PATH` and `NFS_DFLASH2_PATH` to the exported paths; `NFS_SERVER1..3` override the per-worker address.
  Every rank still reads every shard: three workers over NFS makes the first load slower than the pair's.
- **Memory.** Each rank holds a quarter of the routed experts, the KV pool is replicated on every rank, so the page
  planner has far more room: eight 1M-token windows and a prompt cache usually fit at the recipe's default reserve.
  Read rank 0's startup line after every start (the pages, the prompt cache, no "cannot keep one state").

An earlier measurement, greedy with thinking off (the current served-traffic numbers are in [Performance at four Sparks](#performance-at-four-sparks)), 2026-11-04 on this kit (spark1 rank 0, image `74e19a63a9a5`, 8 streams, 1M window, FP8 KV on pages, q8
dense and head, DFlash2; greedy, the same harness as the pair's numbers under [Performance](#performance)):

| | two Sparks (TP=2) | four Sparks (TP=4, a ring) |
| --- | ---: | ---: |
| 1 stream, prose / code | 38.4 / 56.6 tok/s | 56.1 / 85.0 tok/s |
| 4 streams, prose, total | 79-81 tok/s | 115-120 tok/s |
| 8 streams, prose, total | 105-106 tok/s | 163-166 tok/s |
| prefill 16K / 64K | ~1,600-1,800 tok/s | 1,752 / 1,889 tok/s |
| KV pool | 4,098 pages (2.1M tokens) | 16,392 pages (8.4M tokens), 3.6 GiB prompt cache |

Rank 0 loads 45.6 GiB; each rank's NCCL connects only to its two neighbours. Greedy replies are repeatable run to
run and the same with drafts off; they are not the pair's tokens (a different partition rounds differently).

Settings beside the pair's: `WORKER2`, `WORKER3`, `MASTER_ADDR`, `SOCKET_IFNAME`, `PREFILL_REDUCE` (`gather`; `ring`
stops at the ranks' first point-to-point send on this fabric, where NCCL picks a port without the peer),
`RING_NCCL_ALGO` (`Ring`), `ROCE_RING` (`off`; `on` carries the decode and verify all-gathers up to `ROCE_MAX_KB` as
one-shot RDMA writes to both neighbours, each relaying half a shard to the rank across: engine patch 0108, a candidate
until its gates in docs/speed/RESULTS.md are met), `NFS_MODEL_PATH`, `NFS_DFLASH2_PATH`, `NFS_SERVER1..3`. Logs:
`logs/current/rank0..3.log`.

### Block FP8 weights (`FP8=1`: Z.AI's own release), four Sparks only

`FP8=1` with either four-Spark launcher serves `zai-org/GLM-5.3-Flash` @ `46a050be`, the model as Z.AI publishes it: block FP8, e4m3
codes with 128 x 128 scales, ~306 GiB (`unsloth/GLM-5.3-Flash-FP8` is a byte-identical mirror; both pins are in
`scripts/config.sh`). Engine patch `0107` reads it: the routed experts on the engine's block-FP8 expert kernels, the
dense MLPs, shared experts and the DSA layers' projections on its block-FP8 lane, the rest bf16 as stored; the scale
tensor may be `weight_scale` or `weight_scale_inv`; another block size, a per-tensor FP8 or another code format is
refused by name. It downloads into `DATA_DIR/model-fp8` when missing (or set `MODEL_DIR` and `NFS_MODEL_PATH` to a copy
you have); ~81 GiB of weights a rank, so it fits only at TP=4 and leaves three 1M-token windows (3.1M tokens of KV)
and, at its defaults (`MEMORY_RESERVE_GIB=6.5 CACHE_FIRST=1 PROMPT_CACHE_GIB=4`), a 3.9 GiB prompt cache a rank (none at
the earlier 9.5). An earlier measurement on this kit (greedy, same harness as the pair's; the current numbers are in [Performance at four Sparks](#performance-at-four-sparks)), image
`cb33a3d5fc68`):

| | EXL3 4-bit, TP=4 | FP8, TP=4 |
| --- | ---: | ---: |
| 1 stream, prose / code | 56.1 / 85.0 tok/s | 44.7 / 65.2 tok/s |
| 4 streams, prose, total | 115-120 tok/s | 83-87 tok/s |
| 8 streams, prose, total | 163-166 tok/s | 112-114 tok/s |
| prefill 16K / 64K | 1,752 / 1,889 tok/s | 1,746 / 1,787 tok/s |
| 4-passage NLL (`docs/quality.py`) | 0.7487 | 0.7425 |

The expert kernels compute bf16 activations against widened e4m3 weights with fp32 accumulation and the block scale
(no activation quantizer); a decode round reads twice the expert bytes of the 4-bit pack, which is the whole speed
difference. Greedy replies are repeatable and the same with drafts off; nine of the ten `quality.py` answers equal the
EXL3 pack's.

## Logs and stopping

`logs/current` holds `rank0.log` and `rank1.log`, collected from each Spark's Docker output. `./stop.sh` stops rank 0
first (it drains for `TF_DRAIN_S`, 20 s), then rank 1, archives both logs as `logs/glm53-spark-<time>.tar.gz`, and
only ever stops containers labelled `tf.recipe=glm53-spark`.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `[E_WORKER]` | ssh, Docker or the NVIDIA runtime is not ready on the worker; the message names which |
| A reply ends with `<\|assistant\|>` followed by the same answer again, `finish_reason: length` | The checkpoint's EOS list leaves out `<\|assistant\|>`. Keep `EXTRA_EOS` at its default (`<\|assistant\|>`) |
| With thinking off, `content` starts with a draft ("We need ...") and holds `</think>` | The model reasoned anyway. Set `THINK_OFF_SPLIT=1024` (off by default) so the draft goes to `reasoning_content`; raise it if the draft is longer |
| A thinking-off reply works the sum out in its answer and `finish_reason` is `length` on a short `max_tokens` | The prompt carries the template's `Reasoning Effort: Max` line (`EFFORT_LINE=always`, or a `CHAT_TEMPLATE` of your own); `EFFORT_LINE=thinking` (default) writes it with thinking on only. `/tokenize` on a one-line user message shows it: 7 tokens more |
| `[E_FABRIC]` | No directly reached address, or no RoCE v2 GID holding an IPv4 on the link's port: cable the Sparks on one subnet, or set `FABRIC_PEER` / `NCCL_IB_GID_INDEX` |
| `[W_FABRIC]` a rail left out | The second rail does not carry a full-size ping; check the cable, the twin's address and MTU |
| `[E_MEMORY]`, `[E_PORT]` | Less than `MIN_AVAIL_GIB` available (another rank, usually), or a port in use |
| `RankMismatch`, `RankNeverCame`, NCCL timeouts | The ranks were started differently, or rank 1 did not reach rank 0 within 300 s: check `rank1.log` and the ports; `NCCL_DEBUG=INFO` for detail |
| `vendor_err 0x81` | A rail without a path to its partner was set by hand through `TF_ROCE_HCA` |
| A long prompt dies in a RoCE gather | The gather has no wait setting at this pin; `ROCE=off` puts every exchange on NCCL |
| `--parallel N needs … GiB`, `[memguard]` | The window or stream count does not fit, or a Spark went under the floor: lower `CONTEXT`/`PARALLEL` or raise `MEMORY_RESERVE_GIB` |
| No prompt cache in the startup line, `cannot keep one state` | The page pool took a third window: raise `MEMORY_RESERVE_GIB` (the engine's own hint to lower it is the wrong direction here; see [Memory](#memory-on-a-128-gb-unified-spark)) |
| A prompt cache far under `PROMPT_CACHE_GIB`; `prompt cache: the room funds X GiB of the Y GiB` | The cap never binds: the cache is the remainder after the pool and the streams' fixed parts. Set `CACHE_FIRST=1`, or lower `PARALLEL` or `CONTEXT` (see [Memory](#memory-on-a-128-gb-unified-spark)). With `CACHE_FIRST=1` already on, the pool kept a whole window more than the cache left room for: raise `PROMPT_CACHE_GIB` until the page count drops by a window (issue 16) |
| `[E_CONFIG] NCCL_SOCKET_IFNAME must be a netdev name.`, or `[W_CONFIG] NCCL_SOCKET_IFNAME=^… is an NCCL exclusion list` | An exported value wins over `scripts/local.sh`. An exclusion list such as `^lo,docker` from `/etc/profile.d` is set aside with that warning and the link is found as when empty (issue 17); any other value must be one netdev: `NCCL_SOCKET_IFNAME=<netdev> ./start.sh`, or `env -u NCCL_SOCKET_IFNAME ./start.sh` |
| `git pull` will not fast-forward a clone from before 2026-11-04 | The history was rewritten that day, so a clone from then (such as `0049d7c`) shares no commit with `main`: take a fresh clone and carry `scripts/local.sh` and `.env` over (issue 16) |
| `[E_CACHE] Rank N funds a X GiB prompt cache, under PROMPT_CACHE_MIN_GIB` | That start's room funded less cache than the minimum (the room varies by a GiB or more between starts). Lower `PARALLEL` or `CONTEXT`; with `CACHE_FIRST=1`, a larger `PROMPT_CACHE_GIB` can fund more, because it can cost the streams a whole window (issue 16); or lower the minimum |
| An exact value (an id, a hash) copied wrong in a JSON reply on `ABLIT=1` | The Ablit weights' own error (issue 9): the published weights copy it right on the same engine and request, and the Ablit weights do too with thinking on. See [Ablit weights](#ablit-weights) |
| `libtfvideo.so is required for video inputs` (HTTP 500 on a `video_url` part) | The default image builds no video runtime. Set `WITH_VIDEO=1` and `./start.sh restart` (it builds its own image tag); a server that already refused one video request keeps refusing until restarted |
| Raw `<tool_call>name<arg_key>…</arg_key><arg_value>…</arg_value></tool_call>` text in an agent's reply (Hermes and others) | The client sent that turn without the tool the model called: no `tools`, `tool_choice: "none"` (an agent's final-summary turn after its iteration limit, typically), or a list without that tool. The server only turns calls to offered tools into `tool_calls`. Fix the client (raise its iteration limit, enable the toolset), or turn on the guard: [Hermes and other agents](#hermes-and-other-agents-the-tool-call-guard). If the tool was offered, the model wrote a malformed call: `docker logs glm53-spark 2>&1 \| grep "tool call rejected"` gives the request id and reason, and `TF_TOOL_CALL_DEBUG=1` the text |
| The first start takes long | Each rank hashes all 164 GiB once (cached in `CACHE_DIR`); raise `WAIT_TIMEOUT` |

## Ablit weights

`ABLIT=1` serves
[Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit),
the same checkpoint abliterated, at the revision pinned in `scripts/config.sh`. The repository is gated: `start.sh`
checks that `HF_TOKEN` reaches it before building or downloading. Thinking off, the recipe's default, is also how
these weights give their best results. One measured exception (issue 9): asked to return a record id
`LANTERN-ca6241dd2684` in schema-constrained JSON, the Ablit weights drop a digit, with drafting on or off; the
published weights copy it exactly on the same engine and request, and the Ablit weights do too with thinking on. For
replies that must copy ids, hashes or numbers exactly, serve `ABLIT=0` or turn thinking on for those requests.
DFlash2 accepts fewer drafts on these weights; measure them at your own memory settings before choosing a drafter.
One pair (issue 17) saw drafting turn into a net loss at `local.sh.example`'s `PARALLEL=4 MEMORY_RESERVE_GIB=12`, and
about 15% below the published weights (74.6% against 79.8% accepted) at the served `PARALLEL=8` with `CACHE_FIRST=1`.

### A partial overlay

`ABLIT=1` serves the whole transplant. `scripts/graft.py` builds a tree of your own: it copies `self_attn.o_proj` for
the layers you name out of the Ablit repository into a copy of the published weights, byte for byte and at the offsets
each safetensors header gives, so neither tree is loaded, no framework is needed and nothing but the grafted shards is
copied (the rest are hardlinks to the parent).

```
scripts/graft.py ~/.cache/aevonix-glm53/model ~/.cache/aevonix-glm53/model-ablit \
    ~/.cache/aevonix-glm53/model-partial1529 --layers 15-29 --verify
# then: MODEL_SOURCE=local MODEL_DIR=~/.cache/aevonix-glm53/model-partial1529
```

The Ablit repository replaces `self_attn.o_proj` on layers 15-43 and the MTP head (`ABLIT_META.json`), so a subset is a
subset of that transplant rather than a new one; the graft rebuilds a 164 GiB tree in about 30 s. Measured on the pair:
15 of 30 layers (15-29) passed all 32 prompts of the refusal suite drowzeys publishes for its Keys overlay, kept issue
9's exact-copy case (the `LANTERN-ca6241dd2684` reply byte-exact), and decoded at the parent's rate (56.5 / 69.6 tok/s
code against 59.7 / 62.4, inside this pair's 12% run-to-run spread), where the full 30-layer overlay measured about 15%
below the parent on code and reproduces the dropped digit. Layer choice is per graft: rebuild the tree to change it.

## Credits

TensorFold and its native engine are by Ash Hart and contributors
([ashhart/TensorFold](https://github.com/ashhart/TensorFold)). The GLM-5.3-Flash EXL3 checkpoints are by Mia's AI Lab,
from zai-org's GLM-5.3-Flash. DFlash2 is by incoai. The two-host launch, the link discovery and the unified-memory
accounting are adapted from Mia's AI Lab's
[GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)
(branch `zig-engine`), and the engine's RoCE transport is the GPU side of b12x's "RoCEnante" exchange
([local-inference-lab/b12x](https://github.com/local-inference-lab/b12x), Apache 2.0).

Patches 0103-0104 are ported from the lab's own [2x RTX PRO 6000 recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-2x-RTX-PRO-6000-TensorFold).
Full credits, including the authors of the carried upstream patches (Adam Hillson, ajensenwaud, Bizuayeu,
SvangenStudios, with their names in each patch's `From:` line), are in [CREDITS.md](CREDITS.md). Patch 0113 is a backport of upstream #1411, and 0114 (KV sharding) is toward upstream #1431, both by Robert Wojciechowski ([@wojo](https://github.com/wojo), X [@rwojo](https://x.com/rwojo)). Required notices are
in [NOTICE.md](NOTICE.md).

## License

Apache License 2.0: see [LICENSE](LICENSE). Downloaded software and weights keep their own terms; DFlash2's weights are
CC BY-NC-ND 4.0.
