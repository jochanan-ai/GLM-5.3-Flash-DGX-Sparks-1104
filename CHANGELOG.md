# Changelog

- **2026-11-13** -- Running under a supervisor (systemd's `Type=simple`, llama-swap): `scripts/supervise.sh`, sourced by
  `./start.sh` and `./start-tp4-switchless.sh`. `--allowed-host NAME` (repeatable) passes the names a proxy reaches the
  API under to every rank's `serve` flags, for TensorFold's loopback `Host` check; any other argument is refused as
  before. `PREPARE=0` makes a start check instead of build, download or copy (the image here and on each worker, the
  checkpoint, DFlash2; `rsync` checked as `check`), stopping with `E_PREPARE` before any rank runs. `FOREGROUND=1`
  keeps the start attached to rank 0 after the smoke test (start.lock let go), stops every rank on SIGINT/SIGTERM,
  on a failure after the launch and when rank 0 ends, and exits 0 for a stop that was asked for, 1 for the memory
  guard's stop (its subshell now exits 3 after stopping the server), else rank 0's code; a stop at work is waited
  for up to `FOREGROUND_LOCK_WAIT_S` (120 s) and a Docker daemon that does not answer for a while ends nothing. `FABRIC_EXPECT=ring|mesh` in
  the environment sends `./start.sh` to `./start-tp4-switchless.sh` and `./stop.sh` to `./stop-tp4.sh`. Without these
  a start's plan and behaviour are unchanged (`DRY_RUN=1` output identical). `tests/test_supervise.py`: offline tests
  with fake `docker` and `ssh`.

- **2026-11-11** -- `MEMORY_RESERVE_GIB=6.5` and `CACHE_FIRST=1` are the defaults everywhere, the pair included (the
  reserve was 12, and the pair ran 9.5 in its `local.sh`; cache-first was off), with `PROMPT_CACHE_GIB` 5 on the pair
  (8 and 4 at four Sparks, as below). Cache-first is part of it: at 6.5 and windows first the pair's page pool would
  cross the 19.5 GiB threshold for a third 1M window and leave no prompt cache (RESULTS.md 2026-11-04). The pair's
  5 GiB beside its two windows is predicted, not yet measured. `scripts/local.sh.example` and the README follow.

- **2026-11-11** -- Four Sparks with EXL3 take the same defaults, `MEMORY_RESERVE_GIB=6.5 CACHE_FIRST=1`, with
  `PROMPT_CACHE_GIB=8`: at 12 they already held all eight 1M windows (16,392 pages) and a 3.6 GiB prompt cache, and the
  5.5 GiB the lower reserve frees goes to the cache: measured on the ring, the same 16,392 pages (8.39M tokens) and an
  8.0 GiB prompt cache a rank, 7.4-10.4 GiB still available on each Spark. The pair keeps 12 and no cache-first.
  README: a memory table for both weights under "4 Sparks".

- **2026-11-11** -- Four Sparks with `FP8=1` default to `MEMORY_RESERVE_GIB=6.5 CACHE_FIRST=1 PROMPT_CACHE_GIB=4`
  (`scripts/config.sh`; EXL3 and the pair keep 12 and no cache-first). At the earlier 9.5 the FP8 start's KV pages left
  0.08 GiB, so it served with no prompt cache; at 6.5 the ring measured the same 6,147 pages (3.15M tokens) and a
  3.9 GiB prompt cache a rank, with 7.6-10.5 GiB still available on each Spark. A value in `local.sh` or the
  environment still wins.

- **2026-11-11** -- Four Sparks (TP=4) join main, with `./start-tp4.sh` (a switch) and `./start-tp4-switchless.sh` (a
  ring of direct cables), `./stop-tp4.sh`, `scripts/tp4.sh`, the README's "4 Sparks" section with a step-by-step guide,
  and their measurements under `docs/speed/tp4`. **The engine patches are renumbered for everyone, the pair too:** the
  four TP=4 patches take 0107-0110 (block FP8 checkpoints, the four-rank RoCE ring all-gather, the DFlash2 drafter MLP
  split across every rank, `TF_GLM_CHUNK_SOLO`), so main's former 0107-0117 are now 0111-0121 (0107 tool-call guard ->
  0111, 0108 `CACHE_FIRST` -> 0112, 0109 learned conversations -> 0113, 0110 KV sharding -> 0114, 0111 -> 0115, 0112
  -> 0116, 0113 `L2PF` -> 0117, 0114 -> 0118, 0115 32-row windows -> 0119, 0116 `EXTRA_EOS` -> 0120, 0117
  `THINK_OFF_SPLIT` -> 0121); entries below this one use the old numbers. None of the four changes the pair's
  serving: FP8 reads only a block-FP8 checkpoint, the ring all-gather runs only with `ROCE_RING=on` at four ranks,
  `TF_GLM_CHUNK_SOLO` is off unless set, and the drafter MLP split at two ranks is the same halves and the same single
  fp32 add as before (same bits). The rank fingerprint's layout is 17. The pair's next start rebuilds its image (121
  patches) under a new label. `start.sh` still runs two Sparks; `KV_SHARD=1` stays two-rank only (the TP=4 starts
  refuse it).

- **2026-11-11** -- Two four-Spark launchers. The ring's is now `./start-tp4-switchless.sh` (renamed from
  `./start-tp4.sh`, otherwise as before); `./start-tp4.sh` is new, for four Sparks on a switch, adapted from the
  earlier Python-engine recipe's switch launcher (MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold, branch tp4):
  it runs the same script with `FABRIC_EXPECT=mesh`. On a switch `scripts/tp4.sh` requires every pair of Sparks to
  share a RoCE subnet, takes the workers in any order, writes no ring graphs and gives NCCL its own algorithms and
  devices (no `NCCL_ALGO=Ring`, no subnet-aware routing or prefix), and refuses `ROCE_RING=on` (the ring's relay;
  this engine has no other four-rank RoCE path). Each launcher stops on the other's cabling and names it. README:
  the "4 Sparks" section and its guide cover both. The switch launcher is not run yet.

- **2026-11-11** -- tp4-local takes main up to 74cf93c (the entries below, in main's numbering) on its own stack. Main's
  0107 and 0108 are this stack's 0111 and 0112, as before; main's 0109-0117 are this stack's 0113-0121, each regenerated
  against 0001-0112 here (3-way; conflicts only where they meet 0110's CHUNK_SOLO and 0108's ring kernels). The rank
  fingerprint's layout is 17 here: CHUNK_SOLO and KV_SHARD each moved it to 15 on their own branches with different
  words, so the combined struct takes 16, and L2PF 17. The 121 patches apply to `f350cd3e` with the Dockerfile's loop.
  `./start-tp4.sh` gains what the pair's start has: `LEARN_GIB` (`--learn-conversations`, 0113), `L2PF` (0117),
  `EXTRA_EOS` (0120, on with `<|assistant|>`), `THINK_OFF_SPLIT` (0121, opt-in), `EFFORT_LINE=thinking`'s template
  (written after the first download with `VISION=0`), `|| true` on the post-LIVE pool line and the one-stream
  `W_STREAMS` match. It refuses `KV_SHARD=1` (0114 runs at TP=2 only) and says that `PROMPT_CACHE_MIN_GIB` is not
  checked at four. Not yet run on the four Sparks.

- **2026-11-11** -- three contributed changes, ported from their PRs (authorship kept):
  - Engine patch 0116, `EXTRA_EOS` (default `<|assistant|>`, `TF_EXTRA_EOS`): tokens that end a reply besides the
    checkpoint's EOS ids. GLM-5.3's `generation_config.json` leaves out `<|assistant|>`, which the model can write to
    close a reply; it then reached the reply as text and the model wrote its whole reply again until the length cap
    (#36, thanks @wagzee; submitted as 0112, renumbered after main's 0112-0115). `EXTRA_EOS=none` keeps the
    checkpoint's list.
  - Engine patch 0117, `THINK_OFF_SPLIT` (`TF_THINK_OFF_SPLIT`), **opt-in**: a thinking-off chat reply that writes
    `</think>` within the hold has its draft as `reasoning_content` and its answer as `content`. It ships off
    (`THINK_OFF_SPLIT=0`, the engine's own default when unset) because a hold delays every streamed thinking-off
    reply's first content by up to that many tokens; 1024 is the hold its author measured (#37, thanks @wagzee;
    submitted on by default as 0113).
  - `validate_config`: `KV_SHARD=1` also needs `KV_PAGES=on` and `PARALLEL` other than 1, which patch 0110 refuses
    after `./start.sh restart` has stopped the serving pair; they now stop before anything runs, with
    `scripts/tests/kv-shard-config.sh` (#29, thanks @travellingsoldier85).
  The 117 patches apply to `f350cd3e`; the image builds with 0116 and 0117. Not measured on the pair.

- **2026-11-11** -- contributed script fixes, ported from their PRs (authorship kept):
  - `PROMPT_CACHE_MIN_GIB`: a `PARALLEL=1` start's "1 stream at once" line is read instead of stopping both ranks
    (issue #31), and `E_CACHE` names the log archive `stop.sh` wrote rather than the run directory it removed (#25,
    thanks @laidick; `scripts/tests/cache-min.sh` gains the one-stream case).
  - `start.sh` after a `CONTEXT=auto` start: KV windows (no pages line) no longer end the start with exit 1 under
    pipefail, and `W_STREAMS` stays quiet for `PARALLEL=auto` and matches "1 stream at once" (#28, thanks
    @travellingsoldier85).
  - `DRY_RUN=1` prints the whole plan with a placeholder for rank 1's `--user` when the worker is out of reach; a real
    start still stops with `E_WORKER` (#30, thanks @travellingsoldier85).
  - `VISION=0` with `EFFORT_LINE=thinking` on a checkout that has not downloaded the checkpoint writes the template
    after step 3's download instead of stopping before step 1 (#27, thanks @travellingsoldier85).
  - `scripts/graft.py`, a partial Ablit overlay: the layers you name copied byte for byte from the Ablit tree into a
    hardlinked copy of the published weights, with `scripts/tests/graft.sh` (#26, thanks @mjjjjaazing). On top: an
    OUT that resolves to PARENT or DONOR through a symlink is refused before anything is unlinked (before, the
    parent lost its shard), with a test that fails without it.

- **2026-11-11** -- engine patch 0115 (32-row decode windows, from upstream #2257) and the pair's measurements of
  0113-0115. 0115 raises the verify window from 16 to 32 rows, so a context copy verifies up to 31 tokens a round
  (DFlash2 still drafts 7), with the fixes the wider window exposes here: shared-window scratch is said outright (at
  32 and 32 the KDA projection buffer would have shrunk to 4 bytes and shared rounds lost 0075/0088's one-launch
  segment attention), the follower's draws hold 32 rows, and the window width joins the learned-state store's
  identity. Not taken from upstream: load-timed pricing (0096's GB10 prices stay), 15-deep MTP chains, and #1779's
  shared draft reservation, which upstream itself turned off for DFlash2 (cc2fa8d23, #2720: a cut chance left a
  stream at no drafts while its peers decoded). Measured on spark3+4 (`docs/speed/ab-upstream.sh`,
  `ab-upstream2.sh`; `docs/dec.py` 3 trials, greedy, thinking off; new `docs/speed/copybench.py`, a 1,195-token
  file returned with one rename), every leg's contract and copy token hashes identical to base:

  | leg | 1 stream | 4 streams | 8 streams | file rewrite, 1 stream | file rewrite, 4 streams |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | base (114 patches, L2PF=0) | 37.6 | 72.7 | 121.1 | | |
  | L2PF=1 | 38.4 | 73.1 | 121.4 | 151 (14.9 tok/round) | 34.2 a stream (4.0) |
  | 0115, L2PF=1 | 38.5 | 73.2 | 121.4 | 203 (26.6 tok/round) | 34.4 a stream (4.0) |
  | 0115, L2PF=0 | 37.5 | 72.8 | 121.6 | 201 (26.6 tok/round) | 34.0 a stream (4.0) |

  Decode tok/s (prompt time removed). 0115: +34% on a one-stream rewrite, no change elsewhere, 34 MiB more fixed
  memory a stream (451 against 417 MiB). L2PF: +2% at one stream (round 55.1 to 53.9 ms), flat under load; it stays
  opt-in. Shared rounds take almost no copies (4 tokens a round at four streams, before and after): the next copy gain
  is there, not in the window width. 0113-0115 build clean on the pair (images 6a5156b33a41 and 1d9dcf33f2e0).

- **2026-11-10** -- engine patches 0113 and 0114, ported from upstream TensorFold main (22fb601) onto `f350cd3e` and
  this recipe's 0001-0112, with `TF_REF` unchanged. Upstream had 2,662 commits since the pin; a trial rebase of the
  112 patches onto main conflicted in about 90, so only two GLM wins were picked, after a review that dropped the
  router-on-shared-mma-tiles refactor (it loses 0083's PDL trigger), `TF_GLM_KV_LIVE_HALF` (unsafe beside 0109's disk
  images and 0110's KV sharding), the two-heads-a-block prompt attention (not GLM's prompt path) and the bounded peer
  warm-up (0106 covers it; the port would pull in a long unrelated chain). 0113: `L2PF=1` (`TF_GLM_L2PF`, opt-in)
  prefetches each decode window's next weights into L2 on a side stream; upstream runs it by default on GB10 at TP>=2,
  kept off here until it is measured on this stack. 0114: a JSON tool call `max_tokens` cuts after naming an offered
  tool is returned as the reply's cut call, as XML and GLM calls were, and a whole `/v1/messages` reply closes its
  input instead of answering 500; two local fixes over upstream (a key with no value no longer indexes an empty
  slice; the name matches offered tools in any case, as a closed call's does). The 114 patches apply to `f350cd3e` with the Dockerfile's exact
  `git apply --check && git apply` loop. Built and measured on the pair 2026-11-11 (the entry above).

- **2026-11-09** -- engine patch 0112: rejected tool calls are logged one by one, and `TF_TOOL_CALL_GUARD=1` no longer
  empties replies. A Hermes Agent report (2026-11-06, guard on) showed 25 replies of 436 with calls hidden on requests
  that offered tools, 11 of them left empty, and Hermes retrying each empty reply unchanged until its 39 s timeout;
  the log said only `kept N malformed or unoffered tool envelopes as content`, with no request id, name or reason.
  Now each envelope the parser rejects gets `tool call rejected from ID: REASON ['NAME'] (N tools offered)
  thinking= effort=; FATE` (reasons `unknown_tool_name`, `malformed`, and `unclosed` for a call the end token cut
  before its closer), and `TF_TOOL_CALL_DEBUG=1` adds the envelope's text. With tools offered the guard hides only
  well-formed calls to unoffered tools: malformed calls stay as text, and a reply that hiding would leave with no text
  and no call keeps its calls. The no-tools guard path is unchanged; guard off, replies are byte-identical apart from
  the log. The report's "nameless calls passed through" (Hermes' `tool_call calls[0] requires a 'name'`) is Hermes'
  own `tool_call` bridge tool refusing the model's arguments, not a server leak (host test added). Host tests: 64
  `test-tool-parse-host`, 19 `test-tool-visibility-http-host`, 956 across `test-server-cpu` and the tool, JSON-answer
  and recovery steps, all passing; the 112 patches apply to `f350cd3e`. Not yet measured on the pair: a rebuilt image
  and a day of live agent traffic. Response to the report: `docs/issue-2026-11-06-tool-call-guard-silent.md`.
- **2026-11-08** -- engine patch 0111: the image build's nvcc prints no compiler warning. A clean-machine build showed
  49 warning lines that the builds in docs/speed's logs had carried for days: two fp8 expert prompt kernels declared a
  kernel-local `RW` their bodies never read, `prefill_attention.cu`'s `SMEM_MAX` had no reader anywhere,
  `glm_mm.cu`'s `Tile` kept an `SMEM` member no instantiation uses, and `flashnext_fact.cu` / `flashnext_attn.cu`
  held `#pragma unroll` loops behind an `if constexpr` branch whose taken side made them unreachable (nvcc #128-D,
  once per such kernel). The declarations are gone; the two loops sit behind an inverted `if constexpr` (the
  `flashnext_fact` wrap covers the rest of its iteration's scales, rows and epilogue), so each instantiation compiles
  only the side it can reach. No kernel's arithmetic, launch shape or output changes: all six files' fatbins compile
  byte-identical to the unpatched build (nvcc 13.3, sm_121), and every fatbin of the recipe's build now compiles
  warning-free.
- **2026-11-07** -- The pair serves `SERVE_EXTRA_ARGS='--top-k 20'` (in its `scripts/local.sh`): a request that
  sends no `top_k` was drawn from the whole top-p 0.95 nucleus of `generation_config.json` (temperature 1.0, top_p
  0.95, no top_k: `serve`'s `--top-k` is unset by default), where the old deployment used 20. Raised by raw, malformed
  tool-call markup in an agent turn (`<arg_key>` straight after the tool name) on the 2x RTX PRO 6000 bundle with the
  Ablit weights; a seed replay there was said to show a rank-252 draw. **Not reproduced on this pair, so the fix is a
  mitigation and the cause is unproven**: 520 sampled agent-shaped requests (text, then a tool call; 400 thinking off,
  120 thinking on) came back 520 correct with no `top_k`, and 520 with `top_k 20` likewise; `logprobs` are refused
  by this backend, so a draw's rank cannot be read here. Copying a tool-call markup line 30 times broke 30% of lines
  greedily and 33-36% sampled, with `top_k 20` or `min_p 0.001` or neither, the same lines at the same seeds, so those
  breaks are the model's own choices (they went away with `EFFORT_LINE=thinking`, below). With it set: greedy hashes
  unchanged (`00ca87eb05ae` / `c88e25e03b02` / `650453259cef`); T=0.7 prose decode 39.2 tok/s at one stream and 81.4
  at four against 40.2 / 80.1 with `top_k: 0` sent, the same within run spread (6 runs each); startup line 4,098 pages
  and a 3.1 GiB cache.
- **2026-11-07** -- `start.sh`: an exported `NCCL_SOCKET_IFNAME` that is an NCCL exclusion list (`^lo,docker,...`,
  often from `/etc/profile.d`) no longer stops the start with `E_CONFIG`; it is set aside with a `W_CONFIG` warning and
  the link is found as when empty (issue 17, mjjjjaazing). Test: `scripts/tests/nccl-ifname.sh`.
- **2026-11-07** -- `scripts/tests/patch-numbers.sh`: a number in `patches/engine/` names one patch, checked by
  `scripts/tests/run.sh` (issue 15, travellingsoldier85's check). README: with `CACHE_FIRST=1`, `PROMPT_CACHE_GIB` is a
  priority, not a floor, and a larger ask can fund more cache by costing the streams a window (issue 16, jds1-ai's
  arms and formula; laidick); troubleshooting rows for an exported `NCCL_SOCKET_IFNAME` exclusion list (issue 17,
  mjjjjaazing) and for clones from before the 2026-11-04 history (issue 16); the Ablit weights' drafter cost depends
  on the memory plan (issue 17). From outside PRs: `cache-replay.py` names warm turns 0104 and sends `SERVED_NAME`
  (#12), the settings table's `HEAD` and `REASONING_EFFORT` defaults (#13), and the startup check's page count with
  `CACHE_FIRST=1` (#14), all by travellingsoldier85.
- **2026-11-07** -- `PROMPT_CACHE_MIN_GIB`, opt-in (empty by default; #21 by laidick): `start.sh` reads the prompt cache each rank's
  startup line funds and, before LIVE, stops both ranks with `E_CACHE` when one funds less. `PROMPT_CACHE_GIB` caps
  the cache but does not guarantee it, even with `CACHE_FIRST=1`: the cache is what that start's room leaves after the
  streams' windows, and the room moved between 17.8 and 18.9 GiB across starts with no config change (issue 16,
  jds1-ai). "no prompt cache" and a cache in host memory count as 0 on the device; a rank 0 log with no startup line
  stops the start; rank 1, which follows rank 0's plan and prints no startup line on the pair, is checked only when
  its log has one. Test:
  `scripts/tests/cache-min.sh` (startup lines in the engine's formats, ranks stubbed), run by `scripts/tests/run.sh`.
  On the pair: `PROMPT_CACHE_MIN_GIB=1` starts (rank 0 funds 3.1 GiB), `=20` stops both ranks with `E_CACHE`.
- **2026-11-06** -- engine patch 0110: `KV_SHARD=1` (`TF_GLM_KV_SHARD`), opt-in (#24), KV sharding at TP=2 by
  Robert Wojciechowski ([@wojo](https://github.com/wojo), X [@rwojo](https://x.com/rwojo)), toward upstream [#1431](https://github.com/ashhart/TensorFold/issues/1431). Each stream's FP8
  latent rows live on its owner Spark alone, so the pool holds about twice the tokens (4,098 to 8,196 pages at the
  defaults). On the pair: every greedy reply identical to unsharded across eight request sets (1-8 concurrent
  streams, eight 30k contexts, a 102.5k conversation); decode -1.6% at one stream, -3.6% aggregate at eight; cold
  first token unchanged; `tf-cuda-test glm-kv-shard-check` byte-identical to an unsharded twin over 10 paths, kept
  states through the disk image included. With `CACHE_FIRST=1 PROMPT_CACHE_GIB=6 LEARN_GIB=30`: a 4.1 GiB cache and
  3.15M tokens of pool a rank, and a 102.5k conversation resumed from disk after a restart in 1.13 s.
  Re-measured on the maintainer's pair (2026-11-07, image `8734689e7a73`, published weights): 6,147 pages and a
  4.3 GiB cache with `CACHE_FIRST=1 PROMPT_CACHE_GIB=6` (4,098 pages and 3.1 GiB unsharded); greedy hashes equal;
  decode 37.5 / 55.4 tok/s at one stream (prose / code; unsharded 38.3 / 56.4), 77-78 at four (79-80);
  `glm-kv-shard-check` passes on both ranks (10 paths, 510 rows); an 18k and a 101k conversation resume from disk
  after a restart under sharding with the in-memory reply.
- **2026-11-06** -- engine patch 0109: `LEARN_GIB` (`--learn-conversations`), opt-in (#22), a backport of upstream
  [#1411](https://github.com/ashhart/TensorFold/pull/1411) by Robert Wojciechowski ([@wojo](https://github.com/wojo), X [@rwojo](https://x.com/rwojo)). Kept prompt states also go to disk on
  the head Spark, every rank's part, so conversations survive eviction and restarts. On the pair at the served
  `PARALLEL=8` and 1M window: off, greedy hashes equal main's (6 of 6) and decode is unchanged (35.07 against 34.60
  tok/s, 1,024 tokens); on, an 18.7k-token turn resumes after a restart in 1.9 s (cold 10.7 s), after an eviction in
  2.5 s, queued behind a 52k-token prompt with its state read ahead, and with `"draft": false`; a 102.5k-token one
  after a restart in 1.8 s (cold 58.4 s); every reply equal to the in-memory resume. Damaged state or block files are
  refused and the turn prefills. Writing a state costs a stream decoding beside it about 1%. Host tests: the disk image
  round trip on a device-only lease and on one that runs onto 0015's host pages.
  Re-measured on the maintainer's pair (2026-11-07): off, all greedy hashes, the startup line (4,098 pages, 3.1 GiB)
  and decode (38.3 / 56.4 at one stream, 79-80 at four) equal main's; on (`LEARN_GIB=30`), after a full restart a
  101.5k-token conversation resumes 98,304 tokens cached (first token 3.6 s) and an 18.6k one 16,384, both replies
  byte-identical to the in-memory resume; rank 1 logs clean.
- **2026-11-06** -- `EFFORT_LINE=thinking` (default; #23 by wagzee): the template's `<|system|>Reasoning Effort: Max` line with
  thinking on only, as TensorFold v0.6.0's recipe wrote it; `always` serves the template as it is. The checkpoint's
  template (and the vision copy) sets the line to Max unless a request's effort is low or high, so every prompt
  carried it, thinking off included (7 tokens; `/tokenize` on a one-line user message: 27 against 20), and a
  thinking-off reply reasoned in its answer. `start.sh` writes the chosen template to `.state/chat_template.jinja`
  with its second line taking the effort from `reasoning_effort`, which the engine passes with thinking on and
  leaves null with thinking off (prompt.zig), and serves that copy on both ranks; a template without the line the
  checkpoint ships stops the start (`E_CONFIG`); `DRY_RUN=1` checks and writes nothing. Measured with sparkDash's
  quality suite on this pair (755 items, greedy, fixed seed, `PARALLEL=4 KV=bf16`), the same image: QA 79.3% (119/150)
  -> **86.0%** (129/150), the five "binary representation, digits only" items that ran out of their 64 tokens
  mid-division and the four transposed string reversals answered as v0.6.0's run did (140 of 150 QA replies
  byte-identical to that run, served `DENSE=fp8`); MMLU 91.2% (260/285) -> 88.4% (252/285), the line's visible
  reasoning no longer helping the thinking-off letters (v0.6.0's run: 88.1%); reason, arith, track and GSM8K
  byte-identical to the run before (their prompts think, so the line stays). No reply ends on `length` any more.
  README: the setting, "The effort line", a troubleshooting row.
  Checked on the pair (2026-11-07): a one-line thinking-off prompt is 8 tokens instead of 15, thinking on 14 both
  ways; "binary representation of 217, digits only" at 64 tokens answers `11011001` (it ran out of tokens before);
  the thinking-on greedy hash is unchanged (`650453259cef`). Copying a tool-call markup line 30 times with thinking
  off went from 30-36% malformed lines (greedy included, `top_k` 20 or not) to 3 of 1,419.
- **2026-11-06** -- engine patch 0108: `CACHE_FIRST=1` (`TF_PROMPT_CACHE_FIRST`), opt-in. The pool counted the room in
- **2026-11-06** -- engine patch 0112 (main's 0108): `CACHE_FIRST=1` (`TF_PROMPT_CACHE_FIRST`), opt-in. The pool counted the room in
  whole windows before the prompt cache, so `PROMPT_CACHE_GIB` only capped the remainder (issue 7: crystal8194,
  tanchunsiong, unitea1992; issue 3). With it, the cache is set aside first and the streams share the pages left, one
  window at least. On the pair at `PARALLEL=8` and the 1M window with `PROMPT_CACHE_GIB=12`: a 9.3-9.7 GiB cache; a
  380k-token conversation resumes 99% cached after eight ~31k ones (3.0 s instead of 234 s) and all eight resume;
  hashes and decode unchanged; three concurrent agent loops as fast. Off, the plan is as before. The startup log says
  when the room funds less cache than `--prompt-cache-gib` asks, and names what frees more (tanchunsiong's
  suggestion). README: the long-session setting is now `CACHE_FIRST=1 PROMPT_CACHE_GIB=12` at eight streams.
  `WITH_VIDEO=1`, opt-in: `docker/Dockerfile` builds `libtfvideo.so` over upstream's static FFmpeg 9.0.2 into its own
  image tag (the default image and its tag are unchanged), so `VISION=1` serves `video_url` parts (issue 8, chai-md;
  checked: a three-colour clip described in order, text hashes unchanged). Issue 9 (pepelaskov): the dropped digit
  reproduces on the Ablit weights only; the published weights copy the id exactly on the same image and request, and
  so do the Ablit weights with thinking on (README "Ablit weights").
- **2026-11-06** -- README: a measured setting for long agent sessions, `PARALLEL=2 CONTEXT=524288
  PROMPT_CACHE_GIB=32`. On the pair, a 382k-token conversation, then eight unique ~31k ones, then the long one's next
  turn: with this setting rank 0 funds a 12.4 GiB cache, the long turn resumes 100% cached (0.8 s instead of 237 s)
  and all eight short ones resume at 94%, with decode and greedy hashes unchanged. The served `PARALLEL=8` (3.4 GiB)
  resumes 1 of 8 and not the long one; `PARALLEL=4 PROMPT_CACHE_GIB=16`, with or without `CONTEXT=524288`
  (5.1-5.4 GiB), resumes 3-4 of 8 and not the long one. The trade is two streams at once; there is still no disk or
  host tier on GB10. Measured alongside, with no change taken: tool-eval-bench 2.8.0 `--hardmode`, 3 seeds, scores 88
  thinking off and 93 thinking on (reasoning high). A build with 2,048-row prompt chunks (without 0094/0095) has the
  same hashes and a shorter worst pause beside a long prefill (1.6 s against 2.7 s), but p95 pauses 10 times longer
  and three concurrent agent loops 3.6 times slower, so 4,096 rows stay.
- **2026-11-05** -- engine patch 0111 (main's 0107): `TF_TOOL_CALL_GUARD=1`, opt-in. A Hermes agent showed raw
  `<tool_call>session_search<arg_key>…` text: its turn after two tool rounds was sent without `session_search`, and
  the server (as vLLM) keeps a call to a tool the request does not offer as text. With the guard, such a request gets a
  closing system note that tools are unavailable and the markup hidden (no tools, or `tool_choice: "none"`), or only
  the unoffered call hidden (other tools offered); streamed or whole. Measured on a live serve with the patch: off,
  every case replies as before and the greedy hashes are unchanged; on, no markup in 12 of 12 such replies (prose or
  empty), calls to offered tools still parse (4 of 4), hashes and speed unchanged. README: the patch row and a
  troubleshooting row.
- **2026-11-05** -- the memory guard `start.sh` leaves running holds none of the caller's stdin, stdout or stderr: it
  kept the pipe of `ssh spark './start.sh'` open, so that ssh session never returned after LIVE (issue 7, crystal8194).
  Its one message now goes to `memguard.log` in the run directory (it went to the terminal and the log). A test,
  `scripts/tests/memguard-detached.sh`, reads the guard's descriptors from /proc; `scripts/tests/run.sh` runs the
  syntax check, shellcheck when installed, and the tests.
- **2026-11-05** -- README: the prompt cache is funded from the remainder after the page pool and the streams' fixed
  parts, so `PROMPT_CACHE_GIB` only caps it (eight streams at 1M leave under 1 GiB; `PARALLEL=4` funds 8 GiB), the
  room differs by a few GiB between Sparks so the reserve that holds two windows is a per-pair check, hits land on the
  4,096-row chunk grid, and nothing survives a restart at this pin (issue 3: jds1-ai, micachi; issue 7: crystal8194).
  Video input: the template accepts `video_url` parts but the image ships no `libtfvideo.so`, so they return HTTP 500;
  the settings table, the "What is checked" line and a troubleshooting row say so, with the out-of-image build that
  works (issue 8, chai-md). Building the video runtime into the image is a Dockerfile change with an FFmpeg stage,
  not taken here.
- **2026-11-04** -- `WORKER_WEIGHTS=check` accepts a local checkpoint with no `.revision` marker when its config.json
  and shards are there, as `worker_weights_ready`'s comment always said; `MODEL_REVISION`'s default made the marker
  mandatory even for `MODEL_SOURCE=local` (issue 3, failure 1). The `E_WEIGHTS` hint names the marker.
- **2026-11-04** -- the worker's state paths (`WORKER_CACHE_DIR` and the state and overlay built on it) default under
  the worker's own home when its login differs from this Spark's and every default is untouched; an explicitly set
  path wins as before (issue 3, failure 2). The `E_START` failure to create the directory says what to set.
- **2026-11-04** -- `start.sh` runs rank 1 as the worker's own login when `CONTAINER_USER` is this Spark's (the default).
  With different ids on the two Sparks (uid 1000 and 1001 on ours), rank 1 could not write
  to its state folder and stopped at `hashing the checkpoint failed at model-00001-of-00083.safetensors` with
  `AccessDenied`. Equal ids, root and another `CONTAINER_USER` run as before; a `CONTAINER_USER` set to this Spark's own ids
  counts as the default. With different ids the start stops, with the fix, when a folder in rank 1's `HOME`
  (`WORKER_CACHE_DIR/home`) is not writable by that login, which a workaround that let an earlier rank write there can leave
  behind. A worker reading shards only the head's id can read (mode 0600 over NFS without `all_squash`) now needs shards its
  login can read.
- **2026-11-04** -- `0108` (candidate, off by default): the four-rank RoCE ring all-gather. `ROCE_RING=on ./start-tp4.sh`
  carries the decode and verify all-gathers up to `ROCE_MAX_KB` as one-shot RDMA writes to both neighbours, each
  relaying half a shard to the rank across (`TF_ROCE_RING=1`, `TF_ROCE_HCA=<toward previous>,<toward next>` per
  rank); any rank that cannot bring it up puts all four back on NCCL. `tf-cuda-test roce-ring <rank>` checks bytes
  against NCCL, graph replays and times a call (16 KiB a rank: 20 us against NCCL's 28). On FP8 at TP=4, same image
  A/B: +1-2% at one stream, +5-7% at four, prefill, hashes and NLL identical. Under the pre-registered 3% one-stream
  line, so the default stays off; the image (5d37e5287d7b, 108 patches) is on all four Sparks. docs/speed/RESULTS.md
  "The four-rank RoCE ring all-gather".
- **2026-11-04** -- `0107`: block FP8 checkpoints (`quant_method: fp8`, e4m3, 128 x 128 scales) load on this engine:
  `FP8=1 ./start-tp4.sh` serves Z.AI's own release `zai-org/GLM-5.3-Flash` @ `46a050be` (the model is published in
  block FP8; `unsloth/GLM-5.3-Flash-FP8` is its byte-identical mirror, the copy this kit measured) at TP=4 on the
  four-Spark ring (README "Block FP8 weights"), ~81 GiB a rank, NLL
  0.7425 against the EXL3 pack's 0.7487, decode 44.7 / 65.2 tok/s prose / code at one stream against 56 / 85 (twice
  the expert bytes a round), prefill the same. The routed experts take the engine's existing block-FP8 expert kernels;
  every other quantized linear its block-FP8 lane; `weight_scale_inv` is read as the scale. Reviewed with Codex before
  and after; the pre-registered gates and the measurements are in docs/speed/RESULTS.md.
- **2026-11-04** -- CREDITS and NOTICE: the Aevonix entry removed. On reflection it earns no line here: the serving scripts are
  this lab's own (Aevonix's recipe reworked the lab's GLM-5.3-Flash-EXL3-4x-DGX-Sparks-TensorFold scripts, and this recipe's
  copies reach it through the lab's 2x RTX PRO 6000 recipe), the banner was never theirs, and the lab's own 2x RTX PRO 6000
  repository keeps the credit where its scripts are Aevonix's rework outright. The one debt that survives is the K12 loop, now
  recorded where it is used: patch 0078's comment names patches/aevonix/0073 in Aevonix's
  GLM-5.3-Flash-EXL3-4x-RTX-PRO-6000-TensorFold (it said MiaAI-Lab/Aevonix, a repository that does not exist).
- **2026-11-04** -- CREDITS and NOTICE, the Aevonix entry corrected: it said this recipe's `start.sh`, `stop.sh`,
  `scripts/config.sh`, `scripts/prepare.sh` and `scripts/banner.sh` are adapted from Aevonix's 4x RTX PRO 6000 recipe, which
  states the direction backwards. The adaptation began from this lab's own scripts: Aevonix's NOTICE names the lab's
  `scripts/config.sh` and `start-tp4.sh` (GLM-5.3-Flash-EXL3-4x-DGX-Sparks-TensorFold, created 2026-10-01, the day before
  Aevonix's repository) as "the basis" of their `start.sh` and `scripts/config.sh`, and their `stop.sh` and
  `scripts/prepare.sh` as "adapted from Mia's scripts"; and `scripts/banner.sh` here is the lab's own art - the two-Spark and
  four-Spark recipes carry the same MIA AI LAB banner file and this recipe recolours it, while Aevonix's banner is their own
  artwork, so it never came from them. The old text also cited patch 0073 to "MiaAI-Lab/Aevonix", a repository that does not
  exist; the K12 loop patch 0078 follows is `patches/aevonix/0073` in Aevonix's repository. What survives of the credit: this
  recipe's four serving scripts do pass through Aevonix's rework (the 2x RTX PRO 6000 recipe adopted it - their texts are the
  closest matches to ours), so Aevonix stays credited for the intermediate rework and the K12 loop, as the reworker of the
  lab's scripts rather than their origin.
- **2026-11-04** -- Upstream TensorFold since this recipe's pin (`f350cd3e`, 1,059 commits behind main) was reviewed for
  fixes worth carrying (docs/speed/RESULTS.md, "Upstream since the pin"). The one that applied cleanly, the lane fix
  `ed39acc5a` (a stream finishing on its first token released its lane only after its trace), was built as `0106` and
  **killed on the pair**: with it, decode fell to 28 / 36 tok/s at one stream and 51-66 at four, with a connection and a
  waiter left open after twenty one-token replies; the same sequence on the 105-patch image is normal (38 / 57 / 78-80).
  Upstream's deferred release does not fit this stack's lane host (0093, 0104). The candidate stays in
  docs/speed/candidates as a record; the served image is the 105-patch `8c4ca7bd8c39`.
- **2026-11-04** -- `0105`: a tool parameter whose schema names no type (`{}`, a description alone, a `$ref` that resolves
  nowhere) takes the JSON object or array the model wrote instead of coming back as a JSON string of it; scalars, plain
  words and quoted strings stay text, and a tool declared without a parameter schema keeps streaming its arguments as
  before. The rule is the maintainer's from the 2x DGX Sparks recipe's issue #180 (upstream keeps untyped values as text);
  a union that admits an object already decoded one on this engine pin. Found by a sweep of the sibling repos' issues
  (docs/speed/RESULTS.md, "Sweep of the sibling repos"), reproduced here, host-tested (43 parser + 7 parameter tests).
- **2026-11-04** -- `0104`, warm turns for GLM (ported from the 2x RTX PRO 6000 recipe): after a reply the server prefills the
  next turn's rendering in the background (a prompt-only pass that steps aside for real work), so a user message after an
  agent loop resumes past the loop instead of re-reading it. Measured here (docs/speed/RESULTS.md): the next user message
  after a 9.3K-token tool loop cached 8,192 tokens (was 4,096), 1.6 s to answer (was 3.5 s); 17 of 18 passes stepped aside
  under back-to-back requests; first-token latency 0.15 s unchanged; decode 38.4 / 56.6 at 1 stream, 79-81 at 4 streams,
  greedy hashes unchanged. Each pass logs `next turn: background prefill of N tokens done` or `stepped aside`.
- **2026-11-04** -- `0103` (ported from the RTX recipe's `0096`, with a guard): a tool call that follows prose quoting
  tool-call markup ("Qwen's `<tool_call>{}`") is parsed as the call instead of coming back as content. The guard: an opener
  inside a value still open in the text before it (an unbalanced `{`, an open `<arg_value>` or `<parameter=`) is never
  retried - without it a call quoting a whole call inside a JSON string argument invoked the quoted call (found by Codex,
  confirmed by a host test). `zig build test-tool-parse-host`: 42 pass.
- **2026-11-04** -- `MEMORY_RESERVE_GIB` 8 -> 9.5 (scripts/local.sh): the prompt cache had silently gone to ZERO. The page
  planner takes whole 1M-token windows of pages before the cache, and the room (19.8 GiB at reserve 8, up from 18.2 the
  day before) crossed the threshold for a third window; rank 0 logged "no prompt cache" and every chat turn re-prefilled
  its history (6 s a turn at 9.3K tokens). At 9.5 the pool is two windows (4,098 pages, 2.1M tokens) and the cache 3.6 GiB;
  decode, prefill, mixed and 8-stream numbers unchanged, 1.5 GiB more free memory. Check rank 0's startup line after
  every start: "4098 pages", "prompt cache: N GiB", no "cannot keep one state" warning.
- **2026-11-04** -- Decode campaign items closed by measurement (docs/speed/RESULTS.md): KDA C1 (merging the KDA chain's
  launches) - fusion loses, dependent launches save ~0.2 ms a round, below the engine's resolution; the exact PDL
  variant is parked as candidate `0199` (opt-in `TF_GLM_PDL_CHAIN=...,kda`). Sampled decoding at temperature 0.7: this
  pair 36.7-38.9 tok/s at 1 stream and 75-76 at 4, the spark1+2 reference 21 and 30-31 (its round doubles under
  sampling); its 4-stream lead over this pair is greedy-only.
- **2026-11-03** -- `0099`: `TF_GLM_PROF=1` also brackets shared windows (several streams in one forward), reported as
  `glm prof shared N rows` by total rows; off unless set, no kernel or token change. With it, a 4-stream round is
  attributed (docs/speed/RESULTS.md, "Decode attribution campaign"): routed experts 4.2-5.8 ms a row at the GB10's
  memory bandwidth, KDA mixers 16-17 ms flat, drafter 7-8 ms. Two exact candidates measured neutral and parked in
  docs/speed/candidates: `0098` (`TF_GLM_DFLASH2_EDGE`) and `0100` (`TF_GLM_MM_STAGES`). The chain cut, the tpm rule,
  a bf16 drafter, bf16 dense and stock TensorFold v1.2.1 were all slower or equal; the numbers are in that section.
- **2026-11-03** -- `PREFILL_SHARE` (default `0.68`, passed to both ranks as `TENSORFOLD_PREFILL_SHARE`): a stream
  decoding beside another request's long prefill no longer gets one decode round per prompt chunk. A chunk is atomic
  (4,096 rows with 0094/0095, about 2.5 s here), and at the engine's default share of 1 a chat beside a long prefill
  produced a token or two per chunk. At `0.68` each chunk first owes the decoders about 1.2 s of decode rounds (~22),
  and the long prompt's first token comes about 47% later. Replies are byte-identical at any value: the share only
  moves when the next chunk runs. The rows in the README are the knob's arithmetic on this pair's measured chunk and
  round times; the mixed-traffic measurement is from the 2x RTX PRO 6000 recipe (2.3 -> 27.2 chat events/s during a
  cold 175K prefill, that prompt's first token 33.4 -> 49.8 s). `PREFILL_BUDGET` is passed as
  `TENSORFOLD_PREFILL_BUDGET` when set; it cannot split a chunk. Takes effect at the next `./start.sh restart`
  ([README](README.md#prefill-beside-decoding)).
