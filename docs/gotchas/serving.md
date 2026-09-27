# Gotchas: serving ops

Prefix cache, vision tower, warmup, streaming, loaders, and concurrent prepares.

Entries 8, 37, 38, 39, 41, 44, 45, 55, 59, 60 of [docs/gotchas.md](../gotchas.md) (original numbers kept, so `gotcha N` references still land).

[← gotchas index](../gotchas.md) · [quickstart](../quickstart.md)

8. **`--language-model-only` drops the vision tower cleanly** (no weights
   loaded), and it is the default in both start scripts. `VISION=1` keeps the
   tower for a client that sends images. The tower is **0.858 GiB**, not the
   2.7 GB this entry used to give: `model.visual.*` sums to 0.858 GiB of BF16 in both
   `Qwen3.8-27B-W4A16-AutoRound` and the `-fast` variant, and a runtime A/B
   agrees — model loading reads 15.13 GiB against 14.26 with
   `--language-model-only`, same server, same config, with non-weight overhead
   0.42 against 0.41 GiB either way. Quantization is not the explanation: the
   tower is BF16 in both dirs. Where 2.7 GB came from I could not work out.
   The KV pool came out within 0.4% across that pair (183,673 against 184,438
   tokens at 150k/fp8) — but the profiled activation peak differed by 0.87 GiB
   between those two starts, which is the same run-to-run swing the V2 runner
   shows, so read the pool difference as noise rather than as a measurement of
   what the tower costs.

   On `SPEC=dflash2` that 0.858 GiB is not optional headroom, it is the difference
   between booting and not. Measured here on a 3090 at 250 W, `SPEC=dflash2 VISION=1
   VISION_OFFLOAD=0`, `CTX=fast`: the engine dies in graph capture with
   `torch.OutOfMemoryError: Tried to allocate 960.00 MiB ... 787.50 MiB is free`
   (`spec_decode_attn.py:184`, the split-KV verify `part_o` buffer). The pool there is
   pinned by bytes (`KV_MEM`), so the tower cannot come out of the KV cache — it comes
   out of the ~1.1 GiB transient margin, and it is 0.85 of it. `VISION_OFFLOAD=1` (the
   default) makes the same config come up at the full 69,758-token pool and read images.
   The `SPEC=mtp` path has no such problem: it boots either way, and there the pool is
   profiling-sized, so what the tower costs is buried in the ±0.87 GiB profiling swing
   above (79,271 against 80,055 tokens across the pair — 1%, i.e. noise).

   What `VISION_OFFLOAD=1` does: the tower's weights live in pinned host RAM and each
   module is copied to the GPU for the duration of its own forward
   (`patches/vision-tower-cpu-offload.patch`). Isolated-tower measurement, RTX 3090 at
   PCIe 4.0 x16, one 8192-patch image, median of 10 forwards: resident weights 891.3 ->
   9.0 MiB, peak allocation 1160.5 -> 308.2 MiB, encode 296 -> 333 ms, output bit-exact
   against the resident tower. Note which offload path that is: vLLM's UVA *zero-copy*
   mode saves the same memory and costs 3327 ms, because the GEMMs then re-read operand
   tiles over PCIe inside the inner loop. The patch forces the bulk-copy path and does
   not touch what `--cpu-offload-gb` does elsewhere -- which could not reach the tower
   anyway, since the offloader is only installed in `make_layers()`.

   One precision from re-verifying the premise on a headless 3090, same 250 W:
   with nothing else on the card, `VISION_OFFLOAD=0` *does* boot -- and lands at
   440 MiB free after boot, inside gotcha 35's kill zone (396 MiB free died on a
   concurrent burst where 436 survived). The hard no-boot above needs something
   else holding a share of the card -- the measuring box also ran a desktop
   compositor and a browser, which is the normal state of a 3090 in a
   workstation. Same conclusion from both geometries: a resident tower puts the
   engine at the headroom cliff, the offload puts it at the full margin, and
   that is why the default is on. On the current tree the pool prints 68,605
   tokens (`KV_MEM` has moved since the 69,758 above was measured), identical
   between `VISION=0` and `VISION=1`, and the image round-trip reads a marker
   that exists only in the pixels either way.

37. **sm80 (GA100) Marlin repack can Xid-31 the whole card under memory
    pressure — and the kernel in the traceback is innocent.** Community
    finding, [@ahnguyen17 in #27](https://github.com/syv-ai/HyperQwen/issues/27#issuecomment-5397500895),
    on a CMP 170HX 40 GB: with ~27 GB resident, `gptq_marlin_repack`'s GB-scale
    int64 intermediates (k×n int64 ≈ 1.4 GB per 27B layer, several live at
    once) churn sm80 VMM mappings until an unrelated, trivially correct
    elementwise kernel takes an async write fault — the faulting frame drifts
    between runs, the Xid 31 wedges the card until reboot, and
    `compute-sanitizer` is clean on sm86 with identical inputs. Their
    workaround, serving in production since: compute the repack on CPU
    (bit-exact, ~3 min extra boot) —
    [`sm80-int8-repack-cpu-fallback.patch`](https://github.com/ahnguyen17/cmp-170hx-vllm)
    — with `expandable_segments` kept **off**, which on that card is an
    independent Xid-31 trigger. Not shipped here (no sm80 to regression-test
    against); recorded so the next GA100/A100 report starts from the answer
    instead of from five reboots.

38. **The OffloadingConnector's CPU tier can be silently useless: uniform
    blocks meet asymmetric chunk sizes, and one request evicts everything
    (issue #33).** The tier allocates equal-size blocks sized for the LARGEST
    group's offload chunk. Under KVarN the drafter's sliding-window group
    carries 128-token chunks against the 2,176-token maximum, so every SW
    crumb occupies a full ~14.6 MiB block — a single 23k-token request eats
    ~264 of a 4 GiB tier's 293 blocks and LRU-evicts every previous
    document. Stores succeed, `complete_store` succeeds, and every
    cross-request lookup is a MISS: 41 GB written, 0 bytes ever read back,
    with nothing in the logs. On bf16 KV the SW group happens to share the
    large per-token size (gotcha 28's 4096-B coincidence), the geometry
    stays uniform, and the same connector uplifts at PCIe speed — the KV
    dtype was never the mechanism, the chunk geometry it induces was. Since
    `offload-dflash-eagle-groups.patch` the config builder warns at boot
    with the waste factor and the `cpu_bytes_to_use` multiplier that would
    compensate (~17x under KVarN). Same patch fixes an adjacent quiet bug:
    upstream only ever sets `is_eagle_group` for DeepSeek V4, so the
    connector's fallback marked EVERY group as draft attention under
    `method=dflash`/`mtp` and silently excluded each group's trailing chunk
    from store while decoding; with dflash the flag now lands on the
    drafter's sliding-window group alone. Also: on bare-metal Linux the
    connector refuses this stack's default allocator
    (`expandable_segments:True`) at config validation. All three launchers
    now default `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` whenever
    `EXTRA_ARGS` carries `--kv-offloading-size` or `--kv-transfer-config`
    (and print a line saying so); an explicit `PYTORCH_CUDA_ALLOC_CONF` in
    the environment still wins, so setting it to `True` by hand reproduces
    the refusal. The MTP/EAGLE serve fixes are in the series as
    `offload-mtp-serve.patch` (upstream #52771 and #52807 plus the
    finished-request store watermark clamp, #54288): with it the tier serves
    stored hits under `SPEC=mtp` and the cached count on a replay lands on the
    same block formula as the GPU path, one 832-token block more than before.
    **Sizing the tier, and why a default boot can show it doing nothing.**
    Residency has no gauge in 0.28.0 or 0.29.0: both `kv_offload_cpu_cache_usage_perc`
    and its read twin count in-flight transfer pins (`num_used = allocated -
    free - evictable`, and `complete_store()` marks a block evictable the
    moment it lands), so a full tier reads 0.0 between transfers and a 0%
    reading is not an idle tier — read `vllm:kv_offload_total_bytes_total`
    and the external-prefix-cache hit counter instead. Size the tier from
    blocks x tokens-per-block for YOUR arm, never from a GiB constant:
    `blocks` = mmap bytes / bytes-per-block (the usage gauge quantises at
    1/blocks, e.g. 0.02252... = 10/444), and `tokens-per-block` = distinct
    tokens stored / `kv_offload_cpu_allocation_size_sum`. Measured on the
    3090, `CTX=long` (fp8, MTP) at `--kv-offloading-size 12`: mmap
    12,861,308,928 B over 444 blocks = 28,966,912 B per block; 367 blocks for
    123,148 distinct tokens = ~336 tok/block; 444 x ~336 ~= **149k tokens**,
    i.e. about ONE 150k session, not three. `CTX=huge` (KVarN) at the same
    12 GiB is 110 x 2048 ~= 225k. Then read the GPU side off the SAME boot
    log (`GPU KV cache size: N tokens`) and compare in TOKENS, not bytes: a
    tier several times the pool in bytes can be smaller in tokens (86.2 KB
    per tier token against 37.9 KB per pool token here, a 2.3x ratio), and a
    tier smaller in tokens than the GPU prefix cache can never hold anything
    the GPU cache has already dropped — which is the real reason a
    default-pool 3x150k workload can show the tier doing nothing useful.
    Capacity, not a scheduler veto. Both numbers must come from the same
    boot: the pool is profiled per boot and moves ~1 GiB between a cold and a
    warm compile cache (item 12), enough that a default `CTX=long` boot which
    fits beside a tier one day refuses at KV sizing the next. Sized right,
    the tier behaves as a high-churn working set rather than a resident
    store: the reporter in
    [#95](https://github.com/syv-ai/HyperQwen/issues/95) measured
    658 GiB CPU->GPU, 541 GiB GPU->CPU and 15.95M external prefix hits
    through a 12 GiB tier in ~3 h (dfein38347g).
    And when eviction probing, keep the resend prompt BYTE-identical: a
    two-token label difference shifts every block hash and manufactures a
    convincing, fake "per-request hash instability" (ask how we know).

39. **"Every request re-prefills" is measurable, and the cause is usually the
    client's bytes, not the cache.** Reported against an agent client in
    [#47](https://github.com/syv-ai/HyperQwen/issues/47) (44 s mean
    TTFT at ~44k context, i.e. a full recompute per turn, while plain chat
    clients on the same server sat at the documented decode rates). Work the
    list in order:

    1. **Measure, per request.** The launchers pass
       `--enable-prompt-tokens-details`, so every response's
       `usage.prompt_tokens_details.cached_tokens` says how much of that
       prompt hit. (vLLM never emits DeepSeek's `prompt_cache_hit_tokens`
       field; this is the equivalent.) Server-side,
       `vllm:prefix_cache_queries` / `vllm:prefix_cache_hits` on `/metrics`
       give the same as counters.
    2. **Know the floor, and it has a closed form.** Hits are counted in
       whole hash units and the recurrent state resumes only at aligned
       boundaries (`--mamba-cache-mode align`), so the hit length truncates
       DOWN to a multiple of the hybrid attention block `B` — and the final
       block of a cached request is never served, because its recurrent
       state was never checkpointed at that boundary. Measured on the box
       over two block sizes ([#102](https://github.com/syv-ai/HyperQwen/issues/102)):

           cached_tokens = max(0, floor(n_prefix / B) - 1) * B

       where `n_prefix` is the length of the EARLIER, cached request — not
       the replay's own `prompt_tokens`. Round lengths cannot tell those two
       apart; straddle lengths can. At B=480, a 958-token prompt replayed at
       961 tokens gives `cached_tokens` **0**, where `floor(961/480)-1`
       would predict 480; 1438 replayed at 1441 gives 480, not 960. So a
       prompt shorter than 2B can never hit at all, and a shared prefix pays
       up to two blocks of recompute past the match. This is a fixed tax,
       not the 100%-miss failure mode.

       **`B` is not a constant — read it from the boot log.** Every launch
       prints it:

           INFO [interface.py:928] Setting attention block size to 480 tokens
                to ensure that attention page size is >= mamba page size.
           INFO [interface.py:952] Padding mamba page size by 1.27% to ensure
                that mamba page size and attention page size are exactly equal.

       `B` is whatever makes one attention page cover one mamba page:
       `B = 16 * ceil(mamba_page / (16 * attn_page_1_token))`. On this model
       `attn_page_1_token` is 4096 B at bf16 KV, 2080 B at
       `int8_per_token_head` and 2048 B at fp8, and — the part that surprises
       people — `mamba_page` includes the speculative drafter's state, so it
       grows with the number of draft tokens: 1,634,304 B with no drafter
       plus 20,480 B per draft token. That makes `B` a function of
       `DFLASH_TOKENS`, not of the machine. Measured and reproduced pairs:

       | profile | KV dtype | drafts | B | mamba padding |
       |---|---|---|---|---|
       | CTX=fast (production) | bf16 | 15 | 480 | 1.27% |
       | CTX=fast | bf16 | 7 | 448 | 3.23% |
       | SPEC=dflash2 CTX=long | int8_per_token_head | 7 | 864 | 1.09% |
       | SPEC=mtp CTX=long | fp8 | 3 | 832 | 0.48% |

       Rows 1 and 3 are boot lines from this box; rows 2 and 4 are the same
       formula evaluated against this box's own config (no boot) and they
       reproduce a contributor's boot lines on other 3090s to the last digit,
       padding percentage included. That is the point: two boxes running the
       "same" profile print different `B` purely because their
       `DFLASH_TOKENS` differ (480 here at 15, 448 there at 7) — nothing
       about the silicon, the build or the int8 prefill path is involved.
       The contributor reports 448 and 832 stable from 0.28 to 0.29. Never
       hard-code `B` into a client's cache arithmetic — read the boot line.
    3. **Byte-identity is over the RENDERED prompt.** What the cache hashes
       is the chat-templated token stream: system prompt + tool definitions +
       every message, in order. One changed byte at position P invalidates
       everything after P. The classic offenders are dynamic content early in
       the payload: a timestamp or "current status" block in the system
       prompt, a heartbeat line spliced into the history, compaction that
       rewrites old turns, tool lists whose order is not stable. Diff two
       consecutive requests' FULL bodies (not just system + tools — the
       messages array too) and find the first differing byte; that byte is
       where your cache hit ends.
    4. **Interleaving evicts.** A cached prefix on this hybrid model holds
       KV blocks plus a recurrent-state page (~16% of the pool per request at
       k=7), and the pool is small. Two conversations round-robining — an
       agent's heartbeat pinging between chat turns is exactly that — can
       each evict the other before its recheck: measured as 0-of-3 warm in a
       3-context round-robin on 24 GB
       ([wsl2-4090.md](../wsl2-4090.md), retention section). The CPU
       offload tier turns that back into 3-of-3 (a RAM restore instead of a
       re-prefill).
    5. **The isolating experiment.** Bypass every proxy and fire the same
       long prompt twice at bare vLLM: if TTFT collapses on the second call,
       the server cache is healthy and the variable is the client payload
       (or a proxy that mutates it); if it does not, look at the server —
       and at 2 and 4 above.

41. **On a low-RAM host, don't let the stock loader race page-cache eviction —
    stream the weights.** A 16 GB host (~10 GiB actually free) died loading the
    15.9 GiB checkpoint at shard 5/8
    ([#39](https://github.com/syv-ai/HyperQwen/issues/39)). Measured
    here under a 10 GiB cgroup cap standing in for that box: the **stock
    loader's memory peak was the cap to the byte** (10,737,418,240) — it loads
    by consuming everything and betting reclaim keeps up, which a fast NVMe
    wins and a busy desktop loses; pushed past the edge it reclaim-thrashes so
    hard the process stops responding even to SIGKILL (uninterruptible I/O),
    which is also what a "hung load" looks like from the outside. The **Run:ai
    streamer bounds the load instead, and is faster here**: 6.3 s vs 11.7 s to
    load, 8.46 GB process-wide peak under the same cap (its `memory_limit`
    caps the staging window; the rest is the engine's ordinary host
    footprint):

    ```bash
    venv/bin/pip install runai-model-streamer humanize
    # NOT runai-model-streamer-s3: it force-imports boto3 at package import
    # and breaks the loader on a box without it; local files don't need it.
    EXTRA_ARGS='--load-format=runai_streamer --model-loader-extra-config={"memory_limit":2147483648}' \
      bash single-user/start_qwen.sh
    ```

    Two adjacent facts from the same experiment: `swapon` cannot save the
    stock loader (mmap'd read-only file pages evict, they never swap — only
    the engine's anonymous memory benefits), and the whole test ran under
    `SPEC=off`, which as of this entry is a real mode rather than a silent
    fall-through to mtp.

    Two more from the #39 reporter's own box, once the loader was solved:
    `memory_limit` sized at or above the checkpoint's largest single tensor is
    the conservative setting (the bf16 `embed_tokens` is 2,542,796,800 bytes on
    both variants; the 2 GiB above loaded fine on the reference box, 2542796800
    is what they settled on) — and the *next* failure on a 24 GB card was not
    RAM at all: the plain `-W4A16-AutoRound` checkpoint leaves only 1.56 GiB for
    KV at `GPU_UTIL=0.93`, which cannot hold the 64k default (`max seq len
    65536 ... 4.76 GiB KV cache is needed`). The `-fast` variant (int4 lm_head
    and MTP head, `prepare/fetch_fast_variant.py`) is the launcher default for
    exactly that reason; with it the same box came up at a 72k-token pool.

44. **Align-mode prefix caching periodically drops whole conversations to a
    0% hit — a geometry lottery plus an inverted eviction order on the one
    mamba state page that unlocks them.** A hybrid cache hit is the
    *intersection* of per-group hits, and the mamba group can only resume
    from a retained state snapshot; without one at or below the attention
    match (minus one 448-token EAGLE margin with spec decode), a fully
    cached multi-thousand-token attention prefix reads as a 0% miss and
    re-prefills from scratch (issue #52, upstream vllm#45238 — the same
    veto is why `--prefix-match-unit` can make things *worse* with spec
    decode). Three stock behaviors compose: align mode materializes ~one
    usable snapshot per turn at the last prefill chunk boundary, and on
    ~22% of turns (448/2048) it lands inside the EAGLE margin — that is
    the reported "every 4-5 turns" period; the fallback (the previous
    turn's snapshot, i.e. the block the hit resumed from) is CoW-released
    early in the turn; and mid-decode frees put every reusable snapshot at
    the *front* of the free queue while the attention blocks they unlock
    sit at the back. Any interleaved traffic evicts the snapshots first,
    and an unlucky turn with the fallback gone reconciles to 0. Measured
    (12-turn conversation, two unrelated requests between turns, shrunk
    pool): 81-91% hits for five turns, then 0% on every turn, TTFT 2.1 s →
    17-31 s, the coordinator logging a discarded 16,576-token attention
    match. `patches/mamba-align-checkpoint-order.patch` keeps up to three
    state blocks per running request per mamba group — the CoW-carried key
    plus the last written prompt-region snapshots — until request end,
    freed last. It ships **default off** (measured no-harm at two pool
    sizes, but the win regime — context comparable to the pool over many
    turns — is not cheaply reproducible in a short cell); enable with
    `VLLM_MAMBA_ALIGN_KEEP_CHECKPOINTS=1` if you see the periodic spikes. Two stronger variants — pinning the snapshots against
    eviction, bounded or not — measured *worse*: each skipped eviction
    lands on the conversation's own attention tail instead, which breaks
    the same hit from the other side. If between-turn traffic exceeds the
    whole free pool, nothing survives by policy; that regime needs a
    bigger `KV_MEM`, not a smarter queue. A second align-mode cost, fixed
    since #101 (`patches/mamba-align-retire-null-gaps.patch`, upstream
    #55450): a long prefill leaves null gaps between the state snapshots
    awaiting retirement, and the base block remover stopped at the first
    gap, so every older state stayed allocated until the request ended. The
    maintainer's per-request counter of live non-null state blocks (inside
    `remove_skipped_blocks`, on the repo's reference 3090, #101 review) read
    up to 16 before the patch and 5 with it on every request; on that box
    peak pool usage during a fresh 20k-token prefill fell 40 percent on
    `SPEC=mtp CTX=long` and 21 to 23 percent where a large cached prefix
    dominated the pool. On a second native 3090 and a WSL2 box, with the
    pool pinned identical across arms, the peak fell 30 to 50 percent across
    14k to 56k-token prefills at 0.28 and 0.29 (the reduction grows with
    depth and is smaller on 0.29, which leaks less before the fix). No gap
    forms on `CTX=fast` with DFlash2, so the fix is inert there: on the
    maintainer's production profile the sampled peak agreed to four decimals
    across 20 requests with and without it (#101 review). The win is
    mtp/long shaped. One note for anyone
    editing that patch file: it carries blank context lines that are a single
    space, and a trailing-whitespace trim turns it into a malformed patch that
    `git apply --check` rejects.

45. **First-request Triton compiles on a fresh boot came from four separate
    warmup gaps, and the last one is invisible without logging what Triton
    specialises on.** Issue #48's fingerprint — a stall in the first large
    chunked prefill after boot, preceded by `jit_monitor` warnings — had, on
    the reference 3090 (`SPEC=dflash2 CTX=huge`, one 30k-token first
    request), four kernels compiling inside request 1, each with its own
    cause: (1) `_prepare_dflash_inputs_kernel`'s `BLOCK_SIZE` ladder only
    reaches 256 on a large prefill continuation, never in decode-shaped
    dummies (`patches/dflash2-prewarm.patch` compiles every rung at boot);
    (2) the rejection sampler's three kernels are upstream vLLM's and never
    run in the profile-time dummy sampler pass, which has no draft tokens
    (`patches/spec-sampler-prewarm.patch` runs one spec-shaped verify in
    `kernel_warmup()`); (3) KVarN's block→slot lookup was sized to a 1024
    floor at profile time and resized by the first serving build, and its
    size is a kernel constexpr (`NUM_BLOCKS_LOOKUP`) — now derived once at
    impl construction from the KV budget as an upper bound (48,934 slots for
    6,103 real blocks; the kernels bound-check, so over-sizing is free); and
    (4) the one that survived all of the above: Triton specialises *integer*
    arguments on divisibility by 16, and the block table's row stride is its
    width — the warmup's `cdiv(max_model_len, group)` = 1920 carries the
    attribute, the runner's real table is 1921 wide and does not, so the two
    launches were different compiled variants however faithfully the shapes
    were mirrored. `stride_bt_b` joined `MAX_BLOCKS_PER_REQ` in the kernel's
    `do_not_specialize`. Measured after all four: **zero** `JIT compilation
    during inference` lines on the same boot and request. The tool that
    found (4): `KVARN_SPEC_DEBUG=1` logs pointer alignment and integer
    divisibility / equal-to-1 for the warmup launch and the first real
    launches of the packed-kv kernel — diff the two lines.
    `--jit-monitor-verbose` prints the signature of each in-request compile
    but truncates the specialisation attributes at 120 characters, which is
    why it could name the kernel and not the cause. `KVARN_LOOKUP_BLOCKS`
    pins the lookup size if a deployment ever needs to.

    One scope note, because the zero was measured on one path: that boot was
    `SPEC=dflash2 CTX=huge`. The int8 prefill path shipped since and brings its
    own uncovered kernels — a current production boot (`CTX=fast`,
    `INT8_ACT=int8 PREFILL_ATTN=int8`) still logs three `JIT compilation during
    inference` lines, for `_k_stats_kernel`, `_k_quant_kernel` and
    `_prefill_attn_kernel`. Same class of gap, different kernels, not yet
    prewarmed.

55. **A streaming response sends nothing during prefill, and a proxy in front
    of the server closes the connection before the first token.** vLLM's SSE
    stream is silent from the moment the request is accepted until the first
    output token, so a long prefill looks like a dead socket to anything with
    an idle read timeout — Bifrost's default is 120 s, and a cold 90K-token
    prompt on this card prefills for ~105 s. The client sees a dropped
    connection, not an error. `patches/sse-keep-alive.patch` (upstream vLLM
    [#51034](https://github.com/vllm-project/vllm/pull/51034), backported to
    0.28.0) adds `--sse-keep-alive-interval`, which emits a `: keep-alive`
    comment line while the stream is idle; comment lines are part of SSE and
    every conforming client ignores them.

    `single-user/start_qwen.sh` sets it from `SSE_KEEP_ALIVE`, default 30 s —
    **so every stream now carries a comment line every 30 seconds by default.**
    That is deliberate (4x margin against a 120 s timeout) but it is not
    nothing if you log raw SSE frames: measured on the reference 3090, an
    18.2 s request emitted 18 comment lines at an interval of 1, and would
    emit none at all at 30.
    Set `SSE_KEEP_ALIVE=0` to turn the emission off, or `SSE_KEEP_ALIVE=`
    (empty) to drop the flag entirely — which is also what you need if the
    launcher is pointed at a vLLM tree that has not had the series applied,
    since the flag only exists because the patch is in it.

59. **Two prepares at once leave a model dir half-written.** Every step of
    `docker/prepare.sh` is idempotent, but the script is not concurrency-safe:
    two runs against one model dir can interleave a shard rewrite with an index
    write, and the damage surfaces much later — a shard missing from
    `model.safetensors.index.json`, a config that disagrees with the tensors on
    disk, or a half-fetched fast variant that `verify.sh` then reports far from
    its cause. It happens without anyone typing two commands: the entrypoint
    runs `prepare` before every start, so a booting container races
    `docker compose run --rm prepare`, and two servers starting together after a
    crash race each other. Fix: the script takes an exclusive `flock` on
    `<models dir>/.prepare.lock` — beside the model dir rather than inside it,
    so `BASE_MODEL_DIR` cannot move it out of the volume — and waits up to
    `PREPARE_LOCK_WAIT` seconds (default 600) for a holder before refusing to
    run. The lock is advisory and is released when the holder exits, so a
    leftover `.prepare.lock` file is inert: it is a lock, not a marker, and
    nothing has to clean it up.

60. **Two long conversations advanced in turn can evict each other completely at
    `CTX=huge`, and the hit rate goes to 0, not to a remainder.** With DFlash2 and
    `PREFIX_CACHE=1`, vLLM retains one Mamba state snapshot per attention block by
    default ("dense"). Each long conversation then pins dozens of snapshots, and
    once two of them no longer fit beside each other, whichever one ran last
    evicts the other — every group, attention included, so `cached_tokens` reads
    exactly 0 and every turn re-prefills the whole prompt. On the reference 3090
    (pool 268,169) two ~32.6K chats alternating reused 0% and spent 31.8 s per
    turn re-prefilling; two ~14.9K chats were fine; one ~103K chat alone was
    fine. The knee scales with the pool (#174 found it at ~50-55K per side on a
    542K pool, and halving `--kv-cache-memory` halved it), so it is capacity, not
    a structural trigger. It is invisible in single-user benchmarks and is
    exactly how a two-agent deployment runs. Fix: retain one snapshot in six
    (`VLLM_PREFIX_CACHE_RETENTION_INTERVAL`, a CLI flag from 0.29 on), which the
    launcher now sets at `CTX=huge` with DFlash2 — 93-99.5% reuse on the same
    pair, no cost to a single long chat, and a reuse needle inside the restored
    prefix still comes back right. The interval must be a multiple of the
    attention block, and that block moves with the draft count (2176 at 7 drafts,
    2432 at 15), so the launcher only sets it for measured counts; for any other,
    `PREFIX_RETENTION` = 6 x the `attention block size` line the boot prints.
    Upstream on 0.29, hybrid + EAGLE models are *forced* to dense unless the flag
    is set explicitly, under a boot line ("defaulting prefix_cache_retention_interval
    to dense checkpointing") that reads like a managed setting and is the arm that
    fails ([#174](https://github.com/syv-ai/HyperQwen/issues/174)).
    **What the sparse interval costs, so you can turn it the right way.** It also
    sets hit granularity: a new conversation's first one or two follow-up turns
    reuse only down to the last retained snapshot, and a conversation shorter than
    one interval reuses nothing on those turns. After that all settings are
    the same. Reference 3090, vLLM 0.29, 7 drafts (block 2176), one
    conversation and no other traffic unless stated, cached tokens per turn:

    | workload | dense | `PREFIX_RETENTION=4352` (2 blocks) | 13056 (default) |
    |---|---|---|---|
    | ~8K chat, turns 2-3 | 81% (1.7 s) | 54% (3.6 s) | **0%** (7.3 s) |
    | ~8K chat, turn 4 on | ~98% | ~98% | ~98% |
    | ~20K chat, turn 2 | 98% (0.8 s) | 87% (3.0 s) | 65% (7.2 s) |
    | ~20K chat, turn 3 on | 99% | 99% | 99% |
    | two ~32.6K chats alternating | **0%** (31.8 s) | 93% (2.7 s) | 93-99.5% |

    So the default trades a few seconds on each new conversation's early turns for
    never losing a long one outright. The knob runs one way: a **smaller** interval
    (`PREFIX_RETENTION`, any multiple of the block) gives finer early hits and
    less capacity before two long conversations collide; a **larger** one the
    reverse. Two blocks already halves the early-turn cost and still held the
    ~32.6K pair; where its collision knee sits is not measured, so a workload of
    many short-to-medium chats is the one to try it on, and one that keeps two
    or more long documents live should stay on the default.
