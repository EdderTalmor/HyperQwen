# Gotchas: speculation internals (MTP / DFlash2)

Verify attention, CUDA-graph capture, scheduling, and the lookup controller.

Entries 12, 15, 17, 18, 19, 20, 21, 23, 26, 27 of [docs/gotchas.md](../gotchas.md) (original numbers kept, so `gotcha N` references still land).

[← gotchas index](../gotchas.md) · [quickstart](../quickstart.md)

12. **FlashAttention-2 does not split KV for multi-query decode.** With k
    speculative tokens the verify step has k+1 queries per request and FA2's
    varlen path then runs one thread block per (request, head): 24 blocks on 82
    SMs, 57 µs per layer at 1.5k context and 1.3 ms at 16k. vLLM's Triton
    unified attention has the same restriction (`max_seqlen_q > 1` → 2-D
    kernel). `patches/spec-decode-attn.patch` (`VLLM_SPEC_DECODE_ATTN=1`, bf16
    KV only) is a 180-line Triton fix. Watch its query cap: the kernel used to
    handle at most `BLOCK_M / (heads per kv head)` = 10 query tokens and fall
    back silently past that, which doubled the step at 25k context the moment
    the verify block grew to 16. It now tiles the query rows instead. And
    watch the pool size: until #91 the kernel multiplied the block id by the
    cache stride in int32, so a pool of more than about 2,383 blocks (bf16,
    block 880) was silently reading the wrong memory for any request placed
    above that id (issue #86, a 64 GB card; a 24 GB card never gets there).
    The one `blk.to(tl.int64)` cast fixes it, `bench/test_spec_decode_bigpool.py`
    forces the shape on any card. The int8 prefill kernel carries the same
    pattern; its cast and test are pending in #109.

15. **The V2 model runner (`SPEC=dflash2`) does not count its CUDA graphs when
    sizing the KV pool** (~1.2 GiB on top of whatever `--gpu-memory-utilization`
    you asked for), the hybrid allocator sizes KV groups by the smallest layer
    bucket, and the profiled activation peak varies by ~1 GiB between starts of
    the *same* config — three ways to get a server that either wastes a quarter
    of its pool or dies mid-request. `patches/hybrid-kv-groups-v2-cudagraph.patch`
    fixes the first two; for the third, pin the pool in bytes
    (`--kv-cache-memory`, what `KV_MEM` does) instead of tuning utilization. That
    runner also answers `thinking_token_budget` with 400, and the first request
    after a cold start JIT-compiles four Triton kernels (~5 s once; cached in
    `~/.triton`).

17. **A Triton kernel's scratch buffers may not grow after CUDA graph capture.** The
    split-KV verify attention sizes its partial buffers from the longest query block it has
    been asked for. Once the block got longer than the drafter's — and once a small prefill
    chunk could land on the same kernel — that "longest so far" changed mid-run, the buffers
    were reallocated, and the captured decode graph went on reading the freed ones:
    `CUDA error: an illegal memory access was encountered`, a few hundred tokens into the
    first request. `VLLM_SPEC_DECODE_ATTN_QMAX` (set by `single-user/start_qwen.sh` from
    `DFLASH_TOKENS`) fixes the size at startup instead.

18. **Async scheduling pins the number of speculative tokens.** vLLM only feeds draft token
    ids — and therefore the *count* the worker wants verified — back to the scheduler on the
    synchronous path (`EngineCore.post_step`). With async scheduling on, every decode step is
    padded to `num_speculative_tokens` and a worker asking for fewer is ignored, silently.
    Adaptive block length (`LOOKUP=1` with `DFLASH_TOKENS > 7`) needs `ASYNC_SCHED=0`; at
    batch 1 that costs under 1%.
    Still true on 0.29.0, and worth knowing *why*, because 0.29 looks like it
    handles this for you and does not: `VllmConfig` disables async scheduling
    automatically for speculative methods outside an allowlist, but that
    allowlist is `EagleModelTypes`, and `DFlashModelTypes` is inside it
    (`config/speculative.py:69` on 0.29.0, `:67` on 0.28.0). So dflash keeps async scheduling on unless
    something turns it off, and the launcher is that something.

19. **`--async-scheduling` is already the default in 0.28.0 and 0.29.0.** The flag exists and passing it
    changes nothing; `--no-async-scheduling` is what turns it off. Two hours of "the adaptive
    block isn't working" was this.

20. **The DFlash draft pass is a captured CUDA graph, so its Python runs once.**
    `DFlashSpeculator._generate_draft` — everything the speculator does per step, including
    the lookup — is replayed from a graph. The Triton kernels inside it do run every step and
    do read live buffers, so the lookup itself works; but host-side Python in there executes
    at *capture* time only. A counter, a pinned copy of a flag, a decision computed there is
    frozen at whatever the warm-up produced, silently. Anything the host must see per step
    belongs in a method the model runner calls per step (`next_num_draft_tokens`), reading
    device tensors the replayed kernels wrote. Three separate "the trigger doesn't fire"
    debugging rounds were this.

21. **`torch.cuda.is_current_stream_capturing()` is not a usable guard on this path.** It
    reads True inside the captured draft pass — which is correct, and exactly why a guard
    written as `if not is_current_stream_capturing():` silently disables the code it guards
    for the entire run, not just during warm-up.

23. **A shorter draft block than `num_speculative_tokens` loses the decode CUDA graphs.**
    The V2 runner captures uniform-decode graphs at `decode_query_len = num_speculative_tokens
    + 1` and dispatch requires an exact match, so scheduling the drafter's 8-token block on a
    16-token server matches nothing and the step runs piecewise: 27.9 ms against 25.9 ms for
    the same work, on every short step. `cudagraph_utils.py` already knows how to capture
    several decode lengths (it does it for dynamic speculative decoding); the lookup patch
    adds the drafter's block to that list. Costs 1.8 GiB of graphs instead of 1.45.

26. **The draft model is not redundant during a copy, even when the lookup overwrites every
    token it proposed.** It looks like free money: on a step the lookup controller selected,
    a qualifying match is long enough to take the head of the block too, so all seven of the
    drafter's tokens are replaced before anything is verified — skip its forward and save
    ~3 ms of a 39 ms step. Measured, that trade loses: 15.21 tokens per step becomes 13.79
    for a 5% cheaper step, a net 6% down. The drafter is covering the positions *past the
    end of the match*, which is exactly where a copy lands when the text it is reproducing
    diverges. Restricting the skip to steps where the match reaches the end of the block
    recovers the acceptance but only two runs in three — the flag it keys on is one step
    stale, and a stricter condition is more sensitive to that. Both variants are gone; this
    entry is here so the idea does not look untried.

27. **Any controller state that outlives one step has to be per-request, or batch > 1 stops
    being reproducible.** The lookup's block-length decision is batch-wide by design — a long
    block costs step time on every request in the batch — and taking it from the current
    step's flags is fine, because those are a function of the requests present. Holding it
    across steps is not: `VLLM_DFLASH2_LOOKUP_STICKY` keeps the long block on through steps
    where the flags say no, so with several requests in flight the block length a copying
    request gets depends on when the *others* arrived, and the block is one chunk through the
    recurrent layers, so that changes its greedy text. `bench/labd_soak.py` caught a verbatim
    copy coming out differently in two rounds of an identical four-way batch, and OK in three
    of three with the hold off. It is now applied only with one request in flight. The proper
    fix is per-request draft counts, which `get_uniform_token_count` in
    `gpu/cudagraph_utils.py` will not dispatch a graph for — a ragged batch runs piecewise
    and costs 8%, more than the hold is worth.
