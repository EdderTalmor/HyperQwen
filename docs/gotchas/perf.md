# Gotchas: profiling, KV pool, and benchmark methodology

How the pool gets sized, what moves it between boots, and how to read (and not fool) a benchmark.

Entries 2, 4, 6, 7, 9, 10, 11, 16, 24, 25, 28, 29, 30, 31, 34, 40, 42, 43, 48, 49, 50, 51, 52, 57 of [docs/gotchas.md](../gotchas.md) (original numbers kept, so `gotcha N` references still land).

[← gotchas index](../gotchas.md) · [quickstart](../quickstart.md)

2. **The KV pool is sized by one profiling pass, and that pass measures the
   machine as much as the model.** Two ways it goes wrong, both silent, both
   leaving a server that runs fine and is merely quietly worse for its whole
   life.
   **A dirty GPU.** vLLM profiles free memory once at startup, so if the
   previous process is still releasing VRAM at that moment the pool comes out
   ~40% smaller and stays that way. The systemd units in both mode dirs carry
   an `ExecStartPre` gate that waits for the GPU to actually be free.
   **A cold torch.compile cache.** The profiling forward also runs inductor's
   autotuning, which inflates the peak it measures: batch mode profiles a
   1.96 GiB activation peak instead of 1.09 GiB and comes up with 196k KV
   tokens instead of 224k (`Maximum concurrency ... 1.31x` in the log instead
   of 1.49x). Restart once after the cache is warm (venv: `~/.cache/vllm`,
   Docker: the `qwen-cache` volume). Gotcha 48 has this isolated to the
   byte on the single-user path — 0.92 GiB of peak, 45,000 tokens of context —
   and the reproducibility rule that follows from it.
   The durable fix for both is to stop profiling: pin the pool in bytes with
   `--kv-cache-memory` (`KV_MEM`), which is what the single-user profiles do.
   (The WSL2 notes above pin it the other way round — record the cold-start
   `--kv-cache-memory` recommendation and pass it via `EXTRA_ARGS` — if you
   prefer the extra transient headroom to the extra KV pages.)

4. **With MTP enabled, even that isn't enough — single-user mode runs
   `gpu-memory-utilization 0.93`.** The speculative decode path's DeltaNet
   workspace grows beyond what vLLM's startup memory profiling measures, and
   the engine dies mid-request on long generations at 0.95+. It survives short
   benchmarks, which is exactly how it fools you. We soak-tested 0.93 with a
   100k-token prompt plus 6k-token generations at 4 concurrent.

6. **Random-token benchmarks are meaningless for speculative decoding.** The same
   server does 35, 83 or 151 tok/s on `--dataset-name random` depending on what
   the noise turns into, because acceptance depends entirely on whether the
   drafter can guess it. Use real prompts (`--dataset-name custom`). (This is our
   own measurement, not a description of anyone else's harness — ninfer-3090's
   published cohorts use short real prompts, not random tokens.)

7. **Bigger prefill chunks make things worse.** `--max-num-batched-tokens
   8192` inflates the profiled activation peak, which shrinks the cache pool,
   which caps concurrency. 2048 wins on this card.

9. **`prompt_logprobs` on long prompts OOMs the engine at 0.972 utilization**
    (a 300-token prompt needs ~300 MB of fp32 logits and there is no headroom).
    Run quality checks at 0.93.

10. **Don't chase the DeltaNet kernels.** `bench/tune_gdn.py` microbenchmarks
    the decode kernel across block/warp configs: it already runs at ~85% of the
    3090's memory bandwidth and every variant lands within 3%. The state dtype is the
    lever, not the kernel: `--mamba-ssm-cache-dtype float16`, which both start
    scripts pass. (This used to cite a numbered entry that no longer exists.)

11. **The draft vocabulary is the single-user ceiling.** A draft head can only
    propose tokens in its id list; a miss is a certain rejection that also ends
    the chain. Count the list over the model's *own* outputs (`drafter/gen_data.py`,
    then the frequency step in `prepare/build_draft_vocab.py`), not over web text —
    92% vs 97.5% coverage was the difference between 98 and 109 tok/s greedy.
    Coverage saturates around 40k rows; the model only ever emits ~54k distinct
    tokens.

16. **`INT8_LAYERS=.` needs `GPU_UTIL=0.95`.** Quantizing the activations of every linear
    layer (rather than just the MLP) is worth ~11% throughput — 1,042 vs 942 tok/s at 64
    concurrent — but the extra per-layer scratch no longer fits batch mode's 0.972: the
    engine dies with `torch.OutOfMemoryError` inside `chunk_fwd_o` once ~17 requests are
    resident, which reads as every request returning 500 while `/health` still answers.

24. **A verify block costs step time in steps, not smoothly, and the two stairs are at 16
    and 21 query tokens.** Measured on a copy at 25k context: 39.5 ms per step at 16 query
    tokens (`DFLASH_TOKENS=15`), 47.8 at 19, 47.2 at 21 — a jump between 16 and 19 and then
    flat. The first stair is the target's W4A16 GEMMs: GPTQ-Marlin tiles the M dimension in
    16 rows (`m_block_size = 16 * thread_m_blocks`, `thread_m_blocks = div_ceil(prob_m,
    16)`), so a 17th query token buys a second M block in all 64 layers and the tokens up to
    32 are then free. The second is the verify attention: `SpecDecodeAttention._plan`
    (patches/spec-decode-attn.patch) puts `q_len * G` rows in a 128-row tile, so with this
    model's `G = 24/4 = 6` one tile holds `128 // 6 = 21` query tokens and a 22nd re-reads
    the request's whole KV segment (250/583/1132 us per layer at 8/16/32).
    So there are exactly two sensible block lengths — 16 query tokens, the last one on the
    bottom stair, and 21, the most tokens obtainable for the price of the second. 31 pays
    both stairs and was never worth measuring; two attempts to start it died on memory
    first.

25. **A verify block that outgrows its CUDA-graph reservation OOMs at run time, not at
    startup.** `--kv-cache-memory` pins the pool, so `VLLM_V2_CUDAGRAPH_MEM_MIB` no longer
    sizes it — it only reserves headroom, and if it under-reserves, the server starts, logs a
    healthy pool, and then dies on the first prefill with 50 MiB left. Graph memory grows
    with the block: measured 1.82 GiB at `DFLASH_TOKENS=15`, 2.12 at 18, 2.27 at 20 (the
    capture list length barely matters — 2.21 GiB at 20 with `CG` cut from 63 to 42). Budget
    a request as `64 KiB * context + 102 MiB * (DFLASH_TOKENS + 2)`, the second term being
    the aligned recurrent-state pages, and take the extra graph memory out of the pool.

28. **Halving the KV element size can *cost* memory on a hybrid model with a draft model.**
    `unify_kv_cache_spec_page_size` equalizes page sizes by scaling a layer's block size up by
    the integer ratio `max_page / own_page`, and pads the *page* instead when that ratio is not
    an integer. Sliding-window layers are born at the backend's smallest kernel block — 16 —
    precisely because the code picking it assumes unify will scale it up
    (`_largest_kernel_block_within` in `model_executor/layers/attention/attention.py`: "the
    smallest block is fine — `unify` scales it up by an integer ratio"). When the ratio is not
    an integer that assumption fails silently and every block of that layer pays a whole
    primary page. Divisibility here holds at bf16 only by coincidence — the target's 4 KV heads
    × 256 and the DFlash2 drafter's 8 × 128 both come to 4096 B per token per layer — and
    `int8_per_token_head` breaks it by adding one fp32 scale *per head* (2080 vs 2112 B/token;
    2112 = 2⁶·3·11 shares no factor with the primary page). The drafter's 5 layers then took
    `cdiv(2047 + 4096, 16) + 1 = 385` blocks of 1.71 MiB at 1.88% utilisation — a constant
    5.155 GiB, 75.6% of the per-request budget. Measured: int8 needed **6.82 GiB to serve
    32,768 tokens** where bf16 serves 69,758 in 5.2 GiB, i.e. 2.4× worse from halving the
    dtype. `patches/hybrid-sw-block-promote.patch` rounds such a layer's block *up* instead
    (16 → 864), which turns that into 138,696 tokens. The tell in a log is an "estimated
    maximum model length" that is a small multiple of 16.

29. **The aligned recurrent-state pages scale with the verify block, not with the slot count.**
    Not per request slot, which is the natural guess and is wrong. Measured by asking for an impossible
    `max_model_len` and fitting the two numbers vLLM prints: the fixed term is 0.88 GiB at
    `DFLASH_TOKENS=7` and 1.66 GiB at 15 — the ratio 0.53 is exactly 9/17, i.e. `(k+2)` — while
    `MAX_SEQS` 1 against 8 moves it by about **8 MiB in total**. So dropping to one slot for a
    genuinely single-user server buys no context at all, and `MAX_SEQS=4` at a long block is
    about CUDA graph memory, not state pages.

    That is about the SIZE of the pool. It says nothing about how much of the pool a
    *running* request takes, and there the per-request model is right — it is the same
    page, and the two arrive at it independently (0.88 GiB fitted here, ~0.82 GiB from
    the live occupancy below). Measured live at `CTX=fast` (`bench/conc_ladder.py`, and the ramp in the
    issue-25 notes): one resident `dflash2` request with an empty context occupies
    **15.8%** of the 69,758-token pool, so six or seven fit and the next one is
    **preempted**; with 4k-token prompts it is 19.8% and five fit, with 16k-token
    prompts two. One MTP request takes 8.2% of its
    86,727, so eight fit. Both numbers are the k+1 recurrent-state slots, which is why
    they are in the ratio 8:5. The two facts together are the whole of the concurrency
    story for this mode: extra seats do not cost you pool, and they do not buy you
    residents either.

30. **Asking for an impossible `max_model_len` is the cheapest way to read the memory model.**
    vLLM prints "X GiB KV cache is needed ... available Y GiB ... estimated maximum model
    length is Z" and dies in ~90 s, before torch.compile finishes and long before graph
    capture. Two such points give slope and intercept for `needed(context)`, and the slope
    comes out at exactly `16 × 4 × 256 × 2 × 2 = 65,536` B/token for bf16 — so the fit can be
    checked against arithmetic rather than trusted. Beware that `estimate_max_model_len` is a
    binary search over `max_memory_usage_bytes`, which rounds up to whole blocks, so the
    estimate is quantised by the block size: at an 864-token block the granularity is coarse
    and a two-point inversion at small lengths is unreliable.

31. **`KV_MEM` assumes the card is headless, and the failure lands long after
    startup looks fine.** The single-user pool is pinned in bytes rather than sized
    from `GPU_UTIL` (gotcha 29 and the comment in `single-user/start_qwen.sh` say
    why), and 5.2 GiB is what fits when nothing else is on the GPU. With a desktop
    session on the same card — Xorg plus a compositor plus a browser is easily
    ~1.3 GiB — the server still starts, still captures its graphs, still reports a
    pool, and then dies later on a real request when the spec-decode `part_o`
    buffer cannot get its ~1.5 GiB (`spec_decode_attn.py`). Nothing at startup
    warns you. On a card you also render on, drop `KV_MEM` by at least what the
    desktop is holding (`nvidia-smi` before you start the server): `KV_MEM=4000000000`
    was enough for the reporter of
    [#12](https://github.com/syv-ai/HyperQwen/pull/12). Setting `KV_MEM=`
    empty falls back to `GPU_UTIL`, which profiles the actual free memory instead.

34. **The decode-graph budget is sized for 64 query tokens, and `MAX_SEQS` multiplies
    into it.** `CG = MAX_SEQS x (k+1)` is what the V2 runner captures, and
    `VLLM_V2_CUDAGRAPH_MEM_MIB` is reserved for what the shipped defaults produce —
    8x8 at `DFLASH_TOKENS=7`, 4x16 at 15, i.e. 64 either way. Ask for
    `DFLASH_TOKENS=15 MAX_SEQS=8` and it becomes 128: the server boots, captures its
    graphs, answers `/health`, and then dies on the first concurrent batch with
    `torch.OutOfMemoryError` inside the engine — `EngineDeadError`, every request 500,
    `/health` still 200. Same shape as gotcha 15 and as
    [#18](https://github.com/syv-ai/HyperQwen/issues/18): a memory bill that
    the startup profile does not see. `single-user/start_qwen.sh` now caps the derived
    `CG` at 64, which leaves every shipped default untouched and makes the oversized
    batches run piecewise instead of not at all. Set `CG` explicitly to override, and
    raise `VLLM_V2_CUDAGRAPH_MEM_MIB` with it.

40. **`CTX=long`'s fp8 KV cache has exactly one attention backend on sm86, and
    it is the one cell of the matrix this repo cannot A/B.** `FLASH_ATTN`
    refuses fp8 KV at startup ("requires FA3 on SM90 or FA4 on SM100" —
    [#34](https://github.com/syv-ai/HyperQwen/issues/34)) and
    `TRITON_ATTN` refuses it too ("native FP8 (fp8e4nv) requires SM89+",
    measured on the reference 3090), so the tier always auto-selects
    FlashInfer. #34 tracks a deterministic Xid-31 MMU write-fault (same
    virtual address twice, ~40 h uptime each) on that combination with MTP +
    prefix caching + chunked prefill at 28-34k context; cause unattributed
    between flashinfer's workspace and the async-scheduling window as of this
    entry. If you hit it, the flashinfer-free fallback is the int8 tier:
    `SPEC=dflash2 CTX=long` ships it by default, and for `SPEC=mtp` it is
    `VLLM_SPEC_DECODE_ATTN=1 EXTRA_ARGS="--attention-backend=TRITON_ATTN
    --kv-cache-dtype=int8_per_token_head"`. Measured cost on the reference
    box: 17.9k in + 256 out takes 23.7 s against fp8/FlashInfer's 18.9
    (~25% wall at that depth, mostly Triton prefill). The default stays
    fp8/FlashInfer: two faults on one box do not justify that tax on every
    other box, but you should know which combination you are running.

    **What the escape actually buys and costs, measured** (reference 3090,
    250 W, vLLM 0.28.0, `SPEC=mtp CTX=long PREFIX_CACHE=1 GPU_UTIL=0.93
    MAX_SEQS=4 MAX_LEN=120000`, warm medians of 3 on `bench/real_rep.sh`,
    C1 = 8 realistic prompts x 1024 out, concurrency 1):

    | arm | decode tok/s | ms/step | tok/step | KV pool | GSM8K n=200 | PPL all |
    |---|---|---|---|---|---|---|
    | stock fp8 / FlashInfer / PIECEWISE | 101.1 | 27.2 | 2.68 | 172,500 | 0.960 | 8.2362 |
    | int8 / TRITON_ATTN / FULL graphs | 103.7 | 25.4 | 2.57 | 145,030 | 0.965 | 8.2375 |

    Three things to take from that. (a) The win at chat length is **+2.5%
    end-to-end**, not the +6% a step-time number alone suggests: dropping the
    spec-decode CUDA-graph downgrade really is worth −6.6% step time
    (27.2 → 25.4 ms), but int8 KV gives ~4% of it straight back in MTP
    acceptance (2.68 → 2.57 tok/step). Quote step time as step time.
    (b) It is **quality-neutral** — GSM8K 96.5 vs 96.0 (n=200, SE ≈ 1.3 pt,
    i.e. indistinguishable) and PPL +0.02%. Unlike int8 *activations*, int8
    KV costs no accuracy here. (c) It **costs pool, it does not save it**:
    145,030 vs 172,500 tokens, −15.9% at this geometry (per-token-head
    scales plus the FULL-decode-graph capture), the opposite sign of what an
    earlier version of this entry claimed.

    **Do not reach for it as the fast path.** The same C1 row under
    `SPEC=dflash2 CTX=fast` (FLASH_ATTN, bf16 KV, FULL graphs, 68,605-token
    pool) reads 135.5 decode tok/s / 3.49 tok/step at the same 26.6 ms/step
    — **+31% over the int8 escape**, all of it acceptance. If your work fits
    64k, that is the answer; the int8 tier is the #34 fallback and the
    `SPEC=mtp` + depth route, not a performance recommendation.

    **And it decays with depth.** Salted prompts (`cached_tokens = 0`), 256
    out, greedy, stock vs int8 escape: equal at 8K (81.7 / 82.9 decode),
    −22% decode at 25K (80.3 / 62.7), −34% decode and −44% fresh prefill at
    60K (70.7 / 46.7 and 872 / 490 tok/s), with TTFT 68.9 → 122.6 s at 60K.
    A WSL2 3090 contributor carried the same curve to 90K (stock ~80 decode
    / 878 prefill, escape 46.3 / 355) while stock stayed flat from 25K out.
    So: **stock fp8 FlashInfer for depth, the int8 escape only for
    short-context decode-heavy work or when #34 forces it.**

    **On sm89+ you do not have to give up the fp8 pool.** 0.28's
    `_create_draft_vllm_config` deliberately does not inherit the target's
    attention backend into the proposer, so the knob is the *speculative
    config's own* `attention_backend` field — put TRITON_ATTN there, leave
    the target on fp8/FlashInfer, and the downgrade goes away with the pool
    intact. Unreachable on sm86: Triton refuses fp8 KV below SM89
    ("native FP8 (fp8e4nv) requires SM89+"), so int8 is the only
    flashinfer-free option on a 3090. Reported and verified on a 4090 by a
    contributor ([#87](https://github.com/syv-ai/HyperQwen/issues/87)).

42. **`vllm bench serve` defaults to `--seed 0`, and with prefix caching that
    poisons every A/B.** Same seed = same prompts call to call; later calls
    get partial prefix-cache hits whose size depends on the arm's pool
    geometry, so the contamination differs *between the configs you are
    comparing*. Measured: a 16k prefill "at" 4.0 s that cold costs 11.2 s
    (spec-off arm, big pool) next to a baseline reading 15-20% low. Every
    random-dataset call needs its own `--seed`; `bench/run_benchmarks.sh`
    does this now (`SEED_BASE` pins the sequence).

43. **`--max-num-batched-tokens` above 2048 costs pool, and 8192 does not boot.**
    Re-measured on vLLM 0.28.0, `SPEC=dflash2 CTX=fast`, pinned `KV_MEM`, three
    cold boots on the reference 3090:

    | `--max-num-batched-tokens` | result | pool |
    |---|---|---:|
    | 2048 (default) | boots | 68,605 tokens, 1.05x |
    | 4096 | **boots** | 66,945 tokens, 1.02x |
    | 8192 | refuses | — |

    Two corrections to what this entry used to say. **4096 boots** — it costs
    2.4% of the pool and takes the concurrency margin from 1.05x to 1.02x, which
    is a trade rather than a wall. And the mechanism is not "inflates the
    profiled activation peak": with `KV_MEM` pinned the engine skips memory
    profiling entirely and says so in the log. What the bigger chunk grows is
    the per-request KV requirement, so 8192 fails as a clean startup refusal —
    `5.35 GiB KV cache is needed, which is larger than the available KV cache
    memory (5.2 GiB) ... estimated maximum model length is 63168` — not as a
    mid-init OOM. Batch mode documents the softer version of the same thing:
    unpinned, bigger chunks shrink the pool instead of refusing.

48. **A benchmark row without its compile-cache state is not reproducible, because the
    autotuner's timing race picks the kernels and the kernels pick the trajectory.**
    Filed as [#75](https://github.com/syv-ai/HyperQwen/issues/75). Six Triton
    kernels in the chunked Gated DeltaNet path are autotuned at first use; vLLM caches the
    winners, but a fresh container or `VLLM_DISABLE_COMPILE_CACHE=1` re-runs the race, and a
    different winner is a different reduction order, a different last bit, and at greedy a
    different token at the first near tie. Measured on one image with the cache wiped before
    each boot: 8 boots, 8 distinct winner sets, 5 distinct trajectories, the same twelve-turn
    tool conversation ranging from 58 to 189 tool calls. With the cache persisted, two boots
    were identical to the call. So: mount a persistent cache into bench containers rather than
    disabling it for hygiene, and copy the `*.autotune.json` records beside a published row so
    a reader can tell whether two rows are even comparable. A quiet bare box re-times to the
    same winners even with the cache deleted; container timing noise is what makes the draw
    vary. The same cache state also moves the profiled peak activation and therefore the KV
    pool, by 0.92 GiB on the reference 3090, which is a separate channel with a separate fix
    (state it, or pin `--kv-cache-memory`).

49. **A high acceptance rate can mean the drafter is good or the generation has collapsed,
    and the counter cannot tell you which.** Degenerate text is trivially predictable. One
    repetition loop during the [#73](https://github.com/syv-ai/HyperQwen/issues/73)
    work scored 79.7% acceptance per drafted token against a normal 35 to 45%, with a
    distinct-word ratio of 0.051 against 0.77 to 0.83, and it was echoing the instruction
    appended to its own prompt. So acceptance is heavy-tailed, a t-test over a dozen short
    generations is the wrong instrument, and an arm that happens to sample more collapsed
    trajectories looks like an acceptance *improvement*. Guard on drafted tokens per round
    above about 1.2 times the drafter width (calibrated on 96 rows: above the width flags 40%
    of ordinary rows, above 1.2 times it flags only the real events), or on distinct-word
    ratio when the text is available. The guard reads the engine's own repetition signal,
    because adaptive verify length extends the block only while a request is reproducing its
    context. Run the guard **before** any stratification on block size: a degenerate row sorts
    into the long-block stratum by construction and one such row moved a stratum estimate by
    two tokens per step.

50. **Per-request acceptance figures depend on what ran before the request, and on some boxes the
    request's whole trajectory does.** Two observations, two boxes. On a quiet native 3090, same
    build, same seeds, one boot, only the request order changed: drafts and accepted tokens came back
    identical on every seed and drafted tokens did not, so `1 + accepted / drafts` repeated exactly
    while `accepted / drafted` had an order-dependent denominator. On an RTX 4090 under WSL2 with a
    prompt that keeps the long block engaged, the same design changed drafts and accepted tokens too
    on four of six seeds, one seed by a factor of two, so there tokens per step itself moved with
    order. Two mechanisms, both real: the long block is sticky and coasts on prior state without
    consulting the emitted count, so a request that follows a long-block request inherits some of its
    block length; and a different block length is a different verify batch shape, which on a
    numerically knife-edged model can flip a near-tie token even with the seed fixed. Prefix caching
    was the obvious third candidate, since the requests share a prompt; with the engine started
    `--no-enable-prefix-caching` and the log confirming it, the order effect was unchanged, so it
    is not the driver. What follows is the same either way: hold the request
    order fixed within a comparison, report tokens per step rather than acceptance per drafted token,
    and treat per-request figures from a sequence as dependent samples, never as independent ones.

51. **At `DFLASH_TOKENS=15` on ordinary text the engine drafts 7 and queries 8, so 15 and 7 are
    the same experiment unless the text repeats.** With the lookup on, the drafter's block is
    clamped to the checkpoint's trained block (7), `num_query_per_req` follows it (8), and
    adaptive verify length asks for the long block only while a request is reproducing its
    context (`dflash2/speculator.py`, on by default). Measured over sixty single-prompt rows
    on the reference 3090: 55% at exactly 7.000 drafted tokens per round, and the lookup
    supplying 4.3% of the eight positions it could fill. On an eight-prompt cohort at 1024
    output tokens the long block engaged on 44% of rows and those rows ran markedly faster.
    Consequences: matching results at both widths are weak evidence that an effect is not
    lookup-specific; anything that acts only in the long block, including
    `dflash2-z-adaptive-emitted.patch`, is invisible on a short single-prompt cell and only
    shows on a cohort; and `VLLM_DFLASH2_LOOKUP_CHEAP_CTX` (default 0, so the branch is dead)
    takes the long block unconditionally below a context threshold and would invalidate any
    bisect run across it. The same split shows in any production log without
    instrumentation: at width 15 the engine's own per-position acceptance line reads seven
    positions in the 0.79-0.96 band and a flat eight-position tail near 0.15 (0.958, 0.921,
    0.899, 0.862, 0.845, 0.820, 0.793, then 0.155 x 6, 0.153, 0.148 -- reference 3090 under
    real traffic).

52. **Two passes with a fixed seed are two replays, and a short single-prompt cell cannot see a
    cohort-scale effect at any number of seeds.** The engine seeds from zero and the noise draw
    is a function of seed and position, so repeat boots on a quiet box come back bit-identical
    on every counter, and "reproducible to three significant figures from two passes" measures
    a deterministic harness rather than bounding an effect. The 16% DFlash2 acceptance
    regression in [#73](https://github.com/syv-ai/HyperQwen/issues/73) was invisible
    on one README prompt at 256 output tokens with six seeds *and* with thirty (seed spread
    ten points, sd about four), and reproduced immediately through `bench/prefill_ab.sh` on the
    cohort at 1024. Prompt choice alone moved acceptance from 22.7% on the cohort to 36.8% on a
    236-character prompt, larger than the regression. Three different questions were asked of
    that short cell during the bisect and it was underpowered for all of them. Match the
    workload to the claim, vary the seed per request, and read a bare number from a fixed-seed
    pass as one draw.

57. **The fp8 split-KV verify route is sm89 and up, and `INT8_ACT=int8` does not
    stack on it.** `patches/triton-spec-attn-fp8-kv.patch` lets the split-KV
    verify kernel read vLLM's per-tensor fp8 cache, so `--kv-cache-dtype fp8`
    can run with `TRITON_ATTN` on both the target and the drafter and keep FULL
    CUDA graphs (the launch line, verbatim, is in `docs/long-context.md`; the
    `"attention_backend":"TRITON_ATTN"` inside the speculative config is the
    part that is easy to drop, and dropping it silently costs the FULL graphs).
    Two limits. Triton has no fp8e4nv conversion on sm86, so on a 3090 the
    route does not exist: the compiler refuses the kernel outright (gotcha 40
    has the backend map), and `bench/test_spec_decode_fp8.py` skips there by
    design rather than dying. And `INT8_ACT=int8` on the fp8 route is slow with
    or without this patch (a Marlin variant choice, tracked separately), so do
    not stack the two until the memory-pressure question behind it is
    understood; the measured rows in `docs/long-context.md` are fp8 KV with
    bf16 activations.
