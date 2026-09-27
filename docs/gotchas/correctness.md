# Gotchas: correctness traps

Stale caches, silent misconfigurations, and bugs that look like something else.

Entries 1, 5, 13, 14, 22, 32, 33, 36, 46, 47, 54, 56, 58 of [docs/gotchas.md](../gotchas.md) (original numbers kept, so `gotcha N` references still land).

[← gotchas index](../gotchas.md) · [quickstart](../quickstart.md)

1. **A benchmark cannot tell you the output is garbage.** The int8-activation
   path served nonsense for an hour of beautiful throughput numbers before a
   perplexity check caught it. Whatever you change, run
   `bench/quality_battery.py` (perplexity + GSM8K against the live server)
   before you believe a tok/s number.

5. **A stale compiled artifact replays a graph built for other shapes, and the
   error never mentions the cache.** Three doors into the same room:
   - **Env vars vLLM does not know about.** Switching `INT8_LAYERS` between runs
     replays a graph expecting the other layer set and dies with `KeyError:
     'input_global_scale'`. Our patch registers the selection env vars with vLLM
     so they become part of the cache key; if you invent your own,
     `VLLM_DISABLE_COMPILE_CACHE=1`.
   - **Shapes baked into the graph.** The compiled graph bakes in e.g. the
     Marlin workspace size, so a new knob that changes it must be registered in
     `envs.py` (`patches/speed-knobs-envs.patch`) or you get `assert_size_stride
     ... expected size 328==82` from a cached artifact.
   - **A second cache vLLM does not control.** The layer-select envs *are* in
     vLLM's compile hash, but `torch_aot_compile` keeps its own; changing
     `VLLM_MARLIN_INT8_INCLUDE_RE` can crash at the first forward with a
     stable-ABI `aten::empty` RuntimeError from a cached inductor artifact. Wipe
     `~/.cache/vllm/torch_compile_cache` when switching layer sets.

   Unrelated to caching but found the same way: `INT8_LAYERS="mlp|linear_attn"`
   (int8 GDN + fp16 attention) crashes even from a clean cache — an inductor
   codegen bug with that mixed set on this torch pin; `mlp` and the full default
   both compile fine.

13. **Greedy is not deterministic across drafter configs.** The target rounds
    differently when it verifies 5 tokens vs 1, so a different drafter changes
    the generated text at near-ties and the 8-prompt acceptance numbers move
    ±3%. Repeat before trusting a small difference; `drafter/README.md` has an
    offline chain simulator that removes the noise.

14. **vLLM picks the speculative method from the model *path*.** `"dflash" in
    model_path` switches `method` to dflash — for the *target* too, since MTP
    uses the target path as its draft model. A checkout under a directory with
    "dflash" in its name turns `SPEC=mtp` into a crash in `EAGLEConfig`
    (`'Qwen3_5Config' object has no attribute 'vocab_size'`). Name your
    directories accordingly.

22. **rsync preserves mtimes, and Python trusts mtimes.** Copying a source file into
    `site-packages` with `rsync -a` can leave the `.pyc` newer than the `.py`, in which case
    the interpreter keeps running the old bytecode and every measurement lands on the
    previous revision. Delete `__pycache__` after installing patched files.

32. **A model dir with no `tokenizer.json` is not an error to transformers — it is an
    empty vocabulary, and vLLM reports it as a reasoning-parser problem.**
    `AutoTokenizer.from_pretrained` on a dir that has `config.json` but no tokenizer
    files returns a `Qwen2Tokenizer` with `vocab_size == 1` that encodes *everything*
    to `[]` — `tok.encode("hello world")` is `[]`, not an exception. Nothing complains
    until `VllmConfig.__post_init__` asks the qwen3 reasoning parser for `<think>`,
    gets `[]` back, and raises

    ```
    ReasoningConfig: failed to tokenize reasoning strings:
    reasoning_start_str='', reasoning_end_str=''.
    ```

    which names neither the tokenizer nor the directory, and prints the strings as
    empty because they are the *unset* config fields, not the ones the parser supplied.
    Reported as [#15](https://github.com/syv-ai/HyperQwen/issues/15), where it
    looked like a `SPEC=dflash2` bug: it reproduces with no speculative config at all,
    and the reason only the single-user modes failed is that they serve
    `models/Qwen3.8-27B-W4A16-AutoRound-fast` while batch mode serves the base dir.
    `verify.sh` now encodes `<think>` against every dir we pass to `--model` instead of
    only checking that the dir exists, and `docker/prepare.sh` counts `tokenizer.json`
    as part of a complete download.

33. **Bug B needs a prefix-cache HIT, and then fires at one prompt length in every
    128. It is not dflash2-only.**
    Under `CTX=huge` with a CAPTURED (FULL) verify step, a request that hits the
    prefix cache and whose prompt length lands on one particular residue mod 128
    collapses: `SPEC=dflash2 DFLASH_TOKENS=7` gives 1.97 tok/step and degenerate
    repetition (`4/3595` characters verbatim, one 40-char block ×79), `SPEC=mtp`
    stops dead and returns `""` or `"#"` with `finish_reason=stop`. Every other
    residue is 794/794 verbatim.

    **The location is deterministic; the damage is not.** Repeats are bit-identical
    on one server, but the same `mtp` residue has now produced three different
    outputs on three geometries: an empty answer, a one-character answer, and — on a
    box running `MAX_LEN=240000` with a tool parser attached — 400 tokens of fluent
    Danish that open with a malformed `<think>` under `enable_thinking=false` and
    invent a translation task, `2/1146` verbatim at 3.38 tok/step
    ([#25](https://github.com/syv-ai/HyperQwen/issues/25), mjungnickel18).
    So do not test for a symptom. The only property all three share is that the copy
    did not come back, which is what `bench/verbatim.py` scores and what both sweeps
    now judge on.

    Two conditions, and it took two people to see both. The **hit** is necessary:
    a fresh server, one request, no warm-up, never collapses at any length
    ([#13](https://github.com/syv-ai/HyperQwen/pull/13), mjungnickel18) —
    which is also why `PREFIX_CACHE=0` always looked clean. The **residue** decides
    whether a hit corrupts, and it is a clean function of the draft count:

    | config | k | verify block L=k+1 | attention block | broken R | free slots 128-R |
    |---|---|---|---|---|---|
    | `dflash2`, `DFLASH_TOKENS=7` | 7 | 8 | 2176 (=17x128) | 124 | 4 |
    | `dflash2`, `DFLASH_TOKENS=5` | 5 | 6 | **2176** | **122** | 6 |
    | `dflash2`, `DFLASH_TOKENS=3` | 3 | 4 | 2048 (=16x128) | 120 | 8 |
    | `mtp`, `DRAFT_TOKENS=3` | 3 | 4 | 2048 | 120 | 8 |

    **R = 117 + k**, equivalently the final 128-token tile has exactly `11 - k`
    free slots, equivalently `L + free = 12` in every configuration measured.
    Fitted on three values of k across two speculators, so treat the constant as a
    fit rather than a derivation — but note what it implies: the step reserves or
    touches a **fixed 12 slots regardless of the verify block length**, which is
    the most specific lead this bug has produced.

    The `DFLASH_TOKENS=5` row is the one that matters for method. Same attention
    block as `DFLASH_TOKENS=7` (2176), different broken residue (122 against 124),
    which rules out the attention block size. An earlier version of this entry
    claimed R tracked the verify block on three points where the two co-varied;
    that was retracted as unevidenced, and then confirmed by running the
    configuration that separates them.

    Confirmed periodic in every case: 24,956 / 25,084 / 25,212 / 25,340 at k=7,
    25,082 / 25,210 / 25,338 at k=5, 25,080 / 25,208 / 25,336 at k=3. Hold the document
    byte-identical and pad the *instruction* by one token and a broken length goes
    clean, so it is the token count rather than the corpus. What is established: `mtp` and
    `dflash2` at the same draft count break at the same residue, so the drafter is
    not implicated and the shared multi-query verify against a partially-hit
    prefix is.

    Mitigation, as it stands at HEAD: `CTX=huge` forces `cudagraph_mode=PIECEWISE`
    for `SPEC=mtp`, which is clean at every residue and costs nothing measurable —
    `SPEC=mtp` over 8k/16k/32k/50k is 87.8/86.1/70.4/63.5 tok/s captured against
    93.5/83.8/70.3/59.6 piecewise. This repo previously scoped that workaround to
    `dflash2` on the theory that MTP's short verify step captures correctly; it
    does not, and `SPEC=mtp CTX=huge` shipped with the bug.

    `dflash2` has since got FULL capture back (`a75ee4b` fixed its residue, and
    `b356e31` swept **all 128** residues under FULL with 0 broken), so the two
    speculators are no longer on the same default. `mtp` keeps PIECEWISE as a
    correctness constraint until residue 4 comes back verbatim under a full sweep,
    not until a particular symptom stops appearing.

    A third trap, learned the hard way on `DFLASH_TOKENS=15`: `bench/bugb_sweep.py`
    used to report the RAW prompt length, not the chat-templated one the engine
    actually sees (+12 tokens for the Qwen3 wrapper). That offset is why the rule
    first read as `R = 117 + k` and then as a mysterious constant 12; both were the
    same relation seen through a harness bug. It also made a k=15 sweep look
    structureless until the offset was applied, at which point the lowest-acceptance
    row sat exactly on `== L`. The script now templates before counting.

    Do not judge a row by its failure signature, and that includes `repeats`. An
    earlier version of this entry said "only `repeats` tells you whether it actually
    collapsed"; two of the three shapes above repeat nothing, and a rule that
    demanded repetition is what filed the `mtp` break as "diverged, probably fine"
    through several full sweeps. Both sweeps now score **coverage** — the fraction of
    the answer's 40-character windows that occur in the source — against the median
    of the other lengths in the same run (`bench/verbatim.py`, which self-tests
    against all three shapes: `venv/bin/python bench/verbatim.py`). Coverage rather
    than the old longest-prefix column because a prefix match reports `38/791` for a
    single wrong character at offset 38 no matter how good the rest is; the prefix
    and repeat counts are still printed, but nothing is decided on them.

    Two traps for anyone measuring this. Sweep prompt length in steps of **1
    token** — at a coarse grid one broken sample below and one above reads as a
    cliff, which is how it was first diagnosed. And send each length to a **fresh
    server**, or request N inherits request N-1's blocks and you measure history
    instead of length; `bench/labd_bench.py` sends two warm-ups on `doc[:4000]`,
    which arms the trigger for everything after it. `bench/bugb_sweep.py` prints
    the `mod 128` column for this.

    And do not sample residues. With one broken length in 128, five distinct samples
    miss it `C(127,5)/C(128,5)` = 123/128 = **96%** of the time — this repo once
    wrote 82% there, which is the figure for six broken residues, and hung a
    "5 of 5 clean" claim on it. `bench/residue_sweep.py` walks all 128 by stepping
    the pad one token at a time, which covers each residue exactly once.

36. **Tool calling / structured output under a speculator killed requests at the
    grammar's end** ([#31](https://github.com/syv-ai/HyperQwen/issues/31),
    fixed by `patches/xgrammar-spec-terminated.patch`). A speculative verify window
    can legally accept tokens past the point where the xgrammar matcher terminates —
    the newline after a closing `</tool_call>` tag, the stop token itself, anything
    after it under `ignore_eos`. The v0.28.0 base treats both arrivals as failure, the
    scheduler logs `Unexpected: grammar rejected tokens ... Terminating request`,
    and the client gets an HTTP error for a request whose output was completely
    valid. The longer the verify block, the more reliably the window covers the
    tokens around the stop, which is why `DFLASH_TOKENS=15` + `--tool-call-parser`
    surfaced it first. Reproduced on the shipped config with a `json_schema` +
    `ignore_eos` request — `grammar rejected tokens [16, 22, 198, 92, 248046, 198]`,
    where 92 is the brace that completes the JSON, 248046 the stop token that
    terminates the matcher, and the trailing newline killed the request. The patch
    backports upstream's current semantics: tokens after termination are ignored,
    real mid-grammar rejections still fail loudly.

    Two log signatures to keep apart, because they look alike. The fatal one is the
    `grammar rejected tokens` line above — gone with the patch. The non-fatal one is
    a burst of `Failed to advance FSM for request ... Please file an issue.` with
    **no** `Terminating request` after it: that is the bitmask builder advancing
    draft tokens past a reasoning end that landed mid-window, a rejection the code
    explicitly tolerates. It is noise, the request completes normally, and it
    predates (and survives) this fix.

46. **`prompt_logprobs` is wrong on `CTX=huge` + `SPEC=mtp` + prefix caching, and
    the NaN 400s are only its visible half.** Reported as
    [#64](https://github.com/syv-ai/HyperQwen/issues/64) from a WSL2
    3090 — `bench/quality_battery.py --ppl-only` failing with
    `{"message":"Out of range float values are not JSON compliant: nan"}` on
    some documents, and perplexity drifting 3.7% between identical runs.
    Reproduced here on bare metal, on the reporter's own document indices, so
    it is not a WSL2 effect. What it actually takes is all three of KVarN, MTP
    and `PREFIX_CACHE=1`; the same 106-document battery, same checkpoint, two
    concurrent workers:

    | profile | NaN batteries | en perplexity |
    |---|---|---|
    | `CTX=huge` `SPEC=mtp` `PREFIX_CACHE=1` | 4 of 5 | 12.6-13.7, drifting |
    | `CTX=huge` `SPEC=mtp` `PREFIX_CACHE=0` | 0 of 5 | **10.7628**, identical across runs |
    | `CTX=huge` `SPEC=off` `PREFIX_CACHE=1` | 0 of 5 | 10.7646 |
    | `CTX=huge` `SPEC=dflash2` `PREFIX_CACHE=1` | 0 of 5 | 10.7643 |
    | `CTX=fast`, `off` / `mtp` / `dflash2` | 0 of 6 | 10.7614 / 10.7659 / 10.7633 |

    So the inflated perplexity and the NaN are one bug, not two: in the broken
    combination the logprobs that come back are ~23% worse on English than the
    same server produces with prefix caching off, and the requests whose
    corruption reaches a non-finite float are the ones that 400. Every clean
    configuration agrees to four decimals, which is also what makes the broken
    one unmistakable.

    Two things that look like the cause and are not. **The documents**: sent one
    at a time on a single thread, all of them return clean logprobs — it needs
    co-scheduled requests, and the failures land on adjacent index pairs, which
    under two workers are exactly the pairs that share a prefill batch. **The
    pool size**: within MTP it is geometry-dependent (failed at 299k and 312k
    tokens of pool, clean at 265k, 352k, 359k and 390k), which is what makes it
    look intermittent across boots — but `SPEC=dflash2` pinned to 312,242
    tokens, matching the failing MTP geometry to 0.02%, is clean, so the
    speculator is the variable and the geometry only decides whether it fires.
    `MAX_LEN` is not involved: 240000 and the 245760 default both fail and both
    pass depending on the rest.

    Practical rule until the read path is fixed: measure perplexity or anything
    else using `prompt_logprobs` on KVarN with `PREFIX_CACHE=0`, or on a
    non-MTP speculator. Ordinary generation is not implicated — needle
    retrieval and decode rates are normal on the same server.

47. **`DFLASH_TOKENS=15` asserted at engine start on the int4 path, because the
    drafter's promoted block only has to *cover* the primary page, not divide
    it.** Filed as
    [#63](https://github.com/syv-ai/HyperQwen/issues/63), fixed in
    `patches/hybrid-sw-block-promote.patch`. `alternative.sh`
    (`int4_per_token_head`) died in a bare `assert` in
    `kv_cache_coordinator.py` at `DFLASH_TOKENS=15` — at any `MAX_LEN`, with
    prefix caching on or off — while `DFLASH_TOKENS=7` on the identical config
    booted.

    The chain, all of it visible in the boot log:

    | | `DFLASH_TOKENS=7` | `DFLASH_TOKENS=15` |
    |---|---|---|
    | mamba page (grows with the spec-decode state) → primary block | 1696 | **1840** |
    | drafter's covering block (16 → smallest multiple whose page covers) | 848 | **928** |
    | primary / drafter | exactly 2 | 1.983 |
    | result | boots | `AssertionError` |

    The scheduler's granularity is the LCM of the primary groups' blocks and
    every group's block has to divide it. `_promote_indivisible_block_sizes`
    only guaranteed the drafter's page *covers* the maximum, and 848 divided
    1696 by luck. The fix rounds the promotion up to the smallest divisor of
    the primary block instead — 928 → 1840, after which the primary layers
    scale 1840 → 3680 through the branch they already take.

    Two things worth knowing. **It is an int4-only shape**: on
    `int8_per_token_head` (`CTX=long`) the primary block and the drafter's
    covering block come out *equal* (864 at 7, 944 at 15), so divisibility is
    free and both arms are byte-identical before and after the fix —
    `CTX=long DFLASH_TOKENS=7` still pools 138,696 tokens. **The wider verify
    block is not free on int4**: at 15 the pool is 53,908 tokens against
    142,843 at 7, and the 256k default no longer fits (`estimated maximum
    model length is 180320`, a clear `ValueError` rather than an assert).

54. **`verify.sh` reported two patches as not applied on a correctly patched
    tree, because a `find -name '*.orig' -delete` cleanup removed files the
    verifier was requiring.** `dflash2-lookup-drafting.patch` and
    `dflash2-prewarm.patch` carried file-creation hunks for three `.orig`
    backups of vLLM source (3,685 lines of copied upstream that were nobody's
    patch). `patches/_check_applied.py` builds its file list from every `+++`
    line in a patch, so those backup paths became files that had to exist in
    the installed tree. Delete the junk and the verifier calls the patch
    missing:

    ```
    dflash2-lookup-drafting: applied(0)      # patched tree, .orig present
    --- find -name '*.orig' -delete ---
    dflash2-lookup-drafting: NOT applied(1)  # same tree, unchanged code
    ```

    It cost a cross-card comparison in
    [#89](https://github.com/syv-ai/HyperQwen/pull/89), where a real
    paired result on the native 3090 was discounted as "not like-for-like"
    on the strength of that false negative. Fixed in
    [#92](https://github.com/syv-ai/HyperQwen/pull/92): the hunks are
    gone, the installed tree no longer collects them, and a box that ran the
    cleanup verifies green. The general rule is that a patch which creates a
    file makes that file part of what verification demands, so a patch should
    only create files it means to own.

56. **Registering an env var in vLLM's `envs.py` puts it in the torch.compile
    cache key, so `VLLM_INT4_MQ_3D_DEBUG=1` now recompiles from cold.**
    `patches/int4-mq3d-envs.patch`
    ([#93](https://github.com/syv-ai/HyperQwen/pull/93)) registers
    `VLLM_INT4_MQ_3D` and `VLLM_INT4_MQ_3D_DEBUG`. That is not cosmetic:
    `vllm/envs.py:compile_factors()` starts from every known vLLM env var,
    drops only the names in its `ignored_factors` set, and hashes the rest;
    `vllm/compilation/backends.py:1031-1066` folds that hash into the compile
    cache directory key. Neither new knob is in `ignored_factors`, so each
    value now selects its own compile cache.

    Two consequences. The good one: an A/B of `INT4_MQ_3D=0` against `=1` on a
    warm cache is no longer a stale-graph trap, and the per-arm
    `rm -rf ~/.cache/vllm/torch_compile_cache` that the #93 review had to do by
    hand is no longer required. The one to watch: a boot with
    `VLLM_INT4_MQ_3D_DEBUG=1` has a different cache key from production, so it
    compiles cold and must never be timed against a warm production boot.
    "Never time an instrumented boot" arrives here by a new route: the
    instrumentation does not have to be in the hot path to cost you the
    startup, it only has to be registered.

58. **`reasoning_effort: "minimal"` from an OpenAI-protocol client 400s every
    request that carries it.** The shipped `chat_template.jinja` accepts only
    xhigh/medium/low and defaults to xhigh, while gpt-5-era clients speak the
    OpenAI vocabulary (none/minimal/low/medium/high/xhigh/max). vLLM's
    `ChatCompletionRequest.reasoning_effort` accepts all seven and passes the
    value verbatim into `apply_chat_template` (`vllm/renderers/hf.py`
    `safe_apply_chat_template`), so `minimal` reaches the template's
    `raise_exception` and comes back as a 400 Bad Request. Headroom passes it
    through untouched — its effort router only rewrites Responses-API turns and
    deliberately does not run on chat/completions (`shape_openai_chat_request`,
    headroom `proxy/output_shaper.py`). First seen 2026-09-15: one 400 among
    eleven requests, session otherwise healthy.
    Fix: `prepare/translate_chat_template.py` rewrites the effort block in
    place — it maps only the names the template does not know (minimal→low,
    high/max→xhigh) and lets every other value fall through unchanged, so the
    template's own levels keep their behaviour and an omitted effort keeps the
    template default (`xhigh`): no measured baseline moves. The raise is
    dropped, so an unknown value no longer 400s — it ends up with no reasoning
    instruction, the same outcome as `medium`. Idempotent (v2 marker, with a
    v1→v2 upgrade so a dir translated by the first cut does not keep its
    `medium` default) and self-healing: `docker/prepare.sh` re-runs it on every
    boot (`TRANSLATE_EFFORT=0` skips the step), including on the model actually
    served (`MODEL`), so a re-download that clobbers the template is
    re-translated. A template whose effort block matches no known shape warns
    and is left alone — this runs under `set -e` after the download, so it must
    not fail a ready model dir. A running server loads the template at startup
    — restart to pick up a translation.
