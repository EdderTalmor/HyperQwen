# Gotchas

Things that each cost us hours. Split into five files by symptom — entry
numbers are the original 1–60, so existing `gotcha N` references still land.

New here? Start at [docs/quickstart.md](quickstart.md), then come back.

[← back to the main README](../README.md)

## wsl2

| # | symptom |
|---|---|
| 3 | [expandable_segments: True on bare metal, False on WSL2 (and with TP>1 all-reduce)](gotchas/wsl2.md) |
| 35 | [first real batch needs non-KV headroom; on WSL2 the failure is silent slowness](gotchas/wsl2.md) |
| 53 | [WSL2 usable VRAM is ~0.5 GB smaller; over the line runs slow, not failed](gotchas/wsl2.md) |

## perf

| # | symptom |
|---|---|
| 2 | [KV pool sized by one profiling pass: dirty GPU / cold compile cache](gotchas/perf.md) |
| 4 | [MTP path needs GPU_UTIL 0.93, not 0.95+](gotchas/perf.md) |
| 6 | [random-token benchmarks are meaningless for speculative decoding](gotchas/perf.md) |
| 7 | [prefill chunks above 2048 shrink the pool](gotchas/perf.md) |
| 9 | [prompt_logprobs OOMs at 0.972 utilization](gotchas/perf.md) |
| 10 | [DeltaNet kernels at ~85% bandwidth; the lever is fp16 state dtype](gotchas/perf.md) |
| 11 | [draft vocabulary coverage is the single-user ceiling](gotchas/perf.md) |
| 16 | [INT8_LAYERS=. needs GPU_UTIL=0.95](gotchas/perf.md) |
| 24 | [verify block costs step time in stairs at 16 and 21 query tokens](gotchas/perf.md) |
| 25 | [verify block outgrowing its CUDA-graph reservation OOMs at runtime](gotchas/perf.md) |
| 28 | [halving the KV dtype can cost memory on a hybrid model](gotchas/perf.md) |
| 29 | [recurrent-state pages scale with the verify block, not the slot count](gotchas/perf.md) |
| 30 | [impossible max_model_len is the cheapest memory-model readout](gotchas/perf.md) |
| 31 | [KV_MEM assumes a headless card](gotchas/perf.md) |
| 34 | [decode-graph budget: CG = MAX_SEQS x (k+1), capped at 64](gotchas/perf.md) |
| 40 | [CTX=long fp8 KV backend matrix on sm86 (+ int8 escape costs)](gotchas/perf.md) |
| 42 | [bench default seed poisons A/Bs under prefix caching](gotchas/perf.md) |
| 43 | [max-num-batched-tokens 8192 refuses; 4096 costs pool](gotchas/perf.md) |
| 48 | [a benchmark row without its compile-cache state is not reproducible](gotchas/perf.md) |
| 49 | [high acceptance can mean a good drafter or collapsed text](gotchas/perf.md) |
| 50 | [per-request acceptance depends on request order](gotchas/perf.md) |
| 51 | [at width 15 on prose the engine drafts 7 — same experiment as 7](gotchas/perf.md) |
| 52 | [fixed-seed replays; a short cell cannot see a cohort-scale effect](gotchas/perf.md) |
| 57 | [fp8 split-KV verify is sm89+; INT8_ACT does not stack on it](gotchas/perf.md) |

## correctness

| # | symptom |
|---|---|
| 1 | [a benchmark cannot tell you the output is garbage — run quality_battery](gotchas/correctness.md) |
| 5 | [stale compiled artifact replays a graph built for other shapes](gotchas/correctness.md) |
| 13 | [greedy is not deterministic across drafter configs (±3%)](gotchas/correctness.md) |
| 14 | [vLLM picks the speculative method from the model path](gotchas/correctness.md) |
| 22 | [rsync preserves mtimes; delete __pycache__ after installing patches](gotchas/correctness.md) |
| 32 | [model dir without tokenizer.json reports as a reasoning-parser error](gotchas/correctness.md) |
| 33 | [Bug B: prefix-cache hit at one prompt residue in 128 collapses](gotchas/correctness.md) |
| 36 | [xgrammar rejects tokens past termination under a speculator](gotchas/correctness.md) |
| 46 | [prompt_logprobs wrong on CTX=huge + MTP + prefix caching](gotchas/correctness.md) |
| 47 | [DFLASH_TOKENS=15 asserts at engine start on the int4 path](gotchas/correctness.md) |
| 54 | [verify.sh false negative after `find -name '*.orig' -delete`](gotchas/correctness.md) |
| 56 | [registering an env var changes the torch.compile cache key](gotchas/correctness.md) |
| 58 | [reasoning_effort "minimal" 400s every request that carries it](gotchas/correctness.md) |

## speculation

| # | symptom |
|---|---|
| 12 | [FA2/FA3 do not split KV for multi-query decode — Triton patch + caps](gotchas/speculation.md) |
| 15 | [V2 runner graphs uncounted in pool sizing; pin the pool in bytes](gotchas/speculation.md) |
| 17 | [Triton scratch must not grow after graph capture (QMAX)](gotchas/speculation.md) |
| 18 | [async scheduling pins the speculative token count — needs ASYNC_SCHED=0](gotchas/speculation.md) |
| 19 | [--async-scheduling is already the default; --no-async-scheduling turns it off](gotchas/speculation.md) |
| 20 | [the DFlash draft pass is a captured graph: its Python runs once](gotchas/speculation.md) |
| 21 | [is_current_stream_capturing() is not a usable guard here](gotchas/speculation.md) |
| 23 | [a shorter draft block than num_speculative_tokens loses decode graphs](gotchas/speculation.md) |
| 26 | [the draft model is not redundant during a copy](gotchas/speculation.md) |
| 27 | [controller state outliving one step must be per-request](gotchas/speculation.md) |

## serving

| # | symptom |
|---|---|
| 8 | [vision tower is 0.858 GiB; offload default and why](gotchas/serving.md) |
| 37 | [sm80 Marlin repack Xid-31 under memory pressure; CPU fallback](gotchas/serving.md) |
| 38 | [OffloadingConnector CPU tier: uniform blocks vs asymmetric chunks + sizing](gotchas/serving.md) |
| 39 | ['every request re-prefills' — measure, floor formula, byte-identity, eviction](gotchas/serving.md) |
| 41 | [low-RAM host: stream weights with runai_streamer](gotchas/serving.md) |
| 44 | [align-mode prefix caching drops to 0% hits; checkpoint-order patch](gotchas/serving.md) |
| 45 | [first-request Triton compiles from four warmup gaps](gotchas/serving.md) |
| 55 | [streaming sends nothing during prefill — SSE keep-alive](gotchas/serving.md) |
| 59 | [two prepares at once half-write the model dir — flock](gotchas/serving.md) |
| 60 | [two long conversations evict each other at CTX=huge — retention interval](gotchas/serving.md) |
