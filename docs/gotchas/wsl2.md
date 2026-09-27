# Gotchas: WSL2 / Docker Desktop

Windows-paravirt memory behavior, the allocator default that differs there, and first-batch headroom.

Entries 3, 35, 53 of [docs/gotchas.md](../gotchas.md) (original numbers kept, so `gotcha N` references still land).

[← gotchas index](../gotchas.md) · [quickstart](../quickstart.md)

3. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on bare metal, and
   `False` on WSL2.** The DeltaNet prefill kernels allocate transient
   workspace; without expandable segments the allocator fragments and the
   engine OOMs at runtime once `gpu-memory-utilization` goes past ~0.975. It is
   not unconditional, which this entry used to imply: expandable segments need
   CUDA VMM, which WSL2's paravirt driver rejects during capture, so both start
   scripts set `expandable_segments:False` when they detect WSL2.
   **And `False` on bare-metal Linux as well, once TP>1 uses custom
   all-reduce.** Same root, a different consumer: with
   `--tensor-parallel-size 2` and CUDA graphs enabled, graph capture aborts
   with `Cuda error /workspace/csrc/custom_all_reduce.cuh:164 'invalid
   argument'` and the worker dies before the server comes up. Line 164 is the
   `cudaIpcGetMemHandle` in `get_graph_buffer_ipc_meta()`, and an expandable
   (VMM) segment has no IPC handle to export. Reported on 2x RTX 3090 NVLink,
   driver 595.91.07, CUDA 13.2, vLLM 0.28.0
   ([#163](https://github.com/syv-ai/HyperQwen/issues/163)), with the matrix
   that isolates it: `NCCL_P2P_LEVEL=SYS` does not help, `--enforce-eager`
   does (no graphs, no capture), and `expandable_segments:False` does while
   keeping the graphs. The two workarounds that get a server up —
   `--disable-custom-all-reduce` (NCCL carries the collectives) and
   `expandable_segments:False` (custom all-reduce works) — are not equivalent:
   in a controlled A/B on that box, C1 greedy decode is 171.8 tok/s on NCCL
   against 182.8 on custom all-reduce, +6.4%, at an identical 3.32 tokens per
   step, with every concurrency up to C8 improving. So on a multi-card box,
   turn the allocator off before you turn custom all-reduce off. Not yet
   soaked for fragmentation on long contexts, which is the thing
   `expandable_segments:True` was turned on for in the first place.

35. **The engine needs non-KV headroom for its first real batch, `MAX_SEQS` and
    `KV_MEM` are two doors into the same shortfall — and on WSL2 the failure is
    silent.** First seen as a seat-count death: `MAX_SEQS > 12` at `CTX=huge` kills
    the engine and the graphs are innocent — same visible failure as gotcha 34
    (boots, captures, `/health` 200, dies on the first prompt with
    `torch.OutOfMemoryError`), different bill. `CG` is pinned at its 64 cap in every
    one of the runs below, so the memory is going to allocations that scale with
    `max_num_seqs` itself, not with the captured batch. Free VRAM after boot on one
    24 GiB 3090, `SPEC=dflash2 CTX=huge PREFIX_CACHE=1` k=7, then a single ~3.7k-token
    prompt:

    | `MAX_SEQS` | free after boot | one ~3.7k prompt |
    |---|---|---|
    | 8 | 596 MiB | ok |
    | 10 | 456 MiB | ok |
    | 12 | 416 MiB | ok |
    | **16** | **356 MiB** | **dead** |

    The allocator says it plainly: `expandable_segments: memory mapping failed with OOM
    on device 0 while trying to map 20971520 bytes (free: 20578304, total:
    25272516608)` — 20 MB wanted against 20 MB left, on a card with 24 GB. It needs no
    concurrency at all: `num_running_reqs=1`, `step_counter=0`, `kv_cache_usage=0.18`.
    Reproduced twice with byte-identical counters.

    The seat count is only one door into that shortfall.
    [@mjungnickel18 named the real subject](https://github.com/syv-ai/HyperQwen/issues/25#issuecomment-5392694387)
    — *how much non-KV headroom does the engine need*, with `MAX_SEQS` and `KV_MEM` as
    two doors into the same room — after a `KV_MEM` pin on his box produced a failure
    this table does not contain (below). The same room walked through the `KV_MEM`
    door, seats pinned at 8, salted prompts, best of 3, prefill tok/s from TTFT:

    | `KV_MEM` | free after boot | 4k / 16k prefill | 8×16k concurrent | outcome |
    |---|---|---|---|---|
    | 5,261,334,938 (stock) | 576 MiB | 1,156 / 1,107 | ok — free bottoms at 110 MiB | ok |
    | 5,414,427,034 | 436 MiB | 1,159 / 1,100 | ok — free bottoms at **8 MiB** | ok |
    | 5,466,855,834 | **396 MiB** | 1,155 / 1,099 — full speed | **dead in 34 s, every request 500** | dead, twice |

    Same fingerprint at the bottom: the identical four failed 20,971,520-byte mappings
    with byte-identical free counters across both repeats, ending in
    `torch.OutOfMemoryError: Tried to allocate 24.00 MiB ... 37.62 MiB is free`. Two
    refinements the second ladder forces. The transient working set is *elastic*: it
    takes ~380 MiB when the room exists (watch `memory.free` during a prefill: 576 →
    194 MiB at stock) but squeezes without measurable cost — at 436 MiB free the 8-way
    cell ran at full throughput with 8 MiB left. What is rigid is the first real
    batch's allocation bill, sized by `max_num_seqs` and by how many requests actually
    run: 16 seats died on a single prompt, while 8 seats at 396 MiB prefilled 16k
    single-stream at full speed and died the moment eight ran at once. No
    configuration anywhere on either ladder was ever merely *slow*.

    That last sentence is the platform note, and it is the part that cost a week of
    cross-box debugging in [#25](https://github.com/syv-ai/HyperQwen/issues/25):
    on bare-metal Linux this failure has exactly two states, full speed or a loud named
    `torch.OutOfMemoryError`. Under WSL2 the WDDM driver backs the failed mapping with
    host memory instead, so the same exhaustion produces **no error at all** — just
    prefill at a fifth of the rate (232 tok/s at 4k against ~1,000 healthy, measured by
    @mjungnickel18 under a `KV_MEM` pin that left ~630 MiB free). A WSL user who raises
    `KV_MEM` gets the context they asked for, no warning, and 5–10× the TTFT, with
    nothing in the logs and no `nvidia-smi` number that flags it. The two boxes in
    [#25](https://github.com/syv-ai/HyperQwen/issues/25) make it concrete: the
    same pin (`KV_MEM=6871947673` at `CTX=fast`, `MAX_SEQS=2`; the boxes produce
    byte-identical pool geometry, 81,368 tokens at the fixed sibling pin) read
    ~630 MiB free after boot on WSL and served — slowly — for four days, while on bare
    metal it boots with 98 MiB free and the first prompt kills the engine. WSL's free-after-boot overstates the Linux number by roughly whatever WDDM
    is host-backing, so a headroom rule of thumb tuned on one platform does not
    transfer to the other in either direction. The detector is the one that found it: a
    prompt-length ladder against a known-good rate. On WSL, ladder any `KV_MEM` above
    stock before trusting it; the launcher now prints a warning when the pin exceeds
    the profile default. A cheaper live check, from a second WSL2 box in
    [#61](https://github.com/syv-ai/HyperQwen/issues/61): `nvidia-smi dmon`
    while it generates. Healthy decode on a 24 GB card is high SM occupancy *and*
    high power draw; host-backed memory shows as **SM near 100% at only 100-200 W**,
    because the SMs are stalled on PCIe rather than doing work. That reporter's rule
    of thumb — keep ~2 GB of VRAM free by lowering `GPU_UTIL`, and do not chase the
    context back with `MAX_LEN`, since the pool is sized by `GPU_UTIL` — matches the
    two boxes above.

    The launcher warns above 12 seats rather than clamping, because unlike `CG` this is
    a VRAM budget rather than a shape: a card bigger than 24 GiB has room where this
    one does not. Seats and pool trade against each other — if you want seats, buy them
    with a lower `KV_MEM`; if you want context, the ladder above is the price list, and
    on this card the floor at 8 seats sits between 396 and 436 MiB of free headroom.
    The shipped `CTX=huge` default is `MAX_SEQS=2`, so nothing here is reachable
    without an override.

    Worth reading next to the concurrency section of the README: seats above the
    residency were already useless (they queue, then preempt). Past 12 at `CTX=huge`
    they stop being useless and become fatal.

53. **On WSL2 the card's usable dedicated memory is about half a gigabyte
    smaller than the same card on bare metal, the shipped `SPEC=dflash2` boot
    sits about 50 MiB under that line, and anything larger runs slow instead
    of failing.** Windows accounts each adapter's memory as *dedicated* (on
    the card) and *shared* (host memory the driver backs GPU allocations
    with). On two RTX 4090s under WSL2 the dedicated figure tops out at about
    23,371 MiB of a 24,564 MiB card, where a bare-metal 3090 of the same size
    reports 23.3 GiB free at boot (the engine's log figure), about 490 MiB more. The default `SPEC=dflash2 CTX=fast` boot
    lands about 50 MiB under the ceiling. Anything that adds to the working
    set crosses it, and crossing it does not fail: the remainder lands in
    host shared memory and the step runs two to six times slower, with
    nothing in vLLM's log to show it. Four separately discovered "4090 costs"
    were this one thing, each measured over the line and then with room:

    | working set | over the line | with room |
    |---|---|---|
    | `DFLASH_TOKENS=15` (about 420 MiB over) | 56.5 ms/step | 23.4 |
    | verify-kernel query maximum 16 at width 7 | 63.5 | 23.2 |
    | a boot that recompiles the drafter's graph while loading the backbone's (about 300 MiB more) | 45.0 | 23.2 |
    | a 4 GiB bf16 drafter at the shipped pin (1.8 GB in host memory) | 150 to 264 | 39 |

    Three things to know and one to do:

    - **Read the counters, not the log.** `nvidia-smi` shows the card full
      either way. The Windows performance counters
      `\GPU Adapter Memory(*)\Dedicated Usage` and `\Shared Usage`
      (PowerShell `Get-Counter`, sampled every 10 s) show the split: shared
      usage at its idle baseline (about 86 MiB here) through a run means the
      working set is on the card; anything above it means part of it is not.
      Gotcha 35's `nvidia-smi dmon` power-and-SM signature is the same state
      seen from the other side.
    - **Acceptance is untouched; only step costs lie.** The spill moves memory,
      not arithmetic. Accepted-tokens-per-step columns from a spilled run
      stand; its tok/s does not.
    - **Boot class and memory state have to be matched before two hosts are
      compared.** With both matched, a 4090 under WSL2 and a bare-metal 3090
      agree on a drafter to a few percent; every cross-host "penalty" in the
      #25 thread that did not survive came from one of the two being
      unmatched.
    - **Give it room.** `KV_MEM=3000000000 DFLASH_MAX_LEN=8192` runs every
      configuration in the table at its bare-metal speed and costs the shipped
      head nothing at width 7 (23.3 against 23.2 ms/step, 3.77 against 3.80
      tokens/step). `SPEC_ATTN=0` frees about a gigabyte of graph capture
      (1.32 against 0.33 GiB, the same on a 3090), which on this host is the
      difference between fitting and not; on bare metal the room absorbs it.
    - **Pin cards with `CUDA_VISIBLE_DEVICES`, not `--gpus device=N`.** On this
      Docker Desktop host the device request is recorded (`docker inspect`
      shows `"DeviceIDs":["1"]`) and not enforced: a container started with
      `--gpus '"device=1"'` enumerated both cards and its CUDA device 0 was
      host GPU 0. `-e NVIDIA_VISIBLE_DEVICES=<uuid> -e CUDA_VISIBLE_DEVICES=<uuid>`
      pinned it, verified in-process before any GPU work. Everything above was
      run with the variable set, and the adapter counters put each run on the
      card it named.

    Measured in the
    [#25](https://github.com/syv-ai/HyperQwen/issues/25) comments of
    2026-09-05 (items 13 and 14, and the corrections to items 9 and 11).
