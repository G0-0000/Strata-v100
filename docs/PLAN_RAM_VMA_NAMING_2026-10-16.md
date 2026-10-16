# PLAN: Engine-side RAM class marking via kernel VMA naming

Date: 2026-10-16. Status: approved strategy (CONTEXT.md, docs/adr/0001), this is the engine implementation plan. Reader/script work is a later phase and must not constrain this design.

## Goal

Mark the engine's RAM allocations so an external reader can attribute every resident page to one of the seven user-defined classes with a node0/node1 split, using only `/proc/<pid>/maps`, `smaps`, `numa_maps`. Marking = `prctl(PR_SET_VMA, PR_SET_VMA_ANON_NAME, base, len, name)` on VMAs the engine owns from mmap to munmap. No sidecar contract, no registry, no invalidation protocol.

Class names (≤79 bytes, static strings):

| Class | VMA name | Target node |
|---|---|---|
| 冷专家 cold experts | `strata/cold_experts` | node1 (existing mbind) |
| 当前会话缓存 active session | `strata/session_active` | node1 (existing MPOL_BIND scope) |
| 二级缓存 parked conversations | `strata/conv_cache` | node0 (existing cold binding) |
| 热专家 hot experts | (no such pool; reader reports grounded zero) | — |
| 文件缓存 / 引擎杂项 / 其他 | (reader-side: mincore + residuals; nothing to mark) | — |

Naming must never fail the engine: on prctl error, log once and continue unnamed (those bytes land in the misc residual).

## Step 1 — Selective cleanup (remove sidecar, keep clamp)

Per scout-20261016-b41c §4: every hunk is classified. Remove:

- Untracked files: `include/strata/core/ram_pool_map.hpp`, `tests/core/ram_pool_map_test.cpp`, `tests/core/parked_ram_map_test.cpp`. Keep `docs/RAM_MONITOR_HANDOFF_2026-10-16.md` (historical record).
- `src/core/expert_source.cpp`: the `#include "strata/core/ram_pool_map.hpp"` (:11), `ram_pool_close` in `close()` (:1041-1042), the `ram_pool_open`/`set_ranges` block (:3506-3517). **Keep the clamp fix (:2925-2943, claim_credit, 19 lines) byte-identical.**
- `include/strata/core/expert_source.hpp`: member `ram_pool_handle_` (:925-927).
- `include/strata/core/conversation_buffer.hpp`: `visit_segments` (:50-54).
- `include/strata/core/conversation_cache.hpp`: all +106 sidecar lines (destructor close, `RamPoolMutation` wrappers, `visit_ram_ranges` chain, `withdraw_ram_pool_`/`refresh_ram_pool_`, members).
- `include/strata/core/session_save_reclaim.hpp`: `RamPoolMutation` scope (:27-30).
- `src/program/generate.cpp`: `RamPoolMutation` around the admission loop (:8491-8499).
- `CMakeLists.txt`: the two test targets (:508-530).

## Step 2 — Naming helper + mmap allocator (new header)

New `include/strata/core/vma_alloc.hpp`:

- `bool vma_name(void* base, size_t len, const char* name)` — prctl wrapper; returns false on error (caller logs once).
- `template<class T> struct VmaAllocator` — stateful std allocator carrying `const char* name`; `allocate()` = `mmap(NULL, n, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0)` + `vma_name()`; `deallocate()` = `munmap`. Standard rebind/equality. Page granularity acceptable.
- `bool vma_rename(const std::vector<uint8_t, VmaAllocator<uint8_t>>&, const char* name)` helper for transition-point renames (uses data()/capacity()).

## Step 3 — Instrument the three classes

### 3a. 冷专家 → `strata/cold_experts`

`src/core/expert_source.cpp` complement arena mmap (~:2826, single contiguous MAP_PRIVATE|ANON + mbind): call `vma_name(base, bytes, "strata/cold_experts")` right after mmap succeeds, inside the same function, before first touch. One site, allocated once at load.

### 3b. 当前会话缓存 → `strata/session_active`

1. **KvHostPools** (`src/core/layer.cpp` `qsa_state_init` :801-857): after `cudaHostAlloc` (:816) succeeds and inside the existing `ThreadNodeBinding::bind(hot)` scope (:808-825), call `vma_name(host_ptr, bytes, "strata/session_active")`. The cudaHostAlloc VMA is MAP_SHARED anon (`/dev/zero (deleted)`), confirmed nameable on this kernel; if prctl rejects it, log and continue (falls to misc). Same for the draft QsaState host pool (`src/core/mtp.cpp` :352-370).
2. **Active checkpoint chain** (`checks`, generate.cpp:8289, type `ConversationCheckpoint`): allocator-ize the byte/id members (see 3d) and construct them with name `strata/session_active` at `checkpoint_at` (:8600-8665).

### 3c. 二级缓存 → `strata/conv_cache`

1. **ConversationBuffer** (`include/strata/core/conversation_buffer.hpp`): change `Segment` (:144) to `std::vector<uint8_t, VmaAllocator<uint8_t>>` constructed with name `strata/conv_cache`. Only allocation site is `resize()` (:96-99). **Preserve the existing `ThreadNodeBinding` cold-node scope (:79-93) ordering: bind → reserve → resize (first touch stays bound to node0).** mmap itself touches no pages. `segment_capacity` growth logic unchanged; `bytes()` still by capacity.
2. **Parked checkpoint copies**: `conversation_checkpoint_save` (conversation_state.cpp:174-175) and `copy_checkpoints_beating` (conversation_copy.hpp:67-84) construct `ConversationCheckpoint` with name `strata/conv_cache`.
3. **Restore transition**: at `checks = std::move(incoming->checkpoints)` (generate.cpp:11269) the moved buffers still carry `strata/conv_cache`; add a rename pass over the moved chain → `strata/session_active` (per member vector, data()/capacity()). K/V `ConversationBuffer`s stay `strata/conv_cache` throughout (entries_/reuse_ single name, per approved Q11b).

### 3d. ConversationCheckpoint member allocator-ization

Members `ids/imgs/gdn/ple/tails/dead/block_pos` (conversation_cache.hpp:27-56) become named-allocator vectors. The name must be selectable per construction site (3b/3c), so `ConversationCheckpoint` needs a constructor (or factory) taking the class name and seeding member allocators; default constructor keeps `strata/conv_cache`. Check ripple points: `conversation_file.cpp` serialization (uses `.data()/.size()` — compatible), `conversation_copy.hpp`, `generate.cpp` (`ConvCheckpoint` alias :1540), `conversation_checkpoints_merge`. Small members (<128 KiB: tails/dead/block_pos) may keep the default std::vector if the ripple shrinks materially; `gdn` (112.2 MiB), `ple`, `ids` must be covered.

## Step 4 — Test

New `tests/core/vma_alloc_test.cpp` (CMake target next to the other core tests): mmap+name → read `/proc/self/maps` → assert `[anon:strata/test]` present; VmaAllocator vector reserve/resize/touch/destruct round-trip; rename helper flips the name. CPU-only.

## Step 5 — Build, regression, deploy

- Full default-target build + the 11 selected CPU regressions (@worker). No GPU, no production stop needed for building (production is currently stopped anyway).
- Deploy (@worker): backup current `/home/g0/strata/engine-cuda12/strata` + `BUILD.json` into `personal/backups/<date>-vma-naming/`, atomic replace, set BUILD.json label `0.1.44.1+vma-names`, remove drop-in `~/.config/systemd/user/strata.service.d/ram-pool-map.conf`, stop+disable+delete `strata-memory-monitor.service` (keep its logs), `daemon-reload`, start `strata.service`.
- Post-start read-only check: `/proc/<engine-child-pid>/maps` shows all three names; cold experts ≈48 GiB all N1 in numa_maps; session_active present. Smoke via personal/tools/test_quick.py.

## Non-goals / red lines

- No changes to NUMA bindings, resident budget, MemoryHigh/Max, sampling config, KQ env, or serve/. `STRATA_PREFILL_CPU_SHARE=0` untouched. 9router forwarding stays paused.
- No reader/script work in this phase; the engine marks, the script (later) reads.
- This is private-fork instrumentation, always-on with graceful degradation; the upstream opt-in rule does not apply (not an upstream PR).
