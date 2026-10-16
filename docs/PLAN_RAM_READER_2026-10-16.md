# PLAN: RAM reader — rewrite strata_memory.py as a /proc snapshot parser

Date: 2026-10-16. Phase 2 of the VMA-naming program (engine phase done: PLAN_RAM_VMA_NAMING_2026-10-16.md, deployed 0.1.44.1+vma-names). The reader only READS what the engine marks; it must never demand engine changes.

## Goal

Rewrite `/home/g0/strata-v100/personal/tools/strata_memory.py` (personal/, not in git) from the retired STRATA-RAM-POOLS contract parser into a pure `/proc` snapshot parser that reports the seven user-defined classes (CONTEXT.md) with a node0/node1 split, and closes the accounting.

## Data sources (all snapshot, no engine cooperation)

- `/proc/<engine-pid>/maps` — VMA names (`[anon:strata/cold_experts]`, `[anon_shmem:strata/session_active]`, `[anon:strata/conv_cache]`); accept both `anon:` and `anon_shmem:` prefixes for any class name.
- `/proc/<engine-pid>/smaps` — per-VMA Rss; this kernel's smaps has NO Name field, so correlate smaps blocks to maps ranges by start address.
- `/proc/<engine-pid>/numa_maps` — per-range N0=/N1= page counts (missing field = 0); correlate by address range (numa_maps has no names).
- `/proc/<engine-pid>/smaps_rollup` — process Rss for the closure check.
- `/proc/meminfo` + `/sys/devices/system/node/node{0,1}/meminfo` — host totals and per-node FilePages (system-level file-cache distribution).
- Engine PID discovery: `systemctl --user show strata.service -p MainPID` (python serve parent) → scan its children → the child whose maps contains `strata/` names (fallback: largest-Rss child). Never trust the parent.

## Classification (seven classes)

| Class | Rule | Node split |
|---|---|---|
| cold_experts | VMAs named strata/cold_experts: Rss from smaps | numa_maps N0/N1 |
| session_active | VMAs named strata/session_active | numa_maps |
| conv_cache | VMAs named strata/conv_cache (zero when cache empty — expected) | numa_maps |
| hot_experts | grounded zero: no such pool in this build (cite basis, don't fake) | — |
| file cache | mincore probe of model weight files (keep existing approach) for the program-total; per-node attribution of the program's cache is NOT knowable — instead print system-level per-node FilePages from /sys node meminfo | system-level |
| engine_misc | smaps_rollup Rss − Σ named classes (residual; unnamed /dev/zero stager rings etc. land here) | numa_maps total minus named |
| other | MemTotal − MemAvailable − engine process Rss (host residual); print MemAvailable as annotation only, never add it into used | host-level |

## Mechanics

- **Consistency**: read maps twice (before/after the smaps/numa_maps pass); if they differ, retry once, else warn "snapshot inconsistent" on the output.
- **Closure check (acceptance)**: named classes + engine_misc ≡ smaps_rollup Rss (± a page or two of sampling skew, reported as-is); with a non-empty conv_cache present.
- Output: keep the existing CLI shape (text tables + `--json`), extend per-class lines to `Rss (node0 / node1)`. hot_experts line prints `0 (declared-zero: no pool in this build)`.
- Retire all STRATA-RAM-POOLS contract parsing, lock/generation/boot-id validation, updating-state handling, and the old parser's tests; keep the mincore file-cache probe and CLI conventions. Keep personal/tools/README.md in sync.

## Validation (live, low-intensity)

1. Run against live production: cold_experts ≈48 GiB all N1; session_active ≈1.55 GiB all N1; conv_cache = 0; closure holds.
2. Positive case: run `personal/tools/dual_session_cache_test.py` (single-shot, STRATA_API_KEY via grep -oP from unit into a variable, never echoed) to park a conversation → script now shows conv_cache > 0 with node0 majority; closure still holds.
3. No service restarts, no config changes, no GPU experiments — inference goes through the running production endpoint only.
