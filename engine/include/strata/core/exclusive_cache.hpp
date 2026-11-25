// include/strata/core/exclusive_cache.hpp - in-place exchange targets for the PCIe fetch kernel.
#pragma once

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRATA_EXCL_HD __host__ __device__
#else
#define STRATA_EXCL_HD
#endif

namespace strata::core::exclusive {

// deviation: constexpr permits a compile-time check of the tagged, aligned destination encoding.
STRATA_EXCL_HD constexpr unsigned long long fetch_target(unsigned long long slot, bool swap) { return slot | (swap ? 1ull : 0ull); }
STRATA_EXCL_HD constexpr unsigned long long fetch_slot(unsigned long long target) { return target & ~1ull; }
STRATA_EXCL_HD constexpr bool fetch_swaps(unsigned long long target) { return (target & 1ull) != 0; }
static_assert(fetch_slot(fetch_target(16ull, true)) == 16ull && fetch_swaps(fetch_target(16ull, true)) &&
              !fetch_swaps(fetch_target(16ull, false)));

// deviation: omit the unused start_refusal and the upstream comment's nonexistent switch.
}  // namespace strata::core::exclusive

#undef STRATA_EXCL_HD
