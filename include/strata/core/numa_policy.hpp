// include/strata/core/numa_policy.hpp - S1/S2/S3: the NUMA placement primitives the engine places pages with.
//
// S1 (`--numa-hot-node`) placed the resident complement's pages on one node with three kernel calls; S2 and S3 place
// more of them, so those primitives live here instead of in expert_source.cpp's anonymous namespace.  Nothing here
// allocates: a caller mmaps its own range and asks this header to bind that range, or to bind the calling thread for
// the scope in which it faults its pages in (the range's own mbind is what decides where they go).
//
// The calls go through syscall(2): the engine links no libnuma (numaif.h is that library's header) and
// mbind/set_mempolicy/get_mempolicy are the kernel's own interface, as expert_source.cpp's io_submit path already
// does.  Every name below needs the three SYS_* numbers, so on a platform without them it is simply not declared
// and a caller guards its own use the same way.
#pragma once

#include <cstdint>
#include <string>

#if defined(__linux__)
#include <sys/syscall.h>   // SYS_mbind / SYS_set_mempolicy / SYS_get_mempolicy
#endif

namespace strata::core::detail {

#if defined(__linux__) && defined(SYS_mbind) && defined(SYS_set_mempolicy) && defined(SYS_get_mempolicy)

/// The values set_mempolicy/mbind take from user space, and the width of the node masks below - each one is
/// kNumaMaxNode / 64 unsigned longs.
constexpr int kMpolDefault = 0, kMpolBind = 2;
constexpr unsigned long kNumaMaxNode = 1024;

/// One bit set for `node`, the rest cleared; `mask` holds kNumaMaxNode / 64 unsigned longs.
void numa_node_mask(int node, unsigned long* mask);

/// mbind(MPOL_BIND) on [addr, addr + bytes); the range must be page-aligned, and a mask no node satisfies (an
/// out-of-range node) comes back as EINVAL.  On false, `error_number` is the errno the call set.
bool numa_bind_range(void* addr, uint64_t bytes, int node, int& error_number);

/// The calling thread's memory policy, bound to one node for a scope and put back when the scope ends.  The VMA's
/// own mbind is what decides where its pages go; the thread policy covers the faults that bring them in and
/// anything this thread allocates meanwhile.  `bind` false (with `why`) = the caller goes on with mbind alone,
/// which is the graceful degradation (the pages then follow the process's own policy, as they always did).
struct ThreadNodeBinding {
    bool bound = false, saved = false;
    int saved_mode = kMpolDefault;
    unsigned long saved_mask[kNumaMaxNode / 64] = {};

    bool bind(int node, std::string& why);
    void restore();
    ~ThreadNodeBinding() { restore(); }
};

#endif

}  // namespace strata::core::detail
