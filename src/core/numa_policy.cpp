// src/core/numa_policy.cpp - S1's NUMA placement primitives, moved out of expert_source.cpp's anonymous namespace
// so S2 and S3 reach the same implementation.  The header says what each one does and why the calls go through
// syscall(2); this file is only where they are defined.
#include "strata/core/numa_policy.hpp"

#if defined(__linux__)
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <unistd.h>   // syscall(2)
#endif

namespace strata::core::detail {

#if defined(__linux__) && defined(SYS_mbind) && defined(SYS_set_mempolicy) && defined(SYS_get_mempolicy)

void numa_node_mask(int node, unsigned long* mask) {
    std::memset(mask, 0, sizeof(unsigned long) * (kNumaMaxNode / 64));
    mask[node / 64] |= 1ul << (node % 64);
}

bool numa_bind_range(void* addr, uint64_t bytes, int node, int& error_number) {
    unsigned long mask[kNumaMaxNode / 64];
    numa_node_mask(node, mask);
    errno = 0;
    if (syscall(SYS_mbind, addr, (unsigned long) bytes, kMpolBind, mask, kNumaMaxNode, 0) == 0) return true;
    error_number = errno;
    return false;
}

bool ThreadNodeBinding::bind(int node, std::string& why) {
    errno = 0;
    if (syscall(SYS_get_mempolicy, &saved_mode, saved_mask, kNumaMaxNode, nullptr, 0) != 0) {
        why = std::string("cannot read this thread's memory policy (") + std::strerror(errno) + ")";
        return false;
    }
    saved = true;
    unsigned long mask[kNumaMaxNode / 64];
    numa_node_mask(node, mask);
    errno = 0;
    if (syscall(SYS_set_mempolicy, kMpolBind, mask, kNumaMaxNode) != 0) {
        why = std::string("cannot bind this thread to node ") + std::to_string(node) + " (" + std::strerror(errno) + ")";
        saved = false;
        return false;
    }
    bound = true;
    return true;
}

void ThreadNodeBinding::restore() {
    if (!bound || !saved) return;
    bound = false;
    errno = 0;
    if (syscall(SYS_set_mempolicy, saved_mode, saved_mask, kNumaMaxNode) != 0)
        std::fprintf(stderr, "FileExpertSource: WARNING: cannot put this thread's memory policy back (%s): "
                             "allocations on it may take the node it was bound to (--numa-hot-node)\n",
                     std::strerror(errno));
}

#endif

}  // namespace strata::core::detail
