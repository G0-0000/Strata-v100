// include/strata/core/numa_policy.hpp - S1/S2/S3: the NUMA placement primitives the engine places pages with.
//
// S1 (`--numa-hot-node`) placed the resident complement's pages on one node with three kernel calls; S2 and S3 place
// more of them, so those primitives live here instead of in expert_source.cpp's anonymous namespace.  Nothing here
// allocates: a caller mmaps its own range and asks this header to bind that range, or to bind the calling thread for
// the scope in which it faults its pages in (the range's own mbind is what decides where they go).
//
// The calls go through syscall(2): the engine links no libnuma (numaif.h is that library's header) and
// mbind/set_mempolicy/get_mempolicy are the kernel's own interface, as expert_source.cpp's io_submit path already
// does.
//
// S2/S3 read the two nodes from the accessors below, and those are defined HERE and not in numa_policy.cpp: the cold
// node is read in conversation_buffer.hpp, a header-only class whose own tests link no library at all, so anything it
// calls has to be inline.  The binding's own kernel calls go the same way, and a platform without them gets the no-op
// below instead of an undeclared name, so a caller never guards its own use.  Only the range binder (which
// expert_source.cpp alone uses, from a library that links this one) is left out of line.
#pragma once

#include <atomic>
#include <cstdint>
#include <string>

#if defined(__linux__)
#include <cerrno>          // errno in the binding below
#include <cstdio>          // std::fprintf in the binding below
#include <cstring>         // std::memset / std::strerror
#include <sys/syscall.h>   // SYS_mbind / SYS_set_mempolicy / SYS_get_mempolicy
#include <unistd.h>        // syscall(2)
#endif

namespace strata::core::detail {

// ---- S2/S3: the nodes the switches chose, process-wide -------------------------------------------------------
//
// -1 ("off") is the default, and every placement site tests it FIRST: a run without --numa-hot-node takes no branch
// below, issues not one syscall, and allocates byte for byte as it always did.  The two are set once at startup -
// generate.cpp derives the cold node from the hot one - and read by the allocation paths on their own threads, hence
// the atomics.  The storage is a function-local static in an inline function: one instance across every translation
// unit, and no name to leak.
inline std::atomic<int>& numa_hot_node_storage() { static std::atomic<int> node{-1}; return node; }
inline std::atomic<int>& numa_cold_node_storage() { static std::atomic<int> node{-1}; return node; }
inline void set_numa_hot_node(int node) { numa_hot_node_storage().store(node, std::memory_order_relaxed); }
inline int numa_hot_node() { return numa_hot_node_storage().load(std::memory_order_relaxed); }
inline void set_numa_cold_node(int node) { numa_cold_node_storage().store(node, std::memory_order_relaxed); }
inline int numa_cold_node() { return numa_cold_node_storage().load(std::memory_order_relaxed); }

#if defined(__linux__) && defined(SYS_mbind) && defined(SYS_set_mempolicy) && defined(SYS_get_mempolicy)

/// The values set_mempolicy/mbind take from user space, and the width of the node masks below - each one is
/// kNumaMaxNode / 64 unsigned longs.
constexpr int kMpolDefault = 0, kMpolBind = 2;
/// mbind's MPOL_MF_MOVE: the bind also MIGRATES the pages already resident in the range to the node mask, not only
/// the faults that come after it (numa_move_range below; numa_bind_range's arena is bound before its first touch,
/// so the policy it sets is what places its pages - there is nothing to move yet).
constexpr int kMpolMove = 1 << 1;
constexpr unsigned long kNumaMaxNode = 1024;

/// One bit set for `node`, the rest cleared; `mask` holds kNumaMaxNode / 64 unsigned longs.
inline void numa_node_mask(int node, unsigned long* mask) {
    std::memset(mask, 0, sizeof(unsigned long) * (kNumaMaxNode / 64));
    mask[node / 64] |= 1ul << (node % 64);
}

/// mbind(MPOL_BIND) on [addr, addr + bytes); the range must be page-aligned, and a mask no node satisfies (an
/// out-of-range node) comes back as EINVAL.  On false, `error_number` is the errno the call set.
bool numa_bind_range(void* addr, uint64_t bytes, int node, int& error_number);

/// Move the pages ALREADY RESIDENT in [addr, addr + bytes) onto `node` - the same page-aligned range
/// numa_bind_range takes, with the one flag that migrates them instead of only deciding their future faults.  A
/// restored conversation's checkpoints were faulted on the cold node when the conversation was parked, and moving
/// them back into the active session re-tags their mappings but moves no page, so this is what puts them where an
/// active checkpoint's pages live (conversation_checkpoint_move, conversation_cache.hpp).  False (with
/// `error_number`) when the kernel could not move every page: the ones it could not move stay where they were,
/// which is the graceful degradation the caller logs and goes on from.  Inline, unlike the bind above, because a
/// header-only class calls it (conversation_cache.hpp is built with its own test, which links no library).
inline bool numa_move_range(void* addr, uint64_t bytes, int node, int& error_number) {
    unsigned long mask[kNumaMaxNode / 64];
    numa_node_mask(node, mask);
    errno = 0;
    if (syscall(SYS_mbind, addr, (unsigned long) bytes, kMpolBind, mask, kNumaMaxNode, kMpolMove) == 0) return true;
    error_number = errno;
    return false;
}

/// The calling thread's memory policy, bound to one node for a scope and put back when the scope ends.  The VMA's
/// own mbind is what decides where its pages go; the thread policy covers the faults that bring them in and
/// anything this thread allocates meanwhile.  `bind` false (with `why`) = the caller goes on with its allocation
/// unchanged, which is the graceful degradation (the pages then follow the process's own policy, as they always
/// did).
struct ThreadNodeBinding {
    bool bound = false, saved = false;
    int saved_mode = kMpolDefault;
    unsigned long saved_mask[kNumaMaxNode / 64] = {};

    bool bind(int node, std::string& why) {
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
            why = std::string("cannot bind this thread to node ") + std::to_string(node) + " (" +
                  std::strerror(errno) + ")";
            saved = false;
            return false;
        }
        bound = true;
        return true;
    }

    void restore() {
        if (!bound || !saved) return;
        bound = false;
        errno = 0;
        if (syscall(SYS_set_mempolicy, saved_mode, saved_mask, kNumaMaxNode) != 0)
            std::fprintf(stderr, "strata: WARNING: cannot put this thread's memory policy back (%s): allocations "
                                 "on it may take the node it was bound to (--numa-hot-node)\n", std::strerror(errno));
    }

    ~ThreadNodeBinding() { restore(); }
};

#else

/// A platform without set_mempolicy has nothing to bind, so this is what the callers there compile against.  `bind`
/// says so instead of pretending it worked, and the caller goes on with its allocation: it is only reached when
/// --numa-hot-node asked for a placement this host cannot make.
struct ThreadNodeBinding {
    bool bind(int, std::string& why) { why = "this platform has no set_mempolicy"; return false; }
    void restore() {}
};

/// A platform without mbind has no second node to move a page to: false, and the caller goes on with the pages
/// where they are.  `error_number` stays 0, so a caller that warns only on a real errno stays quiet here.
inline bool numa_move_range(void*, uint64_t, int, int& error_number) { error_number = 0; return false; }

#endif

}  // namespace strata::core::detail
