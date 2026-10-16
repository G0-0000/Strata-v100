// Kernel VMA naming for the engine's RAM classes (docs/PLAN_RAM_VMA_NAMING_2026-10-16.md).
//
// PR_SET_VMA / PR_SET_VMA_ANON_NAME names an anonymous mapping, and /proc/<pid>/maps then shows it as
// [anon:<name>] (a shared mapping as [anon_shmem:<name>]); that name is the one identity a reader outside the
// process can see, so it carries the class the approved RAM map attributes the pages to.  No registry, no
// sidecar, no invalidation protocol: the name lives in the VMA itself, from the naming call to munmap.
//
// This is instrumentation, not function: nothing here may fail the engine.  A naming call that fails -
// unsupported kernel, or refused (the start must be page aligned: every mmap base is) - is reported once, and
// the bytes simply stay unnamed (the reader reports those in its residual class).
#pragma once

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <type_traits>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#include <sys/prctl.h>
#endif

namespace strata::core {

/// The class names of the approved RAM map (PLAN_RAM_VMA_NAMING_2026-10-16.md §Class names).  Static strings,
/// one spelling per class: a VMA keeps its name for its whole life, so the string must outlive the mapping
/// (the kernel copies the characters, the contents only have to be right at the call).
namespace vma_class {
inline constexpr const char* cold_experts = "strata/cold_experts";
inline constexpr const char* session_active = "strata/session_active";
inline constexpr const char* conv_cache = "strata/conv_cache";
} // namespace vma_class

/// Name the anonymous mapping [base, base + len): false on any error (unsupported kernel, bad range), true when
/// the kernel took it.  `base` is the mmap base (page aligned, as the kernel requires).  Never throws.
inline bool vma_name(void* base, size_t len, const char* name) {
#if defined(__linux__) && defined(PR_SET_VMA) && defined(PR_SET_VMA_ANON_NAME)
    if (base == nullptr || len == 0 || name == nullptr) return false;
    return ::prctl(PR_SET_VMA, PR_SET_VMA_ANON_NAME, (unsigned long) base, (unsigned long) len,
                   (unsigned long) name) == 0;
#else
    (void) base; (void) len; (void) name;
    errno = ENOTSUP;   // for the one log line of vma_name_warn
    return false;
#endif
}

/// vma_name() for call sites that do not want to handle the failure: the FIRST failure is reported once - a
/// naming failure repeats for every one of thousands of segments, one line is all an operator needs - and every
/// later one is silent.  True when named, false when the bytes stay unnamed.
inline bool vma_name_warn(void* base, size_t len, const char* name) {
    if (vma_name(base, len, name)) return true;
    static std::atomic<bool> reported{false};
    if (!reported.exchange(true)) {
        std::fprintf(stderr, "strata: VMA naming is off: cannot tag %zu bytes at %p as [anon:%s] (%s); "
                             "those bytes stay unnamed and the RAM reader shows them in its residual\n",
                     len, base, name != nullptr ? name : "?", std::strerror(errno));
    }
    return false;
}

namespace vma_detail {
/// Test-only failure injection for map_bytes(): armed with k > 0, the next k - 1 mapping calls go through and the
/// k-th returns null - the mmap twin of the operator-new injection the split/save failure tests use for heap
/// storage, so a test can prove an mmap failure surfaces as std::bad_alloc through the containers this allocator
/// backs, rollback included (tests/core/vma_alloc_test.cpp).  0 in production and only a test ever writes it: the
/// cost is one relaxed load per mapping call, against a syscall.
inline std::atomic<uint64_t>& map_fail_after() {
    static std::atomic<uint64_t> calls{0};
    return calls;
}

/// One buffer's RAM class, in a cell its allocator and every copy of that allocator share.  The class has to be
/// able to move under a LIVE buffer - the restore transition moves a parked checkpoint's buffers into the active
/// session (vma_rename below) - and the only handle that rename has is vector::get_allocator(), a copy: sharing
/// the cell is what carries the write into the vector's own allocator, so the mapping of its next reallocation is
/// named with the class the buffer belongs to now.  One cell per buffer, made by its allocator's constructors.
struct ClassCell {
    const char* name = nullptr;   ///< the [anon:<name>] a mapping gets; null: unnamed memory
};

/// One mapping: anonymous private pages, page aligned.  No page is touched here - the first touch stays with the
/// caller's NUMA binding scope (mmap itself faults nothing in).  Null on failure.
inline void* map_bytes(size_t bytes) {
    const uint64_t fail = map_fail_after().load(std::memory_order_relaxed);
    if (fail != 0) {
        map_fail_after().store(fail - 1, std::memory_order_relaxed);
        if (fail == 1) { errno = ENOMEM; return nullptr; }   // the armed call: the injected mapping failure
    }
#if defined(__linux__)
    void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? nullptr : p;
#else
    return std::malloc(bytes);   // no VMA names outside Linux; the allocator still has to work
#endif
}
inline void unmap_bytes(void* p, size_t bytes) noexcept {
#if defined(__linux__)
    (void) ::munmap(p, bytes);   // the length rounds up to the pages mmap took
#else
    (void) bytes;
    std::free(p);
#endif
}
} // namespace vma_detail

/// A standard allocator whose storage is one named mapping per allocation.  Unlike a block of the glibc heap -
/// which shares its mapping with its neighbours and starts at an offset, so naming it after the fact cannot work
/// - the mapping is exactly this allocation, page aligned, and gets the class name right away.
///
/// The class (vma_class above; null: plain unnamed memory) lives in the cell below and travels with the buffer:
/// a copy of the container keeps it (the copy shares the cell), a move steals it with the buffer.  Sharing the
/// cell is also what lets the class change under a live buffer: vma_rename() - the transition-point helper -
/// writes the cell, and the vector's own allocator, which shares it, names the mapping of its next reallocation
/// with the new class.  Copy ASSIGNMENT keeps the destination's class - which class a buffer serves is a property
/// of the container it lands in.
template<class T>
struct VmaAllocator {
    using value_type = T;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    using propagate_on_container_copy_assignment = std::false_type;
    using is_always_equal = std::false_type;

    VmaAllocator() = default;
    VmaAllocator(const char* class_name)
        : cell_(std::make_shared<vma_detail::ClassCell>(vma_detail::ClassCell{class_name})) {}   // NOLINT: vector/std::pmr-style implicit naming
    template<class U> VmaAllocator(const VmaAllocator<U>& other) noexcept : cell_(other.cell_) {}
    VmaAllocator(const VmaAllocator&) noexcept = default;                  // copies share the buffer's class cell
    VmaAllocator& operator=(const VmaAllocator&) noexcept = default;
    VmaAllocator(VmaAllocator&& other) noexcept : cell_(other.cell_) {}    // the class stays with the moved buffer
    VmaAllocator& operator=(VmaAllocator&& other) noexcept { cell_ = other.cell_; return *this; }
    template<class U> struct rebind { using other = VmaAllocator<U>; };

    /// The class this allocator names its mappings with; null: unnamed memory.
    const char* name() const noexcept { return cell_->name; }
    /// Move the buffer to another class: the mapping a later reallocation takes is named `class_name` too.
    /// vma_rename() below is the rename's one caller - it holds the mapping to re-tag in place.
    void set_name(const char* class_name) noexcept { cell_->name = class_name; }

    T* allocate(size_t n) {
        if (n > max_size()) throw std::bad_alloc();
        const size_t bytes = std::max<size_t>(1, n * sizeof(T));   // allocate(0) still hands out addressable bytes
        void* p = vma_detail::map_bytes(bytes);
        if (p == nullptr) throw std::bad_alloc();
        if (const char* class_name = name(); class_name != nullptr) vma_name_warn(p, bytes, class_name);
        return static_cast<T*>(p);
    }
    void deallocate(T* p, size_t n) noexcept {
        if (p != nullptr) vma_detail::unmap_bytes(p, std::max<size_t>(1, n * sizeof(T)));
    }
    size_t max_size() const noexcept { return SIZE_MAX / sizeof(T); }
    template<class U> bool operator==(const VmaAllocator<U>& other) const noexcept { return name() == other.name(); }
    template<class U> bool operator!=(const VmaAllocator<U>& other) const noexcept { return name() != other.name(); }

private:
    std::shared_ptr<vma_detail::ClassCell> cell_ = std::make_shared<vma_detail::ClassCell>();   ///< the class, shared with this allocator's copies
    template<class> friend struct VmaAllocator;
};

/// Re-tag a live buffer in place: the restore transition moves parked containers into the active session and
/// their pages do not move - only the class they are reported under changes.  data() is the mapping's base (page
/// aligned) and capacity() * sizeof(T) the bytes allocate() took, so this names exactly that mapping.  The
/// buffer's allocator follows - get_allocator() hands out a copy that shares the class cell - so the mapping of
/// its next reallocation is named with the new class too.  True when named, false when the buffer stays unnamed
/// (capacity 0: nothing to name, the class cell alone moves).
template<class T>
bool vma_rename(const std::vector<T, VmaAllocator<T>>& v, const char* name) {
    const bool named = v.capacity() == 0 ||
                       vma_name_warn((void*) v.data(), v.capacity() * sizeof(T), name);
    if (named) v.get_allocator().set_name(name);   // a failed rename leaves the buffer - and its class - as it was
    return named;
}

} // namespace strata::core
