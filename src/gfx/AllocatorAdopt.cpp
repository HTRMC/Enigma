// AllocatorAdopt.cpp
// ===================
// Test-only factory for wrapping an already-created VmaAllocator in a
// gfx::Allocator + the shared destructor. See Allocator.h for the contract.
//
// Split from Allocator.cpp so test TUs that bring up their own VMA
// allocator can link just this small file — Allocator.cpp is where
// VMA_IMPLEMENTATION lives and tests already emit their own copy, so
// linking the full Allocator.cpp would violate VMA's single-TU rule and
// drag in core/Log.h (and its Log.cpp dependency, which tests do not link).
//
// This TU includes vk_mem_alloc.h WITHOUT VMA_IMPLEMENTATION — just the
// header declarations so we can call vmaCalculateStatistics + vmaDestroyAllocator.
// Those symbols are resolved at link time from whichever TU in the final
// link owns VMA_IMPLEMENTATION.
//
// We intentionally do NOT include core/Log.h here — leak-gate output uses
// plain fprintf so test binaries link clean without the logging runtime.

#include "gfx/Allocator.h"

#include "core/Assert.h"

#include <volk.h>

#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#if defined(_MSC_VER)
    #pragma warning(push)
    #pragma warning(disable: 4100 4127 4189 4324 4505)
#endif
#include <vk_mem_alloc.h>
#if defined(_MSC_VER)
    #pragma warning(pop)
#endif

#include <cstdio>

namespace enigma::gfx {

Allocator::~Allocator() {
    if (m_allocator == nullptr) {
        return;
    }

    // Adopt path: caller owns the VmaAllocator — skip leak-gate + destroy.
    if (m_externallyOwnedVma) {
        m_allocator = nullptr;
        return;
    }

    // VMA leak gate (AC13). Inspect live allocations before the
    // allocator is destroyed; log both counts and assert that no live
    // allocations remain. `blockCount` is INTENTIONALLY not asserted —
    // VMA retains empty blocks in its block pool and blockCount > 0
    // with allocationCount == 0 is documented as not-a-leak (see plan
    // risk R6-blockcount).
    VmaTotalStatistics stats{};
    vmaCalculateStatistics(m_allocator, &stats);
    std::fprintf(stdout,
        "[vma] shutdown allocationCount = %llu blockCount = %llu\n",
        static_cast<unsigned long long>(stats.total.statistics.allocationCount),
        static_cast<unsigned long long>(stats.total.statistics.blockCount));
    ENIGMA_ASSERT(stats.total.statistics.allocationCount == 0);

    vmaDestroyAllocator(m_allocator);
    m_allocator = nullptr;
}

std::unique_ptr<Allocator> Allocator::adopt(VmaAllocator raw) {
    return std::unique_ptr<Allocator>(new Allocator(raw));
}

Allocator::Allocator(VmaAllocator externalVma)
    : m_allocator(externalVma),
      m_externallyOwnedVma(true) {
}

} // namespace enigma::gfx
