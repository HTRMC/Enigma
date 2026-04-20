#pragma once

// PageCache.h
// ============
// Device-side fixed-address page pool for the Micropoly streaming subsystem
// (M2.2). A single VkBuffer is carved into equally-sized slots; the streaming
// orchestrator (M2.3) asks this class for a free slot, uploads the decompressed
// page bytes into `buffer() + slotByteOffset(slot)`, and tells ResidencyManager
// which slot now backs which pageId.
//
// Contract
// --------
// - `create()` factory returns std::expected; construction is non-throwing.
// - `allocate()` / `free()` / `stats()` / `slotByteOffset()` are thread-safe.
// - `buffer()` / `bindlessIndex()` / `slotBytes()` / `totalSlots()` are
//   immutable after construction and do not lock.
// - Double-free and out-of-range free are typed errors, NOT undefined
//   behavior — the test harness relies on this.
// - Bindless registration is LEFT TO THE CALLER: PageCache exposes buffer()
//   and bindlessIndex() getters, and the owning orchestrator (M2.3) is
//   responsible for the DescriptorAllocator::registerStorageBuffer() call
//   (or equivalent). This keeps M2.2 independent of the engine's descriptor
//   set lifetime.
//
// THREAD-SAFETY CONTRACT: allocate(), free(), stats(), and the lock-free
// accessors (buffer, bindlessIndex, slotBytes, totalSlots, slotByteOffset)
// are safe to call from multiple threads concurrently on a fully-constructed
// instance. However, MOVING a PageCache (move ctor or move assignment) is
// NOT thread-safe with respect to concurrent callers on the SOURCE object.
// Callers must quiesce all threads accessing `other` before moving from it.
//
// The class is engine-private; no header dependency on Vulkan leaks outside
// the micropoly namespace. It is safe to instantiate during Renderer setup
// — construction performs a single vmaCreateBuffer for the slab.

#include "core/Types.h"

#include <volk.h>

// Forward declare the VMA allocation handle; we pull in vk_mem_alloc.h only
// from the implementation TU.
struct VmaAllocation_T;
using VmaAllocation = VmaAllocation_T*;

#include <expected>
#include <mutex>
#include <string>
#include <vector>

namespace enigma::gfx {
class Allocator;
}  // namespace enigma::gfx

namespace enigma::renderer::micropoly {

// Construction-time parameters. Sensible defaults match plan §3.M2 targets
// (1 GiB pool, 128 KiB slots). The caller typically passes a value derived
// from MicropolyConfig::pageCacheMB.
struct PageCacheOptions {
    // Target total VRAM for the page pool. Default 1 GiB. Must be > 0, must
    // be >= slotBytes, and must be <= 4 GiB (single VkBuffer size limit
    // observed on common hardware — VMA can create larger allocations in
    // principle, but SSBO addressing limits keep us at 4 GiB per buffer).
    u64 poolBytes = 1024ull * 1024ull * 1024ull;

    // Per-slot size. Defines the max decompressed page size the cache can
    // hold. Must be > 0, must be a multiple of 16, and must divide poolBytes
    // cleanly (PoolTooSmall is flagged separately if poolBytes < slotBytes).
    // Pages exceeding slotBytes are not supported in M2 — surface NoFreeSlot
    // so the orchestrator (M2.3) can decide policy.
    //
    // Typical decompressed page is 25-80 KiB (M1c DamagedHelmet stats).
    // kMpMaxPageDecompressedBytes is a corruption-protection cap at the reader
    // level, NOT a typical page size. 128 KiB gives 8192 slots on a 1 GiB
    // pool, enough for realistic scenes.
    u32 slotBytes = 128u * 1024u;  // 128 KiB

    // Bindless binding index for this SSBO. PageCache does NOT register the
    // buffer with DescriptorAllocator itself — the caller is expected to do
    // that and pass the returned slot index here so stats/diagnostic output
    // can report it. Use a sentinel (UINT32_MAX) if the caller has not yet
    // registered the buffer.
    u32 bindlessBindingIndex = UINT32_MAX;

    // Queue-family indices used to decide VK_SHARING_MODE for the VkBuffer.
    //   - When graphicsQueueFamily == transferQueueFamily (unified queue
    //     devices — the common consumer-GPU case): EXCLUSIVE sharing, no
    //     barrier overhead.
    //   - When they differ (discrete GPUs with a dedicated transfer family):
    //     CONCURRENT sharing across both families. Writes from the transfer
    //     queue (MicropolyStreaming uploads) and reads from the graphics
    //     queue (cull / material eval) then skip the queue-family ownership
    //     transfer barrier, at a small per-access cost on the driver side.
    //
    // Both default to UINT32_MAX meaning "treat as unified" (i.e. EXCLUSIVE).
    // The PageCache::create factory resolves this into sharingMode below.
    // M3.0 prereq A — routes MicropolyStreaming through the CONCURRENT path
    // when the device exposes a dedicated transfer family.
    u32 graphicsQueueFamily = UINT32_MAX;
    u32 transferQueueFamily = UINT32_MAX;

    // Descriptive name used only for diagnostics. The implementation does
    // not apply a Vulkan debug name — engine-wide debug-utils wiring lives
    // elsewhere and is optional. Keep the string static / long-lived.
    const char* debugName = "micropoly.pageCache";
};

enum class PageCacheErrorKind {
    PoolTooLarge,           // poolBytes > 4 GiB single-buffer limit
    PoolTooSmall,           // poolBytes < slotBytes
    SlotBytesBad,           // slotBytes == 0 or not a multiple of 16
    BufferCreationFailed,   // vmaCreateBuffer returned non-VK_SUCCESS
    NoFreeSlot,             // allocate() called with no slots available
    SlotOutOfRange,         // free() called with slot >= totalSlots
    DoubleFree,             // free() called on an already-free slot
};

struct PageCacheError {
    PageCacheErrorKind kind{};
    std::string        detail;
};

struct PageCacheStats {
    u32 totalSlots       = 0u;
    u32 freeSlots        = 0u;
    u32 usedSlots        = 0u;
    u64 totalBytes       = 0ull;
    u64 usedBytes        = 0ull;
    u64 allocationCount  = 0ull;  // cumulative — for debugging churn
    u64 freeCount        = 0ull;  // cumulative
};

class PageCache {
public:
    // Non-throwing factory. On failure returns PageCacheError with a kind
    // and a human-readable detail string (used by tests and logs).
    //
    // Note: create() intentionally takes only gfx::Allocator&, not gfx::Device&.
    // The page cache is a pure VkBuffer pool — no device-level operations are
    // performed here. Command-buffer work (transfer queue upload in M2.3) uses
    // Device via the caller's existing command pool/context.
    static std::expected<PageCache, PageCacheError> create(
        gfx::Allocator& allocator,
        const PageCacheOptions& opts);

    ~PageCache();

    PageCache(const PageCache&) = delete;
    PageCache& operator=(const PageCache&) = delete;
    PageCache(PageCache&& other) noexcept;
    PageCache& operator=(PageCache&& other) noexcept;

    // Thread-safe. Returns a slot index in [0, totalSlots). To get the byte
    // offset in the buffer, multiply by slotBytes().
    std::expected<u32, PageCacheError> allocate();

    // Thread-safe. Marks the slot as free. Double-free and out-of-range are
    // typed errors, not UB.
    std::expected<void, PageCacheError> free(u32 slotIndex);

    // Immutable accessors — no lock.
    u32 slotBytes()     const { return opts_.slotBytes; }
    u32 totalSlots()    const { return totalSlots_; }
    u32 bindlessIndex() const { return opts_.bindlessBindingIndex; }
    VkBuffer buffer()   const { return buffer_; }

    // Caller-assigned bindless slot. PageCache::create() leaves this at
    // UINT32_MAX; the Renderer registers buffer() with
    // DescriptorAllocator::registerUavBuffer and records the returned slot
    // here so shaders that push bindlessIndex() in a push block address the
    // right descriptor. Without this step every g_rwBuffers[UINT32_MAX] read
    // silently returns zeros (observed as "cluster.vertexCount == 0" in the
    // HW raster mesh shader).
    void setBindlessSlot(u32 slot) { opts_.bindlessBindingIndex = slot; }

    // Returns the byte offset of the slot within `buffer()`. Thread-safe
    // because the slotBytes / totalSlots fields are immutable after
    // construction. In debug builds, out-of-range asserts; in release,
    // wraps (caller is responsible — the allocate() path always returns
    // valid indices).
    u64 slotByteOffset(u32 slotIndex) const;

    // Thread-safe. Returns a snapshot of current stats.
    PageCacheStats stats() const;

private:
    // Private ctor; callers go through create().
    PageCache(gfx::Allocator& allocator,
              PageCacheOptions opts, VkBuffer buf, VmaAllocation alloc,
              u32 totalSlots);

    // Release VMA/Vulkan resources. Safe to call on a moved-from instance.
    void destroy();

    gfx::Allocator*  allocator_  = nullptr;
    PageCacheOptions opts_{};
    VkBuffer         buffer_     = VK_NULL_HANDLE;
    VmaAllocation    allocation_ = VK_NULL_HANDLE;
    u32              totalSlots_ = 0u;

    // Bitmap: one bit per slot. 0 = free, 1 = allocated. Stored as u64
    // words so allocate() can scan word-at-a-time via std::countr_one.
    mutable std::mutex    mutex_;
    std::vector<uint64_t> allocBitmap_;
    u32                   usedSlots_       = 0u;
    u64                   allocationCount_ = 0ull;
    u64                   freeCount_       = 0ull;
};

// Stringification for diagnostics / logs / test output.
const char* pageCacheErrorKindString(PageCacheErrorKind kind);

}  // namespace enigma::renderer::micropoly
