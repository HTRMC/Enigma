// PageCache.cpp
// ==============
// See PageCache.h for the contract. This TU owns the single
// vmaCreateBuffer + free-slot bitmap logic; nothing in here touches the
// GPU beyond that one allocation.

#include "renderer/micropoly/PageCache.h"

#include "core/Assert.h"
#include "gfx/Allocator.h"

// VMA: the engine's Allocator.cpp stamps VMA_IMPLEMENTATION exactly once.
// Here we just need the types, so DO NOT re-define VMA_IMPLEMENTATION.
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

#include <bit>
#include <cstdarg>
#include <cstdio>
#include <utility>

namespace enigma::renderer::micropoly {

namespace {

// 4 GiB single-VkBuffer ceiling. VMA + common drivers will typically refuse
// or degrade badly above this — and the SSBO address range on SPIR-V
// shaders is bounded by this too. Documented in PageCacheOptions::poolBytes.
//
// Practical cap is one slot shy of UINT32_MAX so the shader-side
// `slotIndex * pageSlotBytes` (u32) cannot wrap on the highest slot. The
// raster task/mesh shaders use u32 byte offsets into the page-cache
// ByteAddressBuffer; capping here keeps that math monotonic without
// forcing 64-bit load intrinsics in HLSL.
constexpr u64 kMaxPoolBytes = static_cast<u64>(UINT32_MAX);

// Build a detail string with a snprintf-style formatter. Kept inline so the
// error paths read top-down without jumping through a helper.
std::string formatDetail(const char* fmt, ...) {
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    const int n = std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n <= 0) return {};
    return std::string(buf, buf + (static_cast<std::size_t>(n) < sizeof(buf)
                                   ? n : sizeof(buf) - 1));
}

}  // namespace

const char* pageCacheErrorKindString(PageCacheErrorKind kind) {
    switch (kind) {
        case PageCacheErrorKind::PoolTooLarge:         return "PoolTooLarge";
        case PageCacheErrorKind::PoolTooSmall:         return "PoolTooSmall";
        case PageCacheErrorKind::SlotBytesBad:         return "SlotBytesBad";
        case PageCacheErrorKind::BufferCreationFailed: return "BufferCreationFailed";
        case PageCacheErrorKind::NoFreeSlot:           return "NoFreeSlot";
        case PageCacheErrorKind::SlotOutOfRange:       return "SlotOutOfRange";
        case PageCacheErrorKind::DoubleFree:           return "DoubleFree";
    }
    return "?";
}

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

PageCache::PageCache(gfx::Allocator& allocator,
                     PageCacheOptions opts, VkBuffer buf, VmaAllocation alloc,
                     u32 totalSlots)
    : allocator_(&allocator),
      opts_(opts),
      buffer_(buf),
      allocation_(alloc),
      totalSlots_(totalSlots) {
    // Bitmap sized in 64-bit words; round up.
    const std::size_t words = (static_cast<std::size_t>(totalSlots_) + 63u) / 64u;
    allocBitmap_.assign(words, 0ull);

    // Mark phantom bits in the last word as permanently allocated. This makes
    // the allocate() scan's `slot >= totalSlots_` guard a defence-in-depth
    // assertion rather than load-bearing logic, and protects future refactors
    // of the scan loop.
    const u32 tail = totalSlots_ % 64u;
    if (tail != 0u && words > 0u) {
        allocBitmap_.back() = (~uint64_t{0}) << tail;
    }
}

PageCache::~PageCache() {
    destroy();
}

PageCache::PageCache(PageCache&& other) noexcept
    : allocator_(other.allocator_),
      opts_(other.opts_),
      buffer_(other.buffer_),
      allocation_(other.allocation_),
      totalSlots_(other.totalSlots_) {
    // Move the bookkeeping under `other`'s lock so a concurrent reader on the
    // moved-from instance can't tear. We're the only writer for `*this`.
    std::lock_guard<std::mutex> lk(other.mutex_);
    allocBitmap_     = std::move(other.allocBitmap_);
    usedSlots_       = other.usedSlots_;
    allocationCount_ = other.allocationCount_;
    freeCount_       = other.freeCount_;

    other.allocator_  = nullptr;
    other.buffer_     = VK_NULL_HANDLE;
    other.allocation_ = VK_NULL_HANDLE;
    other.totalSlots_ = 0u;
    other.usedSlots_  = 0u;
    other.allocationCount_ = 0ull;
    other.freeCount_       = 0ull;
}

PageCache& PageCache::operator=(PageCache&& other) noexcept {
    if (this == &other) return *this;
    destroy();

    allocator_  = other.allocator_;
    opts_       = other.opts_;
    buffer_     = other.buffer_;
    allocation_ = other.allocation_;
    totalSlots_ = other.totalSlots_;

    {
        std::lock_guard<std::mutex> lk(other.mutex_);
        allocBitmap_     = std::move(other.allocBitmap_);
        usedSlots_       = other.usedSlots_;
        allocationCount_ = other.allocationCount_;
        freeCount_       = other.freeCount_;

        other.allocator_  = nullptr;
        other.buffer_     = VK_NULL_HANDLE;
        other.allocation_ = VK_NULL_HANDLE;
        other.totalSlots_ = 0u;
        other.usedSlots_  = 0u;
        other.allocationCount_ = 0ull;
        other.freeCount_       = 0ull;
    }
    return *this;
}

void PageCache::destroy() {
    if (allocator_ != nullptr && buffer_ != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator_->handle(), buffer_, allocation_);
    }
    buffer_     = VK_NULL_HANDLE;
    allocation_ = VK_NULL_HANDLE;
}

std::expected<PageCache, PageCacheError> PageCache::create(
    gfx::Allocator& allocator,
    const PageCacheOptions& opts) {

    // --- option validation -------------------------------------------------
    if (opts.slotBytes == 0u || (opts.slotBytes % 16u) != 0u) {
        return std::unexpected(PageCacheError{
            PageCacheErrorKind::SlotBytesBad,
            formatDetail("slotBytes=%u must be > 0 and a multiple of 16",
                         opts.slotBytes),
        });
    }
    if (opts.poolBytes == 0u ||
        opts.poolBytes < static_cast<u64>(opts.slotBytes)) {
        return std::unexpected(PageCacheError{
            PageCacheErrorKind::PoolTooSmall,
            formatDetail("poolBytes=%llu < slotBytes=%u",
                         static_cast<unsigned long long>(opts.poolBytes),
                         opts.slotBytes),
        });
    }
    if (opts.poolBytes > kMaxPoolBytes) {
        return std::unexpected(PageCacheError{
            PageCacheErrorKind::PoolTooLarge,
            formatDetail("poolBytes=%llu exceeds u32 single-buffer limit (%llu)",
                         static_cast<unsigned long long>(opts.poolBytes),
                         static_cast<unsigned long long>(kMaxPoolBytes)),
        });
    }

    // Clamp poolBytes down to a whole-slot multiple. This is a friendlier
    // behavior than rejecting the options outright — the caller's input
    // (e.g. pageCacheMB = 1024) lines up cleanly with sane slot sizes
    // (64 MiB default), and shaving a few bytes off an oddball config
    // keeps the bitmap math tidy.
    const u64 slotBytes64 = static_cast<u64>(opts.slotBytes);
    const u64 alignedBytes = (opts.poolBytes / slotBytes64) * slotBytes64;
    if (alignedBytes == 0u) {
        // Shouldn't happen given the earlier PoolTooSmall guard, but keep
        // the guard explicit so a future change to the bounds doesn't
        // silently let a 0-slot buffer through.
        return std::unexpected(PageCacheError{
            PageCacheErrorKind::PoolTooSmall,
            "poolBytes rounded to 0 after slot alignment",
        });
    }

    const u64 slotCount64 = alignedBytes / slotBytes64;
    if (slotCount64 > static_cast<u64>(UINT32_MAX)) {
        // totalSlots is a u32 in the public surface. A pool this large would
        // need > 4 G slots; clamp it out rather than silently wrap.
        return std::unexpected(PageCacheError{
            PageCacheErrorKind::PoolTooLarge,
            formatDetail("slotCount=%llu exceeds u32 range",
                         static_cast<unsigned long long>(slotCount64)),
        });
    }

    // --- VkBuffer + VMA allocation ----------------------------------------
    PageCacheOptions effective = opts;
    effective.poolBytes = alignedBytes;

    VkBufferCreateInfo bufCI{};
    bufCI.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufCI.size        = alignedBytes;
    bufCI.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                      | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                      | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;  // forward-compat for BDA access in M2.3+

    // M3.0 prereq A — queue-family ownership resolution. When caller passes
    // distinct graphics + transfer families (discrete-GPU path), declare
    // CONCURRENT sharing across both so transfer-queue writes and graphics-
    // queue reads skip the explicit ownership-transfer barrier.
    // Unified-queue devices (UINT32_MAX sentinel, or equal indices) stay on
    // EXCLUSIVE — preserves the M2 path unchanged on consumer GPUs.
    u32 families[2]{};
    const bool hasGfx = (opts.graphicsQueueFamily != UINT32_MAX);
    const bool hasXfer = (opts.transferQueueFamily != UINT32_MAX);
    if (hasGfx && hasXfer && opts.graphicsQueueFamily != opts.transferQueueFamily) {
        families[0] = opts.graphicsQueueFamily;
        families[1] = opts.transferQueueFamily;
        bufCI.sharingMode           = VK_SHARING_MODE_CONCURRENT;
        bufCI.queueFamilyIndexCount = 2u;
        bufCI.pQueueFamilyIndices   = families;
    } else {
        bufCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_AUTO;
    // Dedicated memory: this is a single big sticky allocation (up to 4 GiB).
    // VMA's default block size is 256 MiB, so without DEDICATED we would
    // force block creation that dwarfs the page cache itself.
    allocCI.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

    VkBuffer      buf   = VK_NULL_HANDLE;
    VmaAllocation alloc = VK_NULL_HANDLE;
    const VkResult vr = vmaCreateBuffer(allocator.handle(), &bufCI, &allocCI,
                                        &buf, &alloc, nullptr);
    // Intentionally manual: ENIGMA_VK_CHECK aborts; create() must return
    // std::unexpected so the caller can handle allocation failure gracefully.
    if (vr != VK_SUCCESS) {
        return std::unexpected(PageCacheError{
            PageCacheErrorKind::BufferCreationFailed,
            formatDetail("vmaCreateBuffer returned VkResult=%d for %llu bytes",
                         static_cast<int>(vr),
                         static_cast<unsigned long long>(alignedBytes)),
        });
    }

    return PageCache(allocator, effective, buf, alloc,
                     static_cast<u32>(slotCount64));
}

// ---------------------------------------------------------------------------
// Allocation
// ---------------------------------------------------------------------------

std::expected<u32, PageCacheError> PageCache::allocate() {
    std::lock_guard<std::mutex> lk(mutex_);

    for (std::size_t w = 0; w < allocBitmap_.size(); ++w) {
        const uint64_t word = allocBitmap_[w];
        if (word == ~uint64_t{0}) continue;  // word fully allocated

        // std::countr_one on the word tells us how many low-order 1-bits
        // precede the first 0-bit. That 0 position is our free slot within
        // this word.
        const int bit = std::countr_one(word);
        ENIGMA_ASSERT(bit >= 0 && bit < 64);

        const u32 slot = static_cast<u32>(w * 64u) + static_cast<u32>(bit);
        if (slot >= totalSlots_) {
            // Overflow past the legit slot range — the tail of the last
            // word is padding. Since we scan words left-to-right, once we
            // see a padding bit every remaining bit is also padding, so
            // we're out of slots.
            break;
        }

        allocBitmap_[w] = word | (uint64_t{1} << bit);
        ++usedSlots_;
        ++allocationCount_;
        return slot;
    }

    return std::unexpected(PageCacheError{
        PageCacheErrorKind::NoFreeSlot,
        formatDetail("all %u slots allocated", totalSlots_),
    });
}

std::expected<void, PageCacheError> PageCache::free(u32 slotIndex) {
    std::lock_guard<std::mutex> lk(mutex_);

    if (slotIndex >= totalSlots_) {
        return std::unexpected(PageCacheError{
            PageCacheErrorKind::SlotOutOfRange,
            formatDetail("slot %u >= totalSlots %u", slotIndex, totalSlots_),
        });
    }

    const std::size_t w   = slotIndex / 64u;
    const uint64_t    mask = uint64_t{1} << (slotIndex % 64u);

    if ((allocBitmap_[w] & mask) == 0u) {
        return std::unexpected(PageCacheError{
            PageCacheErrorKind::DoubleFree,
            formatDetail("slot %u already free", slotIndex),
        });
    }

    allocBitmap_[w] &= ~mask;
    --usedSlots_;
    ++freeCount_;
    return {};
}

u64 PageCache::slotByteOffset(u32 slotIndex) const {
    ENIGMA_ASSERT(slotIndex < totalSlots_);
    return static_cast<u64>(slotIndex) * static_cast<u64>(opts_.slotBytes);
}

PageCacheStats PageCache::stats() const {
    std::lock_guard<std::mutex> lk(mutex_);
    PageCacheStats s{};
    s.totalSlots      = totalSlots_;
    s.usedSlots       = usedSlots_;
    s.freeSlots       = totalSlots_ - usedSlots_;
    s.totalBytes      = static_cast<u64>(totalSlots_) * static_cast<u64>(opts_.slotBytes);
    s.usedBytes       = static_cast<u64>(usedSlots_)  * static_cast<u64>(opts_.slotBytes);
    s.allocationCount = allocationCount_;
    s.freeCount       = freeCount_;
    return s;
}

}  // namespace enigma::renderer::micropoly
