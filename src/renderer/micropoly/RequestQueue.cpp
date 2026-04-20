// RequestQueue.cpp
// =================
// See RequestQueue.h for the contract. This TU owns the single
// vmaCreateBuffer call and the drain implementation.

#include "renderer/micropoly/RequestQueue.h"

#include "gfx/Allocator.h"

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

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <utility>

namespace enigma::renderer::micropoly {

namespace {

constexpr u32 kMinCapacity = 16u;
constexpr u32 kMaxCapacity = 1u << 20;  // 1 Mi

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

const char* requestQueueErrorKindString(RequestQueueErrorKind kind) {
    switch (kind) {
        case RequestQueueErrorKind::BufferCreationFailed: return "BufferCreationFailed";
        case RequestQueueErrorKind::CapacityTooSmall:     return "CapacityTooSmall";
        case RequestQueueErrorKind::CapacityTooLarge:     return "CapacityTooLarge";
    }
    return "?";
}

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

RequestQueue::RequestQueue(gfx::Allocator& allocator, RequestQueueOptions opts,
                           VkBuffer buf, VmaAllocation alloc, void* mappedPtr)
    : allocator_(&allocator),
      opts_(opts),
      buffer_(buf),
      allocation_(alloc),
      mappedPtr_(mappedPtr) {
    // Zero the header + slots on construction. The mapped pointer is
    // host-visible + coherent, so a plain memset is sufficient.
    if (mappedPtr_ != nullptr) {
        std::memset(mappedPtr_, 0, static_cast<std::size_t>(bufferBytes()));
        // Write the capacity into the header so the GPU-side bounds check
        // in page_request_emit.hlsl has a correct capacity immediately.
        RequestQueueHeader* h = reinterpret_cast<RequestQueueHeader*>(mappedPtr_);
        h->capacity = opts_.capacity;
    }
}

RequestQueue::~RequestQueue() {
    destroy();
}

RequestQueue::RequestQueue(RequestQueue&& other) noexcept
    : allocator_(other.allocator_),
      opts_(other.opts_),
      buffer_(other.buffer_),
      allocation_(other.allocation_),
      mappedPtr_(other.mappedPtr_) {
    std::lock_guard<std::mutex> lk(other.mutex_);
    stats_ = other.stats_;

    other.allocator_  = nullptr;
    other.buffer_     = VK_NULL_HANDLE;
    other.allocation_ = VK_NULL_HANDLE;
    other.mappedPtr_  = nullptr;
    other.stats_      = {};
}

RequestQueue& RequestQueue::operator=(RequestQueue&& other) noexcept {
    if (this == &other) return *this;
    destroy();

    allocator_  = other.allocator_;
    opts_       = other.opts_;
    buffer_     = other.buffer_;
    allocation_ = other.allocation_;
    mappedPtr_  = other.mappedPtr_;

    {
        std::lock_guard<std::mutex> lk(other.mutex_);
        stats_ = other.stats_;

        other.allocator_  = nullptr;
        other.buffer_     = VK_NULL_HANDLE;
        other.allocation_ = VK_NULL_HANDLE;
        other.mappedPtr_  = nullptr;
        other.stats_      = {};
    }
    return *this;
}

void RequestQueue::destroy() {
    if (allocator_ != nullptr && buffer_ != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator_->handle(), buffer_, allocation_);
    }
    buffer_     = VK_NULL_HANDLE;
    allocation_ = VK_NULL_HANDLE;
    mappedPtr_  = nullptr;
}

std::expected<RequestQueue, RequestQueueError> RequestQueue::create(
    gfx::Allocator& allocator, const RequestQueueOptions& opts) {

    if (opts.capacity < kMinCapacity) {
        return std::unexpected(RequestQueueError{
            RequestQueueErrorKind::CapacityTooSmall,
            formatDetail("capacity=%u below minimum %u", opts.capacity, kMinCapacity),
        });
    }
    if (opts.capacity > kMaxCapacity) {
        return std::unexpected(RequestQueueError{
            RequestQueueErrorKind::CapacityTooLarge,
            formatDetail("capacity=%u exceeds maximum %u", opts.capacity, kMaxCapacity),
        });
    }

    const u64 bytes = static_cast<u64>(sizeof(RequestQueueHeader))
                    + static_cast<u64>(opts.capacity) * sizeof(u32);

    VkBufferCreateInfo bufCI{};
    bufCI.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufCI.size        = bytes;
    bufCI.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                      | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                      | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bufCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    // Host-visible + coherent + mapped. VMA picks an upload heap with
    // WRITE_COMBINED semantics on most discrete GPUs when PREFER_HOST is
    // paired with HOST_ACCESS_RANDOM, which is what we want since the GPU
    // does InterlockedAdd's through a non-cached bar. The CPU drain path
    // only reads + resets — both naturally align with write-combined.
    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    allocCI.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT
                  | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
    // Code-reviewer MAJOR closeout: surface the caller's debugName to VMA's
    // allocation user-data. This is what shows up in vmaBuildStatsString /
    // RenderDoc memory views without forcing a full VK_EXT_debug_utils wire-up.
    allocCI.pUserData = const_cast<char*>(opts.debugName);

    VkBuffer          buf    = VK_NULL_HANDLE;
    VmaAllocation     alloc  = VK_NULL_HANDLE;
    VmaAllocationInfo info{};
    const VkResult vr = vmaCreateBuffer(allocator.handle(), &bufCI, &allocCI,
                                        &buf, &alloc, &info);
    if (vr != VK_SUCCESS) {
        return std::unexpected(RequestQueueError{
            RequestQueueErrorKind::BufferCreationFailed,
            formatDetail("vmaCreateBuffer returned VkResult=%d for %llu bytes",
                         static_cast<int>(vr),
                         static_cast<unsigned long long>(bytes)),
        });
    }
    if (info.pMappedData == nullptr) {
        vmaDestroyBuffer(allocator.handle(), buf, alloc);
        return std::unexpected(RequestQueueError{
            RequestQueueErrorKind::BufferCreationFailed,
            "vmaCreateBuffer returned null mapped pointer — HOST_VISIBLE memory unavailable",
        });
    }

    return RequestQueue(allocator, opts, buf, alloc, info.pMappedData);
}

// ---------------------------------------------------------------------------
// Drain
// ---------------------------------------------------------------------------

RequestQueue::DrainResult RequestQueue::drainEx(std::span<u32> outPageIds) {
    DrainResult res{};
    if (mappedPtr_ == nullptr) return res;

    // Acquire fence so any prior GPU writes made visible via a pipeline
    // barrier + host memory coherency are observable here. The caller's
    // fence/semaphore wait already handles the GPU->CPU sync; this fence
    // keeps the compiler from reordering the memcpy above the host-side
    // reads the caller just issued.
    std::atomic_thread_fence(std::memory_order_acquire);

    RequestQueueHeader* h = reinterpret_cast<RequestQueueHeader*>(mappedPtr_);
    u32* slots = reinterpret_cast<u32*>(
        static_cast<u8*>(mappedPtr_) + sizeof(RequestQueueHeader));

    // memcpy the header fields into a local snapshot, then reset the
    // header in place. Using memcpy rather than a direct dereference
    // ensures strict-aliasing-safe access on the mapped region.
    RequestQueueHeader snap{};
    std::memcpy(&snap, h, sizeof(snap));

    // Clamp count to capacity — the shader-side emitter may have observed
    // slot >= capacity and bumped overflowed, but we still need to ignore
    // the overshoot slots since they were never written.
    const u32 rawCount   = snap.count;
    const u32 validCount = rawCount > opts_.capacity ? opts_.capacity : rawCount;
    const u32 spanCap    = static_cast<u32>(outPageIds.size());
    const u32 copyCount  = validCount < spanCap ? validCount : spanCap;
    const u32 dropCount  = validCount > spanCap ? (validCount - spanCap) : 0u;

    if (copyCount > 0u) {
        std::memcpy(outPageIds.data(), slots, copyCount * sizeof(u32));
    }

    // Reset header for next frame. Keep capacity; clear count + overflowed.
    RequestQueueHeader reset{};
    reset.count      = 0u;
    reset.capacity   = opts_.capacity;
    reset.overflowed = 0u;
    reset._pad       = 0u;
    std::memcpy(h, &reset, sizeof(reset));

    // Release fence so the header reset is observable before the GPU
    // reads header.count on the next frame (the caller's semaphore/
    // barrier chain carries the actual visibility guarantee; this fence
    // keeps the host-side compiler from reordering past the memcpy).
    std::atomic_thread_fence(std::memory_order_release);

    {
        std::lock_guard<std::mutex> lk(mutex_);
        stats_.totalDrained   += copyCount;
        stats_.lastDrainCount  = copyCount;
        if (snap.overflowed != 0u || rawCount > opts_.capacity) {
            ++stats_.overflowEvents;
        }
    }
    res.copied  = copyCount;
    res.dropped = dropCount;
    return res;
}

u32 RequestQueue::drain(std::span<u32> outPageIds) {
    return drainEx(outPageIds).copied;
}

std::vector<u32> RequestQueue::drain() {
    std::vector<u32> out;
    out.resize(opts_.capacity);
    // Span is sized to capacity, so drainEx() will never drop — ignore it.
    const u32 n = drainEx(std::span<u32>(out)).copied;
    out.resize(n);
    return out;
}

RequestQueue::Stats RequestQueue::stats() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return stats_;
}

}  // namespace enigma::renderer::micropoly
