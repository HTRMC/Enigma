#pragma once

// RequestQueue.h
// ===============
// GPU-visible coherent ring buffer used by compute shaders to request
// Micropoly pages from disk. The shader-side pattern is:
//     InterlockedAdd(header.count, 1, slot);
//     if (slot < header.capacity) slots[slot] = pageId;
//     else                        InterlockedOr(header.overflowed, 1);
// The CPU drains the buffer once per frame, routes each pageId through
// MicropolyStreaming, then resets header.count to 0.
//
// Contract
// --------
// - `create()` factory returns std::expected; construction is non-throwing.
// - The VkBuffer is host-coherent (HOST_ACCESS_RANDOM + mapped persistently)
//   so no explicit flush/invalidate is required. The buffer is also
//   STORAGE + TRANSFER_DST + SHADER_DEVICE_ADDRESS (forward-compat BDA).
// - `drain()` is main-thread-only; one drain per frame. Stats() is safe
//   from any thread.
// - The caller is responsible for making sure the GPU is not mid-write
//   at the moment of drain. In practice this is a fence + pipeline barrier
//   from the producing compute pass — MicropolyStreaming (M2.4) wires it.

#include "core/Types.h"

#include <volk.h>

// Forward declare VMA opaque handle — full header only in the .cpp TU.
struct VmaAllocation_T;
using VmaAllocation = VmaAllocation_T*;

#include <expected>
#include <mutex>
#include <span>
#include <string>
#include <vector>

namespace enigma::gfx {
class Allocator;
}  // namespace enigma::gfx

namespace enigma::renderer::micropoly {

// Construction-time parameters.
struct RequestQueueOptions {
    // Maximum in-flight requests per frame. Must be >= 16 and <= 1<<20
    // (1 Mi requests). 4096 covers the typical visible-cluster density
    // on a 1080p frame without overflowing.
    u32 capacity = 4096u;

    // Descriptive name used only for diagnostics.
    const char* debugName = "micropoly.requestQueue";
};

enum class RequestQueueErrorKind {
    BufferCreationFailed,
    CapacityTooSmall,
    CapacityTooLarge,
};

struct RequestQueueError {
    RequestQueueErrorKind kind{};
    std::string           detail;
};

// Mirror of the shader-side layout — kept in sync with page_request_emit.hlsl.
struct RequestQueueHeader {
    u32 count;        // number of valid entries; GPU InterlockedAdd's this
    u32 capacity;     // matches RequestQueueOptions::capacity
    u32 overflowed;   // GPU sets to 1 (InterlockedOr) if count exceeds capacity
    u32 _pad;
};
static_assert(sizeof(RequestQueueHeader) == 16,
              "RequestQueueHeader layout must match page_request_emit.hlsl");

class RequestQueue {
public:
    static std::expected<RequestQueue, RequestQueueError> create(
        gfx::Allocator& allocator, const RequestQueueOptions& opts);

    ~RequestQueue();

    RequestQueue(const RequestQueue&)            = delete;
    RequestQueue& operator=(const RequestQueue&) = delete;
    RequestQueue(RequestQueue&&) noexcept;
    RequestQueue& operator=(RequestQueue&&) noexcept;

    // Immutable accessors — no lock.
    VkBuffer buffer()   const { return buffer_; }
    u32      capacity() const { return opts_.capacity; }

    // Bindless slot accessors. Owned by the caller (Renderer registers via
    // DescriptorAllocator::registerUavBuffer, then stamps the returned slot
    // here). Mirrors the MicropolyStreaming pattern for pageToSlotBuffer /
    // pageFirstDagNodeBuffer so RequestQueue stays free of descriptor-
    // allocator coupling — the smaller test binaries can skip bindless
    // registration and leave the slot at UINT32_MAX.
    void setBindlessSlot(u32 slot) { bindlessSlot_ = slot; }
    u32  bindlessSlot() const      { return bindlessSlot_; }

    // Byte size of the whole buffer (header + capacity*sizeof(uint32_t)).
    u64 bufferBytes() const {
        return static_cast<u64>(sizeof(RequestQueueHeader))
             + static_cast<u64>(opts_.capacity) * sizeof(u32);
    }

    // Drain result — `copied` is the number of pageIds written into the
    // output span; `dropped` is the number of GPU-queued entries that were
    // present but could not be written because the span was too small (M2.3
    // Security MEDIUM-1 follow-up: surface silent-drop events so the caller
    // can bump a telemetry counter instead of losing them unobservably).
    struct DrainResult {
        u32 copied  = 0u;
        u32 dropped = 0u;
    };

    // CPU drain — read header.count, copy up to outPageIds.size() entries
    // into outPageIds, reset header.count + overflowed back to 0. Returns
    // copied + dropped counts; `copied == min(count, span.size())` and
    // `dropped == max(0, count - span.size())`. Stats update regardless.
    //
    // Intended call site: once per frame, after a pipeline barrier on the
    // producing compute pass. Main thread only.
    DrainResult drainEx(std::span<u32> outPageIds);

    // Legacy overload — returns just the copied count. Kept for back-compat
    // with tests/request_queue_test.cpp and any caller that does not need
    // drop diagnostics. Forwards to drainEx().
    u32 drain(std::span<u32> outPageIds);

    // Convenience overload that allocates a fresh vector sized to capacity.
    // Because the vector is sized to capacity, `dropped` is always 0 — the
    // split-out DrainResult is unnecessary; we return the vector only.
    std::vector<u32> drain();

    // Stats for debug / telemetry. Cheap; takes the internal mutex.
    struct Stats {
        u64 totalDrained     = 0u;  // cumulative count of drained requests
        u64 overflowEvents   = 0u;  // cumulative count of frames w/ overflow
        u32 lastDrainCount   = 0u;  // most recent drain result
    };
    Stats stats() const;

private:
    RequestQueue(gfx::Allocator& allocator, RequestQueueOptions opts,
                 VkBuffer buf, VmaAllocation alloc, void* mappedPtr);

    // Release VMA/Vulkan resources. Safe on a moved-from instance.
    void destroy();

    gfx::Allocator*     allocator_  = nullptr;
    RequestQueueOptions opts_{};
    VkBuffer            buffer_     = VK_NULL_HANDLE;
    VmaAllocation       allocation_ = VK_NULL_HANDLE;
    void*               mappedPtr_  = nullptr;  // persistent map; host-coherent

    mutable std::mutex  mutex_;
    Stats               stats_{};

    // Caller-owned bindless descriptor slot for the cull shader's
    // RWByteAddressBuffer dereference (pc.requestQueueBindlessIndex).
    // UINT32_MAX when unregistered — the cull shader's emitPageReq() then
    // no-ops because the bindless slot lookup hits an empty descriptor
    // (NVIDIA silently drops, other vendors may assert).
    u32                 bindlessSlot_ = UINT32_MAX;
};

const char* requestQueueErrorKindString(RequestQueueErrorKind kind);

}  // namespace enigma::renderer::micropoly
