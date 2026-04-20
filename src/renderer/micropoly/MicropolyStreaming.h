#pragma once

// MicropolyStreaming.h
// =====================
// Per-frame orchestrator for the Micropoly streaming subsystem (M2.3).
// Owns the three cooperating components introduced in M2.1/M2.2:
//   * ResidencyManager  — CPU-side LRU over which pages are "hot".
//   * PageCache         — device-side fixed-address pool (VkBuffer slabs).
//   * RequestQueue      — GPU-visible coherent ring buffer written by
//                         compute shaders via InterlockedAdd.
//   * AsyncIOWorker     — background thread that reads compressed pages
//                         off disk and invokes a completion callback.
//
// The per-frame flow (driven by Renderer.cpp in M2.4):
//   1) `beginFrame()` calls `requestQueue_.drain()`.
//   2) Dedup pageIds with an internal per-frame set.
//   3) For each unique id not already resident, look up its MpPageEntry
//      via the caller-owned MpAssetReader and enqueue an AsyncIO request.
//   4) Drain any completions the worker has pushed since last frame.
//   5) For each successful completion, allocate a PageCache slot, stage
//      the bytes for a transfer-queue upload, and update ResidencyManager.
//      Evictions from the LRU release PageCache slots.
//
// M2.4a wires the transfer-queue upload path end-to-end:
//   * a small ring of staging buffers (N=kStagingRingSize, one slotBytes each)
//   * a transfer-queue command pool (from Device::transferQueueFamily if
//     present, falling back to the graphics family on unified devices)
//   * a TIMELINE semaphore that increments once per upload batch;
//     Renderer waits on the current counter value before cull/material-eval
//     to guarantee the PageCache memory is visible to compute reads.
//
// Ownership model: the orchestrator is returned by `create()` via
// std::unique_ptr. This is deliberate — AsyncIOWorker captures `this`
// in its completion callback, and ResidencyManager is non-movable.
// Holding the orchestrator by unique_ptr gives us a stable address for
// the lifetime of the object and sidesteps both constraints.

#include "core/Types.h"

#include "renderer/micropoly/AsyncIOWorker.h"
#include "renderer/micropoly/PageCache.h"
#include "renderer/micropoly/RequestQueue.h"
#include "renderer/micropoly/ResidencyManager.h"

#include <volk.h>

// Forward-declare VMA so we can hold a VmaAllocation member without pulling
// the full vk_mem_alloc.h into every TU that includes this header.
struct VmaAllocation_T;
using VmaAllocation = VmaAllocation_T*;

#include <array>
#include <deque>
#include <expected>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace enigma::asset {
class MpAssetReader;
}  // namespace enigma::asset

namespace enigma::gfx {
class Allocator;
class Device;
}  // namespace enigma::gfx

namespace enigma::renderer::micropoly {

struct MicropolyStreamingOptions {
    std::filesystem::path   mpaFilePath;
    // Caller-owned, held by pointer for page-entry lookup. Must outlive
    // the MicropolyStreaming instance.
    asset::MpAssetReader*   reader = nullptr;

    ResidencyManagerOptions residency{};
    PageCacheOptions        pageCache{};
    RequestQueueOptions     requestQueue{};
    // mpaFilePath + onComplete on this struct are overridden by the
    // orchestrator; any other field (e.g. maxInflightRequests) is honored.
    AsyncIOWorkerOptions    asyncIO{};

    const char* debugName = "micropoly.streaming";
};

enum class MicropolyStreamingErrorKind {
    InvalidOptions,
    ResidencyInitFailed,
    PageCacheInitFailed,
    RequestQueueInitFailed,
    AsyncIOInitFailed,
};

struct MicropolyStreamingError {
    MicropolyStreamingErrorKind kind{};
    std::string                 detail;
};

class MicropolyStreaming {
public:
    // Non-throwing factory. Returns unique_ptr so AsyncIOWorker can safely
    // capture the orchestrator by pointer without being invalidated by a
    // move, and so ResidencyManager (non-movable) can live inside.
    static std::expected<std::unique_ptr<MicropolyStreaming>, MicropolyStreamingError>
    create(gfx::Device& device,
           gfx::Allocator& allocator,
           MicropolyStreamingOptions opts);

    // M3.3: attach a GPU-visible pageId -> slotIndex mapping buffer. When
    // attached, beginFrame() will refresh the mapping each frame so the HW
    // raster task shader can look up the resident slot for a cluster's
    // page. Sized for `pageCount` u32 entries (pageId domain).
    //
    // The buffer is host-visible + coherent + STORAGE — the CALLER is
    // responsible for bindless registration (keeps MicropolyStreaming free
    // of DescriptorAllocator coupling, so the small test binaries can build
    // without pulling in the gfx descriptor machinery). The caller stores
    // the returned bindless slot (by calling setPageToSlotBindless()) so
    // later callers can query it via pageToSlotBufferBindless().
    //
    // Returns true on success. Idempotent — a second call resizes the
    // buffer if pageCount changed, but leaves the bindless slot alone
    // (caller must re-register if the VkBuffer handle changed).
    bool attachPageToSlotBuffer(u32 pageCount);

    // Stamp the bindless slot value returned from
    // DescriptorAllocator::registerStorageBuffer. No-op side effects; kept
    // as a plain setter so MicropolyStreaming never itself touches a
    // descriptor allocator.
    void setPageToSlotBindless(u32 bindlessSlot) { pageToSlotBindlessSlot_ = bindlessSlot; }

    // VkBuffer handle + bindless slot + pageCount + byte size for the
    // page-to-slot buffer. Returns null/UINT32_MAX/0 respectively when no
    // buffer is attached (e.g. in a disabled config or a test harness that
    // skips M3.3).
    VkBuffer pageToSlotBuffer()         const { return pageToSlotBuffer_; }
    u32      pageToSlotBufferBindless() const { return pageToSlotBindlessSlot_; }
    u32      pageToSlotPageCount()      const { return pageToSlotPageCount_; }
    u64      pageToSlotBufferBytes()    const { return pageToSlotBufferBytes_; }

    // M4.5 multi-cluster pages: upload a DEVICE_LOCAL SSBO of one u32 per
    // page holding `firstDagNodeIdx` (global DAG node index of the page's
    // first cluster). Shaders compute `localClusterIdx = globalDagNodeIdx -
    // pageFirstDagNodeBuffer[pageId]` to index the page's ClusterOnDisk
    // array, removing the M3.3 `const uint localClusterIdx = 0u` assumption.
    //
    // Owned by MicropolyStreaming (lifetime matches asset load); registered
    // as a bindless storage buffer on the caller-owned DescriptorAllocator.
    // A staging upload runs at attach time (once per asset), not per-frame.
    //
    // Idempotent — a second call at the same pageCount is a no-op.
    // Returns true on success. Failure leaves the buffer un-attached; the
    // caller (Renderer) logs and continues with multi-cluster pages showing
    // only cluster 0 as in the M3.3 state.
    bool attachPageFirstDagNodeBuffer(std::span<const u32> firstDagNodeIdxPerPage);

    // Stamp the bindless slot value returned from
    // DescriptorAllocator::registerStorageBuffer. No-op side effects; kept
    // as a plain setter so MicropolyStreaming never itself touches a
    // descriptor allocator.
    void setPageFirstDagNodeBindless(u32 bindlessSlot) {
        pageFirstDagNodeBindlessSlot_ = bindlessSlot;
    }

    // VkBuffer handle + bindless slot + byte size for the
    // pageFirstDagNode buffer (M4.5). Returns null/UINT32_MAX/0 when
    // unattached.
    VkBuffer pageFirstDagNodeBuffer()         const { return pageFirstDagNodeBuffer_; }
    u32      pageFirstDagNodeBufferBindless() const { return pageFirstDagNodeBindlessSlot_; }
    u64      pageFirstDagNodeBufferBytes()    const { return pageFirstDagNodeBufferBytes_; }

    // M3.3-deferred: upload the runtime-format DAG node array to a
    // DEVICE_LOCAL bindless SSBO. Each node is 48 bytes (3×float4 = center/
    // radius + coneApex/coneCutoff + coneAxis/packed(pageId|lodLevel<<24)),
    // matching the shader's `MpDagNode` layout in
    // mp_cluster_cull.comp.hlsl::loadDagNode. The caller is responsible for
    // assembling the runtime array via
    // MpAssetReader::assembleRuntimeDagNodes() and forwarding the result as
    // a byte span. Staging upload runs at attach time (once per asset load).
    //
    // Owned by MicropolyStreaming (lifetime matches asset load); registered
    // as a bindless storage buffer on the caller-owned DescriptorAllocator.
    // Idempotent — a second call at the same byte size is a no-op. Resize
    // tears down the old buffer so the next attach allocates fresh (mirrors
    // M4.5 Phase 4 detach-before-reattach fix).
    //
    // Returns true on success. Failure leaves the buffer un-attached; the
    // caller (Renderer) logs and micropoly rendering remains disabled.
    bool attachDagNodeBuffer(std::span<const u8> runtimeDagNodeBytes);

    // Stamp the bindless slot value returned from
    // DescriptorAllocator::registerStorageBuffer. No-op side effects; kept
    // as a plain setter so MicropolyStreaming never itself touches a
    // descriptor allocator.
    void setDagNodeBufferBindless(u32 bindlessSlot) {
        dagNodeBindlessSlot_ = bindlessSlot;
    }

    // VkBuffer handle + bindless slot + byte size for the DAG node buffer.
    // Returns null/UINT32_MAX/0 when unattached.
    VkBuffer dagNodeBuffer()         const { return dagNodeBuffer_; }
    u32      dagNodeBufferBindless() const { return dagNodeBindlessSlot_; }
    u64      dagNodeBufferBytes()    const { return dagNodeBufferBytes_; }

    ~MicropolyStreaming();

    MicropolyStreaming(const MicropolyStreaming&)            = delete;
    MicropolyStreaming& operator=(const MicropolyStreaming&) = delete;
    MicropolyStreaming(MicropolyStreaming&&)                 = delete;
    MicropolyStreaming& operator=(MicropolyStreaming&&)      = delete;

    // Per-frame stats. Returned by beginFrame() so tests + renderer debug
    // UI can drive assertions / HUD output without going through stats()
    // on each sub-component.
    struct FrameStats {
        u32 drained           = 0u;  // request queue drain count
        u32 dedupedFresh      = 0u;  // unique pageIds after dedup
        u32 cacheHits         = 0u;  // already resident
        u32 queuedForIO       = 0u;  // dispatched to AsyncIOWorker
        u32 enqueueFailures   = 0u;  // AsyncIOWorker queue was full
        u32 lookupFailures    = 0u;  // pageId not in the .mpa's page table
        u32 completed         = 0u;  // completions consumed this frame
        u32 uploadsScheduled  = 0u;  // transfer queue writes (0 in M2.3 stub)
        u32 uploadsFailed     = 0u;  // completion.success == false
        u32 evictions         = 0u;  // pages evicted from residency
        u32 slotAllocFailures = 0u;  // PageCache::allocate returned NoFreeSlot
    };

    // Per-frame entry point. See FrameStats for the observable outputs.
    FrameStats beginFrame();

    // Accessors for M2.4 renderer integration.
    RequestQueue&     requestQueue() { return requestQueue_; }
    PageCache&        pageCache()    { return pageCache_; }
    ResidencyManager& residency()    { return residency_; }

    // Timeline semaphore signaled by the transfer queue after each upload
    // batch in beginFrame(). Renderer's graphics submit waits on this
    // semaphore at value `uploadCounter()` before dispatching any shader
    // that reads the page cache. VK_NULL_HANDLE when no transfer work has
    // happened yet (first frame pre-adoption or empty-drain frames).
    VkSemaphore uploadDoneSemaphore() const { return uploadSema_; }

    // Current value the transfer queue will have reached after the most
    // recent beginFrame()'s submit completes. Combined with
    // uploadDoneSemaphore() this gives the Renderer everything it needs
    // for a VkTimelineSemaphoreSubmitInfo wait. Monotonic — never decreases.
    u64 uploadCounter() const { return uploadCounter_; }

    // Look up the slot index for a resident pageId. Returns UINT32_MAX
    // if the pageId is not resident (or never completed upload).
    u32 slotForPage(u32 pageId) const;

    // Drain any completions sitting in the worker-side queue WITHOUT
    // running the full beginFrame pipeline. Exposed for tests that want
    // to tick the completion machinery without also re-enqueueing new
    // requests from the GPU queue.
    u32 drainCompletionsForTest();

private:
    MicropolyStreaming(gfx::Device& device, gfx::Allocator& allocator,
                       MicropolyStreamingOptions opts,
                       PageCache pageCache,
                       RequestQueue requestQueue);

    // AsyncIOWorker completion callback — runs on the worker thread,
    // pushes onto pendingCompletions_ under completionMutex_.
    void onWorkerComplete_(PageCompletion c);

    // Factor: consume a batch of completions into residency + pageCache
    // + transfer-queue uploads. Shared by beginFrame() and
    // drainCompletionsForTest() so the eviction/insert/slot-alloc logic
    // lives in exactly one place (M2.3 architect + code-reviewer + security
    // LOW closeout — previously duplicated across both paths).
    //
    // Returns the delta stats to accumulate into the caller's FrameStats.
    // `recordUploads` gates the transfer-queue submission — test paths
    // that lack a real device pass false.
    struct CompletionPumpResult {
        u32 completed         = 0u;
        u32 uploadsScheduled  = 0u;
        u32 uploadsFailed     = 0u;
        u32 evictions         = 0u;
        u32 slotAllocFailures = 0u;
    };
    CompletionPumpResult processCompletions_(bool recordUploads);

    // Construct the transfer-queue command pool, staging ring, and timeline
    // semaphore. Called at the end of create() when device_ is usable.
    // Returns false on failure; streaming will run in CPU-only mode.
    bool initTransferResources_();

    // Tear down transfer-queue command pool, staging ring, and timeline
    // semaphore. Idempotent; called from the destructor.
    void destroyTransferResources_();

    gfx::Device*              device_    = nullptr;
    gfx::Allocator*           allocator_ = nullptr;
    MicropolyStreamingOptions opts_{};

    // ResidencyManager holds iterators into its own list — non-movable.
    // PageCache + RequestQueue are movable (VkBuffer + VmaAllocation).
    ResidencyManager               residency_;
    PageCache                      pageCache_;
    RequestQueue                   requestQueue_;
    std::unique_ptr<AsyncIOWorker> asyncIO_;

    // Completion queue: AsyncIOWorker's callback pushes here; beginFrame
    // drains. Using a plain mutex-protected vector is fine — completion
    // rate is O(pages per frame) which is tiny.
    mutable std::mutex             completionMutex_;
    std::vector<PageCompletion>    pendingCompletions_;

    // ---- Transfer-queue resources (M2.4a) --------------------------------
    // uploadSema_    : timeline semaphore; value = uploadCounter_ post-submit.
    // transferCmdPool_: command pool on transferQueueFamily (or graphics).
    // transferQueue_ : VkQueue handle used for submits (transfer or graphics).
    // transferFamily_: queue-family index that owns transferCmdPool_.
    VkSemaphore    uploadSema_        = VK_NULL_HANDLE;
    VkCommandPool  transferCmdPool_   = VK_NULL_HANDLE;
    VkQueue        transferQueue_     = VK_NULL_HANDLE;
    u32            transferFamily_    = 0u;
    u64            uploadCounter_     = 0ull;

    // Staging ring — N host-visible buffers, each slotBytes large. The
    // per-slot `readyValue` is the timeline value the transfer queue will
    // have reached once the previous use of this ring slot is retired; we
    // wait on it before reusing (vkWaitSemaphores timeline wait).
    //
    // Ring size bumped from 4 to 64 (Phase-4 architect fix): cold-cache
    // stress shows batches of up to ~40 pages can arrive in a single
    // beginFrame. With slotBytes = 128 KiB, 64 slots is 8 MiB of host-
    // visible staging — comfortable headroom with no intra-batch wrap.
    static constexpr u32 kStagingRingSize = 64u;
    struct StagingSlot {
        VkBuffer       buffer     = VK_NULL_HANDLE;
        VmaAllocation  allocation = VK_NULL_HANDLE;
        void*          mapped     = nullptr;
        u64            readyValue = 0ull;
    };
    std::array<StagingSlot, kStagingRingSize> staging_{};
    u32                                       stagingNextIdx_ = 0u;

    // Per-frame command buffer pool — one cmd buffer allocated per submit,
    // freed at the beginning of beginFrame() once the previous submit has
    // completed (observed via the timeline semaphore). Kept simple: a
    // vector of "pending cmdbufs + value they signal" consumed in FIFO order.
    struct PendingCmdBuf {
        VkCommandBuffer cmd     = VK_NULL_HANDLE;
        u64             waitVal = 0ull;
    };
    // deque (not vector): we pop_front() each retired entry in FIFO order.
    // vector::erase(begin()) is O(n) in the batch size; deque::pop_front()
    // is O(1) (Phase-4 CR HIGH fix).
    std::deque<PendingCmdBuf>      pendingCmdBufs_;

    // Dedup: per-frame set of pageIds already scheduled this frame.
    std::unordered_set<u32>        scheduledThisFrame_;

    // pageId -> PageCache slot. Updated when a completion is accepted;
    // cleared on eviction. Guarded by pageMapMutex_.
    mutable std::mutex             pageMapMutex_;
    std::unordered_map<u32, u32>   pageToSlot_;

    // M3.3 GPU-visible mirror of pageToSlot_.
    //
    // The buffer is HOST_VISIBLE + HOST_COHERENT + STORAGE (bindless) +
    // TRANSFER_DST (so the compiler doesn't reject it if a future path
    // wants to blit into it). It lives in its own VMA allocation, sized
    // for pageCount u32 entries; each entry is the PageCache slot index
    // for the matching pageId, or UINT32_MAX when not resident. The
    // mapping is refreshed at the end of each beginFrame() after
    // pageToSlot_ has been updated by the completion pump.
    //
    // The buffer is persistently mapped so beginFrame() can memcpy the
    // refreshed contents without flush/invalidate (coherent).
    //
    // Register_/release_ with DescriptorAllocator happens in
    // attachPageToSlotBuffer / destructor respectively. The caller must
    // not dispose of the allocator before this object is destroyed.
    VkBuffer       pageToSlotBuffer_       = VK_NULL_HANDLE;
    VmaAllocation  pageToSlotAlloc_        = nullptr;
    void*          pageToSlotMapped_       = nullptr;
    u32            pageToSlotPageCount_    = 0u;
    u32            pageToSlotBindlessSlot_ = UINT32_MAX;
    u64            pageToSlotBufferBytes_  = 0ull;

    // M4.5 pageFirstDagNode mirror. DEVICE_LOCAL, populated once per asset
    // load via a transient staging buffer. One u32 per page, value =
    // `MpPageEntry::firstDagNodeIdx` (global DAG node index of the page's
    // first cluster). Shaders read this to recover `localClusterIdx =
    // globalDagNodeIdx - pageFirstDagNodeBuffer[pageId]`.
    VkBuffer       pageFirstDagNodeBuffer_       = VK_NULL_HANDLE;
    VmaAllocation  pageFirstDagNodeAlloc_        = nullptr;
    u32            pageFirstDagNodeBindlessSlot_ = UINT32_MAX;
    u64            pageFirstDagNodeBufferBytes_  = 0ull;

    // M3.3-deferred DAG node SSBO. DEVICE_LOCAL, populated once per asset
    // load via a transient staging buffer. Layout: dagNodeCount × 48 B
    // (3×float4 per node) matching the shader's MpDagNode format. The
    // source array is assembled by MpAssetReader::assembleRuntimeDagNodes()
    // which joins on-disk MpDagNode.pageId with per-page ClusterOnDisk cone
    // + bounds data (cone fields are NOT in the 36 B on-disk MpDagNode —
    // they live in the 76 B ClusterOnDisk entries inside each page).
    VkBuffer       dagNodeBuffer_       = VK_NULL_HANDLE;
    VmaAllocation  dagNodeAlloc_        = nullptr;
    u32            dagNodeBindlessSlot_ = UINT32_MAX;
    u64            dagNodeBufferBytes_  = 0ull;

    // Refresh the GPU mirror from pageToSlot_. Called at the end of
    // beginFrame() after the completion pump. No-op when no buffer is
    // attached.
    void refreshPageToSlotBuffer_();

    // Release the pageToSlot GPU buffer. Does NOT release the bindless
    // slot — the caller (Renderer) owns registration and is responsible
    // for a matching releaseStorageBuffer() on its DescriptorAllocator.
    // Idempotent.
    void detachPageToSlotBuffer_();

    // Release the pageFirstDagNode GPU buffer. Matches the detach pattern
    // of pageToSlot — bindless slot lifecycle is owned by the caller.
    // Idempotent.
    void detachPageFirstDagNodeBuffer_();

    // Release the DAG node GPU buffer. Matches the detach pattern of
    // pageFirstDagNode — bindless slot lifecycle is owned by the caller.
    // Idempotent.
    void detachDagNodeBuffer_();
};

const char* micropolyStreamingErrorKindString(MicropolyStreamingErrorKind kind);

}  // namespace enigma::renderer::micropoly
