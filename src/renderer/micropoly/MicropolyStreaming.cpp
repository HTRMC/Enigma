// MicropolyStreaming.cpp
// =======================
// See header for the contract. This TU owns the per-frame state machine
// that ties RequestQueue drain -> ResidencyManager dedup -> AsyncIOWorker
// dispatch -> PageCache allocation -> transfer-queue upload (M2.4a).

#include "renderer/micropoly/MicropolyStreaming.h"

#include "asset/MpAssetFormat.h"
#include "asset/MpAssetReader.h"
#include "core/Assert.h"
#include "gfx/Allocator.h"
#include "gfx/Device.h"

// VMA: the engine's Allocator.cpp stamps VMA_IMPLEMENTATION exactly once.
// Here we just need the types + staging-buffer entry points.
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

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <span>
#include <utility>

namespace enigma::renderer::micropoly {

namespace {

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

const char* micropolyStreamingErrorKindString(MicropolyStreamingErrorKind kind) {
    switch (kind) {
        case MicropolyStreamingErrorKind::InvalidOptions:         return "InvalidOptions";
        case MicropolyStreamingErrorKind::ResidencyInitFailed:    return "ResidencyInitFailed";
        case MicropolyStreamingErrorKind::PageCacheInitFailed:    return "PageCacheInitFailed";
        case MicropolyStreamingErrorKind::RequestQueueInitFailed: return "RequestQueueInitFailed";
        case MicropolyStreamingErrorKind::AsyncIOInitFailed:      return "AsyncIOInitFailed";
    }
    return "?";
}

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

MicropolyStreaming::MicropolyStreaming(gfx::Device& device,
                                       gfx::Allocator& allocator,
                                       MicropolyStreamingOptions opts,
                                       PageCache pageCache,
                                       RequestQueue requestQueue)
    : device_(&device),
      allocator_(&allocator),
      opts_(std::move(opts)),
      residency_(opts_.residency),
      pageCache_(std::move(pageCache)),
      requestQueue_(std::move(requestQueue)) {
}

MicropolyStreaming::~MicropolyStreaming() {
    // Order matters:
    //   1) Stop the IO worker first so its callback cannot fire into a
    //      partially-destroyed orchestrator.
    //   2) Drain any pending transfer-queue submits before destroying
    //      the semaphore/pool (vkQueueWaitIdle on transferQueue_).
    //   3) Tear down transfer resources.
    //   4) Release the M3.3 pageToSlot GPU buffer (owns bindless slot).
    if (asyncIO_) {
        asyncIO_->shutdown();
        asyncIO_.reset();
    }
    destroyTransferResources_();
    detachPageToSlotBuffer_();
    // Destroy DAG node SSBO BEFORE pageFirstDagNode so the two teardowns
    // don't race on VMA allocator access from the dtor (both target the
    // same VmaAllocator). Mirrors the M4.5 ordering comment.
    detachDagNodeBuffer_();
    detachPageFirstDagNodeBuffer_();
}

std::expected<std::unique_ptr<MicropolyStreaming>, MicropolyStreamingError>
MicropolyStreaming::create(gfx::Device& device,
                           gfx::Allocator& allocator,
                           MicropolyStreamingOptions opts) {
    if (opts.mpaFilePath.empty()) {
        return std::unexpected(MicropolyStreamingError{
            MicropolyStreamingErrorKind::InvalidOptions,
            "mpaFilePath is empty",
        });
    }
    if (opts.reader == nullptr) {
        return std::unexpected(MicropolyStreamingError{
            MicropolyStreamingErrorKind::InvalidOptions,
            "reader pointer is null",
        });
    }

    // Bring up RequestQueue first. If the device is too memory-starved to
    // even allocate the ring buffer, there is no point spinning up the rest.
    auto rqRes = RequestQueue::create(allocator, opts.requestQueue);
    if (!rqRes) {
        return std::unexpected(MicropolyStreamingError{
            MicropolyStreamingErrorKind::RequestQueueInitFailed,
            formatDetail("RequestQueue::create: %s / %s",
                         requestQueueErrorKindString(rqRes.error().kind),
                         rqRes.error().detail.c_str()),
        });
    }

    // M3.0 prereq A — wire the PageCache's sharing mode to the device's
    // actual queue families. When the caller didn't supply them, infer from
    // the Device so the discrete-GPU case (dedicated transfer family) flips
    // to VK_SHARING_MODE_CONCURRENT without callers needing to pre-fill
    // opts.pageCache.*QueueFamily. Unified-queue consumer GPUs keep EXCLUSIVE.
    PageCacheOptions pageCacheOpts = opts.pageCache;
    if (pageCacheOpts.graphicsQueueFamily == UINT32_MAX) {
        pageCacheOpts.graphicsQueueFamily = device.graphicsQueueFamily();
    }
    if (pageCacheOpts.transferQueueFamily == UINT32_MAX) {
        pageCacheOpts.transferQueueFamily =
            device.transferQueueFamily().has_value()
                ? *device.transferQueueFamily()
                : device.graphicsQueueFamily();
    }

    auto pcRes = PageCache::create(allocator, pageCacheOpts);
    if (!pcRes) {
        return std::unexpected(MicropolyStreamingError{
            MicropolyStreamingErrorKind::PageCacheInitFailed,
            formatDetail("PageCache::create: %s / %s",
                         pageCacheErrorKindString(pcRes.error().kind),
                         pcRes.error().detail.c_str()),
        });
    }

    // Construct via unique_ptr so AsyncIOWorker's completion callback can
    // capture a stable pointer.
    std::unique_ptr<MicropolyStreaming> out(new MicropolyStreaming(
        device, allocator, std::move(opts),
        std::move(*pcRes), std::move(*rqRes)));

    // Transfer-queue bring-up. On failure, log + continue — the streaming
    // subsystem can still run in CPU-only mode (no uploads), which tests
    // that use the Device shim rely on.
    (void)out->initTransferResources_();

    // Wire the AsyncIOWorker with our orchestrator-owned completion sink.
    AsyncIOWorkerOptions aio = out->opts_.asyncIO;
    aio.mpaFilePath = out->opts_.mpaFilePath;
    aio.onComplete  = [raw = out.get()](PageCompletion c) {
        raw->onWorkerComplete_(std::move(c));
    };
    try {
        out->asyncIO_ = std::make_unique<AsyncIOWorker>(std::move(aio));
    } catch (const std::exception& e) {
        return std::unexpected(MicropolyStreamingError{
            MicropolyStreamingErrorKind::AsyncIOInitFailed,
            formatDetail("AsyncIOWorker ctor threw: %s", e.what()),
        });
    } catch (...) {
        return std::unexpected(MicropolyStreamingError{
            MicropolyStreamingErrorKind::AsyncIOInitFailed,
            "AsyncIOWorker ctor threw (unknown)",
        });
    }

    return out;
}

// ---------------------------------------------------------------------------
// Transfer-queue bring-up / tear-down
// ---------------------------------------------------------------------------

bool MicropolyStreaming::initTransferResources_() {
    // device_ may be a test shim (zeroed bytes for headless tests without a
    // real Device). We can detect that by checking that logical() returns
    // a plausible handle and the optional transfer-family accessor behaves.
    // If anything is not-a-handle, bail cleanly — tests opt into transfer
    // work by using Device::adopt() and a real VkDevice.
    if (device_ == nullptr) return false;
    VkDevice logical = device_->logical();
    if (logical == VK_NULL_HANDLE) return false;

    // Pick transfer family — dedicated if present, graphics otherwise.
    if (device_->transferQueueFamily().has_value() &&
        device_->transferQueue().has_value()) {
        transferFamily_ = *device_->transferQueueFamily();
        transferQueue_  = *device_->transferQueue();
    } else {
        transferFamily_ = device_->graphicsQueueFamily();
        transferQueue_  = device_->graphicsQueue();
    }
    if (transferQueue_ == VK_NULL_HANDLE) return false;

    // Command pool for per-frame transient cmd buffers.
    VkCommandPoolCreateInfo pci{};
    pci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.queueFamilyIndex = transferFamily_;
    pci.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT
                         | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(logical, &pci, nullptr, &transferCmdPool_) != VK_SUCCESS) {
        transferCmdPool_ = VK_NULL_HANDLE;
        return false;
    }

    // Timeline semaphore. Matches M2.4 architect recommendation: not a
    // binary semaphore. initialValue=0; each successful submit signals
    // the next monotonic uploadCounter_ value.
    VkSemaphoreTypeCreateInfo tsci{};
    tsci.sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    tsci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    tsci.initialValue  = 0ull;

    VkSemaphoreCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    sci.pNext = &tsci;
    if (vkCreateSemaphore(logical, &sci, nullptr, &uploadSema_) != VK_SUCCESS) {
        uploadSema_ = VK_NULL_HANDLE;
        vkDestroyCommandPool(logical, transferCmdPool_, nullptr);
        transferCmdPool_ = VK_NULL_HANDLE;
        return false;
    }

    // Staging ring. Each slot is slotBytes large; HOST_VISIBLE + coherent.
    // M2.4a strategy: simple round-robin; per-slot readyValue tracks when
    // the ring slot can be reused.
    const VkDeviceSize slotBytes = pageCache_.slotBytes();
    for (u32 i = 0; i < kStagingRingSize; ++i) {
        VkBufferCreateInfo bci{};
        bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size        = slotBytes;
        bci.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo aci{};
        aci.usage     = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        aci.flags     = VMA_ALLOCATION_CREATE_MAPPED_BIT
                      | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
        aci.pUserData = const_cast<char*>("micropoly.streaming.staging");

        VmaAllocationInfo info{};
        if (vmaCreateBuffer(allocator_->handle(), &bci, &aci,
                            &staging_[i].buffer, &staging_[i].allocation,
                            &info) != VK_SUCCESS) {
            destroyTransferResources_();
            return false;
        }
        staging_[i].mapped     = info.pMappedData;
        staging_[i].readyValue = 0ull;
        if (staging_[i].mapped == nullptr) {
            destroyTransferResources_();
            return false;
        }
    }

    return true;
}

void MicropolyStreaming::destroyTransferResources_() {
    if (device_ == nullptr) return;
    VkDevice logical = device_->logical();
    if (logical == VK_NULL_HANDLE) {
        // Defensive — nothing was ever created.
        for (auto& s : staging_) s = {};
        pendingCmdBufs_.clear();
        transferQueue_    = VK_NULL_HANDLE;
        transferCmdPool_  = VK_NULL_HANDLE;
        uploadSema_       = VK_NULL_HANDLE;
        return;
    }

    // Wait for any pending transfer submits to retire before tearing the
    // command pool / semaphore down.
    if (uploadSema_ != VK_NULL_HANDLE && uploadCounter_ > 0ull) {
        VkSemaphoreWaitInfo wi{};
        wi.sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        wi.semaphoreCount = 1u;
        wi.pSemaphores    = &uploadSema_;
        wi.pValues        = &uploadCounter_;
        (void)vkWaitSemaphores(logical, &wi,
                               2ull * 1000ull * 1000ull * 1000ull);
    }

    if (transferCmdPool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(logical, transferCmdPool_, nullptr);
        transferCmdPool_ = VK_NULL_HANDLE;
    }
    pendingCmdBufs_.clear();

    if (uploadSema_ != VK_NULL_HANDLE) {
        vkDestroySemaphore(logical, uploadSema_, nullptr);
        uploadSema_ = VK_NULL_HANDLE;
    }

    if (allocator_ != nullptr) {
        for (auto& s : staging_) {
            if (s.buffer != VK_NULL_HANDLE) {
                vmaDestroyBuffer(allocator_->handle(), s.buffer, s.allocation);
            }
            s = {};
        }
    } else {
        for (auto& s : staging_) s = {};
    }

    transferQueue_ = VK_NULL_HANDLE;
}

// ---------------------------------------------------------------------------
// Completion callback
// ---------------------------------------------------------------------------

void MicropolyStreaming::onWorkerComplete_(PageCompletion c) {
    std::lock_guard<std::mutex> lk(completionMutex_);
    pendingCompletions_.push_back(std::move(c));
}

// ---------------------------------------------------------------------------
// Completion pump (shared by beginFrame + drainCompletionsForTest)
// ---------------------------------------------------------------------------

MicropolyStreaming::CompletionPumpResult
MicropolyStreaming::processCompletions_(bool recordUploads) {
    CompletionPumpResult out{};

    std::vector<PageCompletion> completions;
    {
        std::lock_guard<std::mutex> lk(completionMutex_);
        completions.swap(pendingCompletions_);
    }
    out.completed = static_cast<u32>(completions.size());

    // Nothing to do — return early so we don't submit an empty cmd buffer.
    if (completions.empty()) return out;

    const bool canUpload =
        recordUploads
        && device_ != nullptr
        && device_->logical()    != VK_NULL_HANDLE
        && transferCmdPool_      != VK_NULL_HANDLE
        && uploadSema_           != VK_NULL_HANDLE
        && transferQueue_        != VK_NULL_HANDLE;

    // Allocate a cmd buffer + collect pending copy regions. We record into
    // a single cmd buffer per completion batch; this keeps submit overhead
    // at one per frame even when N pages arrive together.
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    bool cmdBegun = false;

    if (canUpload) {
        VkCommandBufferAllocateInfo cai{};
        cai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cai.commandPool        = transferCmdPool_;
        cai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1u;
        if (vkAllocateCommandBuffers(device_->logical(), &cai, &cmd) != VK_SUCCESS) {
            cmd = VK_NULL_HANDLE;
        }
        if (cmd != VK_NULL_HANDLE) {
            VkCommandBufferBeginInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            if (vkBeginCommandBuffer(cmd, &bi) == VK_SUCCESS) {
                cmdBegun = true;
            }
        }
    }

    // The timeline value this batch will signal. Incremented once up front
    // so every per-slot readyValue we stamp matches the submit. Note: we do
    // NOT stamp ss.readyValue inside the per-completion loop below — stamping
    // during the loop creates a wrap deadlock where a later iteration in the
    // SAME batch sees readyValue == nextUploadVal (a value the semaphore
    // has not reached yet, and will not reach until after this function's
    // vkQueueSubmit returns). Instead, collect the slot indices this batch
    // used and stamp them in one pass AFTER the submit succeeds (Phase-4
    // architect latent-bug fix).
    const u64 nextUploadVal = canUpload ? (uploadCounter_ + 1ull) : uploadCounter_;

    const u32 slotBytes = pageCache_.slotBytes();

    // Slots used this batch — stamped with their readyValue post-submit.
    std::vector<u32> usedSlots;
    usedSlots.reserve(completions.size());

    for (PageCompletion& c : completions) {
        if (!c.success) {
            ++out.uploadsFailed;
            continue;
        }

        // Security pattern note (M2.4 closeout #5): cap decompressed size at
        // slotBytes. Pages larger than a slot cannot be uploaded by this
        // design — surface as a failure rather than silently corrupting the
        // page cache.
        if (c.decompressedData.size() > slotBytes) {
            ++out.uploadsFailed;
            continue;
        }

        // M4.5: multi-cluster pages are fully supported — the per-page
        // firstDagNodeIdx SSBO (pageFirstDagNodeBuffer, attached at asset
        // load via attachPageFirstDagNodeBuffer) lets shaders derive
        // localClusterIdx from the global DAG node index. The M3.3
        // warn-once has been removed.

        // Residency insert returns the ordered event log — evictions
        // first, then the terminal Inserted/Touched.
        const u32 sizeBytes = static_cast<u32>(c.decompressedData.size());
        auto insertRes = residency_.insert(c.pageId, sizeBytes);

        for (const auto& ev : insertRes.events) {
            if (ev.kind == ResidencyEventKind::Evicted) {
                std::lock_guard<std::mutex> lk(pageMapMutex_);
                auto it = pageToSlot_.find(ev.pageId);
                if (it != pageToSlot_.end()) {
                    (void)pageCache_.free(it->second);
                    pageToSlot_.erase(it);
                }
                ++out.evictions;
            }
        }

        if (insertRes.wasAlreadyResident) continue;

        auto slotRes = pageCache_.allocate();
        if (!slotRes) {
            ++out.slotAllocFailures;
            continue;
        }
        const u32 slot = *slotRes;
        {
            std::lock_guard<std::mutex> lk(pageMapMutex_);
            pageToSlot_[c.pageId] = slot;
        }

        // Schedule the upload. If we can't (no transfer resources), the
        // slot is still reserved so tests can observe slotForPage() — the
        // actual VkBuffer bytes are undefined in that case.
        if (!cmdBegun) {
            continue;
        }

        // Pick a staging ring slot. Wait on its readyValue if it's in use.
        const u32 slotIdx = stagingNextIdx_;
        StagingSlot& ss = staging_[slotIdx];
        stagingNextIdx_ = (stagingNextIdx_ + 1u) % kStagingRingSize;
        if (ss.readyValue > 0ull) {
            VkSemaphoreWaitInfo wi{};
            wi.sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
            wi.semaphoreCount = 1u;
            wi.pSemaphores    = &uploadSema_;
            wi.pValues        = &ss.readyValue;
            // Fix D: check wait result; on timeout/error the slot may still
            // be in-flight, so skip this completion rather than stomp a live
            // transfer. Surface as an upload failure.
            const VkResult waitRes = vkWaitSemaphores(device_->logical(), &wi,
                                                      2ull * 1000ull * 1000ull * 1000ull);
            if (waitRes != VK_SUCCESS) {
                std::fprintf(stderr,
                    "[micropoly] staging-slot wait failed (%d); skipping upload\n",
                    static_cast<int>(waitRes));
                ++out.uploadsFailed;
                continue;
            }
        }

        std::memcpy(ss.mapped, c.decompressedData.data(), sizeBytes);

        VkBufferCopy region{};
        region.srcOffset = 0u;
        region.dstOffset = pageCache_.slotByteOffset(slot);
        region.size      = sizeBytes;
        vkCmdCopyBuffer(cmd, ss.buffer, pageCache_.buffer(), 1u, &region);

        // Stamp AFTER submit succeeds — see comment above usedSlots. Record
        // the slot index now so the post-submit pass knows which to update.
        usedSlots.push_back(slotIdx);
        ++out.uploadsScheduled;
    }

    // If we recorded anything, submit.
    if (cmdBegun) {
        vkEndCommandBuffer(cmd);

        if (out.uploadsScheduled > 0u) {
            // Free any pending cmd buffers whose batch has already retired.
            // Hoist vkGetSemaphoreCounterValue out of the loop (Phase-4
            // CR LOW-2) and use pop_front on the deque (Phase-4 CR HIGH).
            u64 completedVal = 0ull;
            vkGetSemaphoreCounterValue(device_->logical(), uploadSema_, &completedVal);
            while (!pendingCmdBufs_.empty() &&
                   pendingCmdBufs_.front().waitVal <= completedVal) {
                vkFreeCommandBuffers(device_->logical(), transferCmdPool_,
                                     1u, &pendingCmdBufs_.front().cmd);
                pendingCmdBufs_.pop_front();
            }

            const u64 signalVal = nextUploadVal;
            VkTimelineSemaphoreSubmitInfo ts{};
            ts.sType                     = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
            ts.signalSemaphoreValueCount = 1u;
            ts.pSignalSemaphoreValues    = &signalVal;

            VkSubmitInfo si{};
            si.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si.pNext                = &ts;
            si.commandBufferCount   = 1u;
            si.pCommandBuffers      = &cmd;
            si.signalSemaphoreCount = 1u;
            si.pSignalSemaphores    = &uploadSema_;

            if (vkQueueSubmit(transferQueue_, 1u, &si, VK_NULL_HANDLE) == VK_SUCCESS) {
                uploadCounter_ = signalVal;
                pendingCmdBufs_.push_back({cmd, signalVal});
                cmd = VK_NULL_HANDLE;  // ownership transferred to pendingCmdBufs_

                // Fix B: NOW stamp readyValue on the slots we used. Only
                // reachable after submit success — means these slots will
                // be freed when the semaphore reaches signalVal, which is
                // a value strictly greater than anything we could have
                // waited on inside the loop above. No wrap deadlock.
                for (u32 slotIdx : usedSlots) {
                    staging_[slotIdx].readyValue = signalVal;
                }
            }
        }

        // If cmd is still owned here (no-op submit or submit failure), free it.
        if (cmd != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(device_->logical(), transferCmdPool_, 1u, &cmd);
        }
    }

    return out;
}

// ---------------------------------------------------------------------------
// Per-frame pump
// ---------------------------------------------------------------------------

MicropolyStreaming::FrameStats MicropolyStreaming::beginFrame() {
    FrameStats out{};

    // Advance the residency manager's LRU tick so touches/evictions in this
    // frame stamp a fresh timestamp.
    residency_.beginFrame();

    // 1) Drain the GPU-produced request ring.
    //
    // BARRIER NOTE (M2.4a): for M3+ when a compute shader writes the ring
    // buffer, a pipeline barrier with dstStageMask=VK_PIPELINE_STAGE_HOST_BIT,
    // dstAccessMask=VK_ACCESS_HOST_READ_BIT | VK_ACCESS_HOST_WRITE_BIT must
    // be recorded on the graphics queue before this drain to make the
    // compute writes visible to the host read. In M2.4a there is no producer
    // yet, so the barrier is a no-op — but the call site in Renderer.cpp
    // documents the expectation.
    std::vector<u32> drained = requestQueue_.drain();
    out.drained = static_cast<u32>(drained.size());

    // 2) Dedup + route. scheduledThisFrame_ survives only for this call.
    scheduledThisFrame_.clear();

    const auto pageTable = opts_.reader != nullptr
                         ? opts_.reader->pageTable()
                         : std::span<const asset::MpPageEntry>{};

    for (u32 pageId : drained) {
        if (!scheduledThisFrame_.insert(pageId).second) {
            // Duplicate within the same drain — skip.
            continue;
        }
        ++out.dedupedFresh;

        if (residency_.isResident(pageId)) {
            ++out.cacheHits;
            continue;
        }

        if (pageId >= pageTable.size()) {
            ++out.lookupFailures;
            continue;
        }
        const asset::MpPageEntry& pe = pageTable[pageId];

        PageRequest req{};
        req.pageId           = pageId;
        req.fileOffset       = pe.payloadByteOffset;
        req.compressedSize   = pe.compressedSize;
        req.decompressedSize = pe.decompressedSize;
        if (asyncIO_ && asyncIO_->enqueue(req)) {
            ++out.queuedForIO;
        } else {
            ++out.enqueueFailures;
        }
    }

    // 3) Process any completions delivered since last frame.
    auto pump = processCompletions_(/*recordUploads=*/true);
    out.completed         = pump.completed;
    out.uploadsScheduled  = pump.uploadsScheduled;
    out.uploadsFailed     = pump.uploadsFailed;
    out.evictions         = pump.evictions;
    out.slotAllocFailures = pump.slotAllocFailures;

    // 4) Refresh the GPU pageId -> slotIndex mirror for M3.3 HW raster.
    //    No-op when no buffer is attached (disabled config or pre-M3.3
    //    test harness).
    refreshPageToSlotBuffer_();

    return out;
}

u32 MicropolyStreaming::slotForPage(u32 pageId) const {
    std::lock_guard<std::mutex> lk(pageMapMutex_);
    auto it = pageToSlot_.find(pageId);
    return it == pageToSlot_.end() ? UINT32_MAX : it->second;
}

u32 MicropolyStreaming::drainCompletionsForTest() {
    // Test entry point: drive the same pump, but let it submit real uploads
    // when a real device is attached. Tests with the Device shim end up
    // with recordUploads effectively gated off (transferCmdPool_ is never
    // created when logical() returns VK_NULL_HANDLE — see initTransferResources_).
    return processCompletions_(/*recordUploads=*/true).completed;
}

// ---------------------------------------------------------------------------
// M3.3 pageId -> slot mirror (attach / detach / refresh)
// ---------------------------------------------------------------------------

bool MicropolyStreaming::attachPageToSlotBuffer(u32 pageCount) {
    // Re-attach-on-resize path: if a buffer already exists at a different
    // size, tear it down first. The bindless slot is NOT released here —
    // the caller owns the DescriptorAllocator side and is responsible for
    // re-registering the new VkBuffer against the existing slot (or
    // releasing + re-registering).
    if (pageToSlotBuffer_ != VK_NULL_HANDLE && pageToSlotPageCount_ != pageCount) {
        detachPageToSlotBuffer_();
    }
    if (pageToSlotBuffer_ != VK_NULL_HANDLE) {
        // Same page count — no-op.
        return true;
    }
    if (allocator_ == nullptr) return false;
    if (pageCount == 0u) return false;

    // Size the buffer for `pageCount` u32 entries. Round up to a 16-byte
    // stride so a StructuredBuffer<float4> view stays legal (the shader
    // reads via float4 components, see mp_raster.task.hlsl::slotForPage).
    const VkDeviceSize entryBytes  = pageCount * sizeof(u32);
    const VkDeviceSize paddedBytes = (entryBytes + 15u) & ~static_cast<VkDeviceSize>(15u);

    VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size        = paddedBytes;
    bci.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                    | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    aci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT
              | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    VmaAllocationInfo info{};
    if (vmaCreateBuffer(allocator_->handle(), &bci, &aci,
                        &pageToSlotBuffer_, &pageToSlotAlloc_,
                        &info) != VK_SUCCESS) {
        pageToSlotBuffer_ = VK_NULL_HANDLE;
        pageToSlotAlloc_  = nullptr;
        return false;
    }
    pageToSlotMapped_      = info.pMappedData;
    pageToSlotPageCount_   = pageCount;
    pageToSlotBufferBytes_ = paddedBytes;

    if (pageToSlotMapped_ == nullptr) {
        vmaDestroyBuffer(allocator_->handle(), pageToSlotBuffer_, pageToSlotAlloc_);
        pageToSlotBuffer_      = VK_NULL_HANDLE;
        pageToSlotAlloc_       = nullptr;
        pageToSlotPageCount_   = 0u;
        pageToSlotBufferBytes_ = 0ull;
        return false;
    }

    // Initialize to "not resident everywhere" (UINT32_MAX sentinel). The
    // shader treats UINT32_MAX as "skip cluster" which is the correct
    // initial condition before any pages arrive.
    std::memset(pageToSlotMapped_, 0xFF, paddedBytes);

    // Refresh once now — if beginFrame() has already populated pageToSlot_
    // in a prior frame (e.g. attach-after-first-frame), the caller shouldn't
    // have to wait another frame for the GPU mirror to catch up.
    refreshPageToSlotBuffer_();

    return true;
}

void MicropolyStreaming::refreshPageToSlotBuffer_() {
    if (pageToSlotBuffer_ == VK_NULL_HANDLE) return;
    if (pageToSlotMapped_ == nullptr) return;
    if (pageToSlotPageCount_ == 0u) return;

    // Stamp every entry to UINT32_MAX first — the map only has residents,
    // so anything absent (evicted or never loaded) must read as "not
    // resident" from the shader's perspective.
    u32* out = static_cast<u32*>(pageToSlotMapped_);
    const u32 count = pageToSlotPageCount_;
    for (u32 i = 0u; i < count; ++i) {
        out[i] = 0xFFFFFFFFu;
    }

    std::lock_guard<std::mutex> lk(pageMapMutex_);
    for (const auto& kv : pageToSlot_) {
        const u32 pageId = kv.first;
        const u32 slot   = kv.second;
        if (pageId < count) {
            out[pageId] = slot;
        }
        // pageIds >= count should be impossible (pageCount is an upper
        // bound from MpAssetHeader::pageCount) but ignore them rather than
        // writing past the allocation.
    }
    // Host-coherent — no flush needed. The GPU picks up the writes on the
    // next barrier that sources HOST_WRITE.
}

void MicropolyStreaming::detachPageToSlotBuffer_() {
    if (pageToSlotBuffer_ != VK_NULL_HANDLE && allocator_ != nullptr) {
        vmaDestroyBuffer(allocator_->handle(), pageToSlotBuffer_, pageToSlotAlloc_);
    }
    pageToSlotBuffer_      = VK_NULL_HANDLE;
    pageToSlotAlloc_       = nullptr;
    pageToSlotMapped_      = nullptr;
    pageToSlotPageCount_   = 0u;
    pageToSlotBufferBytes_ = 0ull;
    // Intentionally do NOT reset pageToSlotBindlessSlot_ — the caller
    // (Renderer) owns the descriptor-allocator slot lifecycle and needs
    // the stamped value to issue its own releaseStorageBuffer().
}

// ---------------------------------------------------------------------------
// M4.5 pageFirstDagNode buffer (attach / detach)
// ---------------------------------------------------------------------------

bool MicropolyStreaming::attachPageFirstDagNodeBuffer(
    std::span<const u32> firstDagNodeIdxPerPage) {
    // M4.5 Phase 4 MEDIUM fix: re-attach-on-resize path. If a buffer already
    // exists with a DIFFERENT padded byte size (e.g. asset reload changed
    // pageCount), tear it down first so a fresh buffer is allocated.
    // Otherwise the old buffer silently retains stale data. Mirrors the
    // pattern in attachPageToSlotBuffer.
    if (pageFirstDagNodeBuffer_ != VK_NULL_HANDLE) {
        const VkDeviceSize requestedEntryBytes =
            static_cast<VkDeviceSize>(firstDagNodeIdxPerPage.size()) * sizeof(u32);
        const VkDeviceSize requestedPaddedBytes =
            (requestedEntryBytes + 15ull) & ~static_cast<VkDeviceSize>(15ull);
        if (requestedPaddedBytes != pageFirstDagNodeBufferBytes_) {
            detachPageFirstDagNodeBuffer_();
        } else {
            return true; // same size, idempotent
        }
    }
    if (allocator_ == nullptr) return false;
    if (firstDagNodeIdxPerPage.empty()) return false;
    if (device_ == nullptr) return false;

    VkDevice logical = device_->logical();
    if (logical == VK_NULL_HANDLE) return false;

    const VkDeviceSize entryBytes  =
        static_cast<VkDeviceSize>(firstDagNodeIdxPerPage.size()) * sizeof(u32);
    // Round up to a 16-byte stride so a StructuredBuffer<float4> view stays
    // legal, matching the pageToSlot buffer pattern.
    const VkDeviceSize paddedBytes = (entryBytes + 15u) & ~static_cast<VkDeviceSize>(15u);

    // DEVICE_LOCAL destination buffer.
    VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size        = paddedBytes;
    bci.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                    | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    aci.pUserData = const_cast<char*>("micropoly.streaming.pageFirstDagNode");

    if (vmaCreateBuffer(allocator_->handle(), &bci, &aci,
                        &pageFirstDagNodeBuffer_, &pageFirstDagNodeAlloc_,
                        nullptr) != VK_SUCCESS) {
        pageFirstDagNodeBuffer_ = VK_NULL_HANDLE;
        pageFirstDagNodeAlloc_  = nullptr;
        return false;
    }
    pageFirstDagNodeBufferBytes_ = paddedBytes;

    // Host-visible staging buffer.
    VkBuffer      stagingBuf   = VK_NULL_HANDLE;
    VmaAllocation stagingAlloc = VK_NULL_HANDLE;
    void*         stagingPtr   = nullptr;
    {
        VkBufferCreateInfo sbci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        sbci.size        = paddedBytes;
        sbci.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        sbci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo saci{};
        saci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        saci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT
                   | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

        VmaAllocationInfo sinfo{};
        if (vmaCreateBuffer(allocator_->handle(), &sbci, &saci,
                            &stagingBuf, &stagingAlloc, &sinfo) != VK_SUCCESS
            || sinfo.pMappedData == nullptr) {
            vmaDestroyBuffer(allocator_->handle(), pageFirstDagNodeBuffer_,
                             pageFirstDagNodeAlloc_);
            pageFirstDagNodeBuffer_     = VK_NULL_HANDLE;
            pageFirstDagNodeAlloc_      = nullptr;
            pageFirstDagNodeBufferBytes_ = 0ull;
            return false;
        }
        stagingPtr = sinfo.pMappedData;
    }

    // Copy caller data; zero the tail padding.
    std::memcpy(stagingPtr, firstDagNodeIdxPerPage.data(), entryBytes);
    if (paddedBytes > entryBytes) {
        std::memset(static_cast<u8*>(stagingPtr) + entryBytes, 0,
                    static_cast<std::size_t>(paddedBytes - entryBytes));
    }

    // Record + submit a one-shot transfer on the (possibly shared) transfer
    // queue. If transferCmdPool_ is absent (test shims without a real queue)
    // we fall back to the graphics queue via a transient pool. Either path
    // vkQueueWaitIdles so the DEVICE_LOCAL buffer is populated before return.
    VkQueue       useQueue  = transferQueue_ != VK_NULL_HANDLE
                            ? transferQueue_ : device_->graphicsQueue();
    u32           useFamily = transferCmdPool_ != VK_NULL_HANDLE
                            ? transferFamily_ : device_->graphicsQueueFamily();
    VkCommandPool localPool = VK_NULL_HANDLE;
    VkCommandPool cmdPool   = transferCmdPool_;

    if (cmdPool == VK_NULL_HANDLE) {
        VkCommandPoolCreateInfo pci{};
        pci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pci.queueFamilyIndex = useFamily;
        pci.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        if (vkCreateCommandPool(logical, &pci, nullptr, &localPool) != VK_SUCCESS) {
            vmaDestroyBuffer(allocator_->handle(), stagingBuf, stagingAlloc);
            vmaDestroyBuffer(allocator_->handle(), pageFirstDagNodeBuffer_,
                             pageFirstDagNodeAlloc_);
            pageFirstDagNodeBuffer_     = VK_NULL_HANDLE;
            pageFirstDagNodeAlloc_      = nullptr;
            pageFirstDagNodeBufferBytes_ = 0ull;
            return false;
        }
        cmdPool = localPool;
    }

    VkCommandBufferAllocateInfo cai{};
    cai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool        = cmdPool;
    cai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1u;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    bool ok = true;
    if (vkAllocateCommandBuffers(logical, &cai, &cmd) != VK_SUCCESS) {
        ok = false;
    }

    if (ok) {
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) {
            ok = false;
        }
    }

    if (ok) {
        VkBufferCopy region{};
        region.srcOffset = 0u;
        region.dstOffset = 0u;
        region.size      = paddedBytes;
        vkCmdCopyBuffer(cmd, stagingBuf, pageFirstDagNodeBuffer_, 1u, &region);

        if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
            ok = false;
        }
    }

    if (ok) {
        VkFenceCreateInfo fci{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        VkFence fence = VK_NULL_HANDLE;
        if (vkCreateFence(logical, &fci, nullptr, &fence) != VK_SUCCESS) {
            ok = false;
        } else {
            VkSubmitInfo si{};
            si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si.commandBufferCount = 1u;
            si.pCommandBuffers    = &cmd;
            if (vkQueueSubmit(useQueue, 1u, &si, fence) != VK_SUCCESS) {
                ok = false;
            } else {
                (void)vkWaitForFences(logical, 1u, &fence, VK_TRUE,
                                      5ull * 1000ull * 1000ull * 1000ull);
            }
            vkDestroyFence(logical, fence, nullptr);
        }
    }

    if (cmd != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(logical, cmdPool, 1u, &cmd);
    }
    if (localPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(logical, localPool, nullptr);
    }
    vmaDestroyBuffer(allocator_->handle(), stagingBuf, stagingAlloc);

    if (!ok) {
        vmaDestroyBuffer(allocator_->handle(), pageFirstDagNodeBuffer_,
                         pageFirstDagNodeAlloc_);
        pageFirstDagNodeBuffer_     = VK_NULL_HANDLE;
        pageFirstDagNodeAlloc_      = nullptr;
        pageFirstDagNodeBufferBytes_ = 0ull;
        return false;
    }

    return true;
}

void MicropolyStreaming::detachPageFirstDagNodeBuffer_() {
    if (pageFirstDagNodeBuffer_ != VK_NULL_HANDLE && allocator_ != nullptr) {
        vmaDestroyBuffer(allocator_->handle(), pageFirstDagNodeBuffer_,
                         pageFirstDagNodeAlloc_);
    }
    pageFirstDagNodeBuffer_      = VK_NULL_HANDLE;
    pageFirstDagNodeAlloc_       = nullptr;
    pageFirstDagNodeBufferBytes_ = 0ull;
    // Intentionally do NOT reset pageFirstDagNodeBindlessSlot_ — mirrors
    // the pageToSlot pattern. Caller owns bindless lifecycle.
}

// ---------------------------------------------------------------------------
// M3.3-deferred DAG node buffer (attach / detach)
// ---------------------------------------------------------------------------
// Wire format: caller passes bytes assembled by
// MpAssetReader::assembleRuntimeDagNodes() — one 48-byte runtime node per
// on-disk MpDagNode, containing center/radius + coneApex/coneCutoff +
// coneAxis/packed(pageId|lodLevel<<24). Staging upload mirrors the M4.5
// pageFirstDagNode pattern: transient staging buffer → DEVICE_LOCAL SSBO,
// waited on via a one-shot fence so the buffer is populated before return.

bool MicropolyStreaming::attachDagNodeBuffer(std::span<const u8> runtimeDagNodeBytes) {
    // Detach-before-reattach on size change. A re-attach at the same padded
    // size is idempotent (early-return). This covers asset reloads where the
    // DAG node count changes between bakes.
    if (dagNodeBuffer_ != VK_NULL_HANDLE) {
        const VkDeviceSize requestedEntryBytes =
            static_cast<VkDeviceSize>(runtimeDagNodeBytes.size());
        const VkDeviceSize requestedPaddedBytes =
            (requestedEntryBytes + 15ull) & ~static_cast<VkDeviceSize>(15ull);
        if (requestedPaddedBytes != dagNodeBufferBytes_) {
            detachDagNodeBuffer_();
        } else {
            return true;  // same size, idempotent
        }
    }
    if (allocator_ == nullptr) return false;
    if (runtimeDagNodeBytes.empty()) return false;
    if (device_ == nullptr) return false;

    VkDevice logical = device_->logical();
    if (logical == VK_NULL_HANDLE) return false;

    const VkDeviceSize entryBytes =
        static_cast<VkDeviceSize>(runtimeDagNodeBytes.size());
    // Round up to 16-byte stride so the SSBO view stays float4-aligned —
    // the shader reads via `StructuredBuffer<float4>` so misalignment would
    // surface as a validation-layer error on discrete GPUs.
    const VkDeviceSize paddedBytes =
        (entryBytes + 15u) & ~static_cast<VkDeviceSize>(15u);

    // DEVICE_LOCAL destination buffer.
    VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size        = paddedBytes;
    bci.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                    | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    aci.pUserData = const_cast<char*>("micropoly.streaming.dagNode");

    if (vmaCreateBuffer(allocator_->handle(), &bci, &aci,
                        &dagNodeBuffer_, &dagNodeAlloc_,
                        nullptr) != VK_SUCCESS) {
        dagNodeBuffer_ = VK_NULL_HANDLE;
        dagNodeAlloc_  = nullptr;
        return false;
    }
    dagNodeBufferBytes_ = paddedBytes;

    // Host-visible staging buffer.
    VkBuffer      stagingBuf   = VK_NULL_HANDLE;
    VmaAllocation stagingAlloc = VK_NULL_HANDLE;
    void*         stagingPtr   = nullptr;
    {
        VkBufferCreateInfo sbci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        sbci.size        = paddedBytes;
        sbci.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        sbci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo saci{};
        saci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        saci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT
                   | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

        VmaAllocationInfo sinfo{};
        if (vmaCreateBuffer(allocator_->handle(), &sbci, &saci,
                            &stagingBuf, &stagingAlloc, &sinfo) != VK_SUCCESS
            || sinfo.pMappedData == nullptr) {
            vmaDestroyBuffer(allocator_->handle(), dagNodeBuffer_, dagNodeAlloc_);
            dagNodeBuffer_      = VK_NULL_HANDLE;
            dagNodeAlloc_       = nullptr;
            dagNodeBufferBytes_ = 0ull;
            return false;
        }
        stagingPtr = sinfo.pMappedData;
    }

    // Copy caller data; zero the tail padding.
    std::memcpy(stagingPtr, runtimeDagNodeBytes.data(), entryBytes);
    if (paddedBytes > entryBytes) {
        std::memset(static_cast<u8*>(stagingPtr) + entryBytes, 0,
                    static_cast<std::size_t>(paddedBytes - entryBytes));
    }

    // Record + submit a one-shot transfer on the (possibly shared) transfer
    // queue. Fallback to graphics queue when the shim device has no
    // transferCmdPool_.
    VkQueue       useQueue  = transferQueue_ != VK_NULL_HANDLE
                            ? transferQueue_ : device_->graphicsQueue();
    u32           useFamily = transferCmdPool_ != VK_NULL_HANDLE
                            ? transferFamily_ : device_->graphicsQueueFamily();
    VkCommandPool localPool = VK_NULL_HANDLE;
    VkCommandPool cmdPool   = transferCmdPool_;

    if (cmdPool == VK_NULL_HANDLE) {
        VkCommandPoolCreateInfo pci{};
        pci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pci.queueFamilyIndex = useFamily;
        pci.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        if (vkCreateCommandPool(logical, &pci, nullptr, &localPool) != VK_SUCCESS) {
            vmaDestroyBuffer(allocator_->handle(), stagingBuf, stagingAlloc);
            vmaDestroyBuffer(allocator_->handle(), dagNodeBuffer_, dagNodeAlloc_);
            dagNodeBuffer_      = VK_NULL_HANDLE;
            dagNodeAlloc_       = nullptr;
            dagNodeBufferBytes_ = 0ull;
            return false;
        }
        cmdPool = localPool;
    }

    VkCommandBufferAllocateInfo cai{};
    cai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool        = cmdPool;
    cai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1u;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    bool ok = true;
    if (vkAllocateCommandBuffers(logical, &cai, &cmd) != VK_SUCCESS) {
        ok = false;
    }

    if (ok) {
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) {
            ok = false;
        }
    }

    if (ok) {
        VkBufferCopy region{};
        region.srcOffset = 0u;
        region.dstOffset = 0u;
        region.size      = paddedBytes;
        vkCmdCopyBuffer(cmd, stagingBuf, dagNodeBuffer_, 1u, &region);

        if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
            ok = false;
        }
    }

    if (ok) {
        VkFenceCreateInfo fci{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        VkFence fence = VK_NULL_HANDLE;
        if (vkCreateFence(logical, &fci, nullptr, &fence) != VK_SUCCESS) {
            ok = false;
        } else {
            VkSubmitInfo si{};
            si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si.commandBufferCount = 1u;
            si.pCommandBuffers    = &cmd;
            if (vkQueueSubmit(useQueue, 1u, &si, fence) != VK_SUCCESS) {
                ok = false;
            } else {
                (void)vkWaitForFences(logical, 1u, &fence, VK_TRUE,
                                      5ull * 1000ull * 1000ull * 1000ull);
            }
            vkDestroyFence(logical, fence, nullptr);
        }
    }

    if (cmd != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(logical, cmdPool, 1u, &cmd);
    }
    if (localPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(logical, localPool, nullptr);
    }
    vmaDestroyBuffer(allocator_->handle(), stagingBuf, stagingAlloc);

    if (!ok) {
        vmaDestroyBuffer(allocator_->handle(), dagNodeBuffer_, dagNodeAlloc_);
        dagNodeBuffer_      = VK_NULL_HANDLE;
        dagNodeAlloc_       = nullptr;
        dagNodeBufferBytes_ = 0ull;
        return false;
    }

    return true;
}

void MicropolyStreaming::detachDagNodeBuffer_() {
    if (dagNodeBuffer_ != VK_NULL_HANDLE && allocator_ != nullptr) {
        vmaDestroyBuffer(allocator_->handle(), dagNodeBuffer_, dagNodeAlloc_);
    }
    dagNodeBuffer_      = VK_NULL_HANDLE;
    dagNodeAlloc_       = nullptr;
    dagNodeBufferBytes_ = 0ull;
    // Intentionally do NOT reset dagNodeBindlessSlot_ — caller owns the
    // DescriptorAllocator slot lifecycle. Mirrors the pageFirstDagNode
    // pattern.
}

}  // namespace enigma::renderer::micropoly
