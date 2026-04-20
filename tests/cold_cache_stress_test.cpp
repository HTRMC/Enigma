// Integration test for MicropolyStreaming cold-cache behavior (M2.4a).
//
// Scenario: build a .mpa from DamagedHelmet, construct MicropolyStreaming
// with a live transfer queue + staging ring, request ALL pages in quick
// succession, wait for completions, and verify:
//   - every requested pageId is resident after completions drain,
//   - PageCache allocated the expected number of slots,
//   - each pageId has a non-sentinel slotForPage(),
//   - the transfer-queue timeline semaphore advanced at least once,
//   - the same sequence repeats cleanly across 3 reset cycles (new
//     streaming instance per cycle — idempotence check).
//
// This is the integration counterpart to the unit-level micropoly_streaming_test.
// Plain main, printf output, exit 0 on pass. Uses the same headless Vulkan
// bundle pattern as micropoly_streaming_test + page_cache_test, but adopts a
// real gfx::Device via Device::adopt() (M2.4 shim replacement).

#include "asset/MpAssetFormat.h"
#include "asset/MpAssetReader.h"
#include "renderer/micropoly/MicropolyStreaming.h"

#include "mpbake/ClusterBuilder.h"
#include "mpbake/DagBuilder.h"
#include "mpbake/GltfIngest.h"
#include "mpbake/PageWriter.h"

#include "core/Types.h"
#include "gfx/Allocator.h"
#include "gfx/Device.h"

#include <volk.h>

#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#define VMA_IMPLEMENTATION
#if defined(_MSC_VER)
    #pragma warning(push)
    #pragma warning(disable: 4100 4127 4189 4324 4505)
#endif
#include <vk_mem_alloc.h>
#if defined(_MSC_VER)
    #pragma warning(pop)
#endif

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <span>
#include <system_error>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

using enigma::u32;
using enigma::u64;
using enigma::u8;
using enigma::asset::MpAssetReader;
using enigma::mpbake::ClusterBuildOptions;
using enigma::mpbake::ClusterBuilder;
using enigma::mpbake::DagBuildOptions;
using enigma::mpbake::DagBuilder;
using enigma::mpbake::GltfIngest;
using enigma::mpbake::PageWriter;
using enigma::mpbake::PageWriteOptions;
using enigma::renderer::micropoly::MicropolyStreaming;
using enigma::renderer::micropoly::MicropolyStreamingOptions;
using enigma::renderer::micropoly::RequestQueueHeader;

namespace {

// ---------------------------------------------------------------------------
// Headless Vulkan bundle (mirrored from micropoly_streaming_test.cpp, with
// Vulkan1.2 timelineSemaphore feature enabled — required by M2.4a's upload
// semaphore).
// ---------------------------------------------------------------------------
struct VulkanBundle {
    VkInstance       instance       = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice         device         = VK_NULL_HANDLE;
    VkQueue          graphicsQueue  = VK_NULL_HANDLE;
    std::uint32_t    graphicsFamily = 0u;
    VmaAllocator     vma            = VK_NULL_HANDLE;

    ~VulkanBundle() {
        if (vma != VK_NULL_HANDLE) vmaDestroyAllocator(vma);
        if (device != VK_NULL_HANDLE) vkDestroyDevice(device, nullptr);
        if (instance != VK_NULL_HANDLE) vkDestroyInstance(instance, nullptr);
    }
};

#define VK_CHECK_RETURN(expr, rv) \
    do { const VkResult _vr = (expr); \
         if (_vr != VK_SUCCESS) { \
             std::fprintf(stderr, "[cold_cache_stress_test] %s failed: %d\n", \
                          #expr, static_cast<int>(_vr)); \
             return rv; \
         } } while (0)

bool initVolk() {
    static bool ok = false;
    if (ok) return true;
    const VkResult r = volkInitialize();
    if (r != VK_SUCCESS) {
        std::fprintf(stderr, "[cold_cache_stress_test] volkInitialize: %d\n",
                     static_cast<int>(r));
        return false;
    }
    ok = true;
    return true;
}

VkPhysicalDevice pickPhysicalDevice(VkInstance instance,
                                    std::uint32_t& outGraphicsFamily) {
    std::uint32_t count = 0u;
    if (vkEnumeratePhysicalDevices(instance, &count, nullptr) != VK_SUCCESS) return VK_NULL_HANDLE;
    if (count == 0u) return VK_NULL_HANDLE;
    std::vector<VkPhysicalDevice> devices(count);
    if (vkEnumeratePhysicalDevices(instance, &count, devices.data()) != VK_SUCCESS) return VK_NULL_HANDLE;

    VkPhysicalDevice fallback = VK_NULL_HANDLE;
    std::uint32_t    fallbackFamily = 0u;
    for (VkPhysicalDevice pd : devices) {
        std::uint32_t qfCount = 0u;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfCount, nullptr);
        std::vector<VkQueueFamilyProperties> qf(qfCount);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfCount, qf.data());
        for (std::uint32_t i = 0u; i < qfCount; ++i) {
            if ((qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0u) continue;
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(pd, &props);
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                outGraphicsFamily = i;
                return pd;
            }
            if (fallback == VK_NULL_HANDLE) {
                fallback = pd;
                fallbackFamily = i;
            }
            break;
        }
    }
    outGraphicsFamily = fallbackFamily;
    return fallback;
}

bool createVulkanBundle(VulkanBundle& out) {
    if (!initVolk()) return false;

    VkApplicationInfo app{};
    app.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName   = "enigma-cold-cache-stress-test";
    app.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    app.pEngineName        = "Enigma";
    app.engineVersion      = VK_MAKE_API_VERSION(0, 0, 1, 0);
    app.apiVersion         = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ici{};
    ici.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    VK_CHECK_RETURN(vkCreateInstance(&ici, nullptr, &out.instance), false);
    volkLoadInstance(out.instance);

    out.physicalDevice = pickPhysicalDevice(out.instance, out.graphicsFamily);
    if (out.physicalDevice == VK_NULL_HANDLE) {
        std::fprintf(stderr, "[cold_cache_stress_test] no Vulkan device with graphics queue\n");
        return false;
    }

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = out.graphicsFamily;
    qci.queueCount       = 1u;
    qci.pQueuePriorities = &priority;

    // Enable BDA + timelineSemaphore via Vulkan 1.2 features (M2.4a requirement).
    VkPhysicalDeviceVulkan12Features v12{};
    v12.sType               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    v12.bufferDeviceAddress = VK_TRUE;
    v12.timelineSemaphore   = VK_TRUE;

    VkDeviceCreateInfo dci{};
    dci.sType                = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1u;
    dci.pQueueCreateInfos    = &qci;
    dci.pNext                = &v12;
    VK_CHECK_RETURN(vkCreateDevice(out.physicalDevice, &dci, nullptr, &out.device), false);
    volkLoadDevice(out.device);
    vkGetDeviceQueue(out.device, out.graphicsFamily, 0u, &out.graphicsQueue);

    VmaVulkanFunctions fns{};
    fns.vkGetInstanceProcAddr                    = vkGetInstanceProcAddr;
    fns.vkGetDeviceProcAddr                      = vkGetDeviceProcAddr;
    fns.vkGetPhysicalDeviceProperties            = vkGetPhysicalDeviceProperties;
    fns.vkGetPhysicalDeviceMemoryProperties      = vkGetPhysicalDeviceMemoryProperties;
    fns.vkAllocateMemory                         = vkAllocateMemory;
    fns.vkFreeMemory                             = vkFreeMemory;
    fns.vkMapMemory                              = vkMapMemory;
    fns.vkUnmapMemory                            = vkUnmapMemory;
    fns.vkFlushMappedMemoryRanges                = vkFlushMappedMemoryRanges;
    fns.vkInvalidateMappedMemoryRanges           = vkInvalidateMappedMemoryRanges;
    fns.vkBindBufferMemory                       = vkBindBufferMemory;
    fns.vkBindImageMemory                        = vkBindImageMemory;
    fns.vkGetBufferMemoryRequirements            = vkGetBufferMemoryRequirements;
    fns.vkGetImageMemoryRequirements             = vkGetImageMemoryRequirements;
    fns.vkCreateBuffer                           = vkCreateBuffer;
    fns.vkDestroyBuffer                          = vkDestroyBuffer;
    fns.vkCreateImage                            = vkCreateImage;
    fns.vkDestroyImage                           = vkDestroyImage;
    fns.vkCmdCopyBuffer                          = vkCmdCopyBuffer;
    fns.vkGetBufferMemoryRequirements2KHR        = vkGetBufferMemoryRequirements2;
    fns.vkGetImageMemoryRequirements2KHR         = vkGetImageMemoryRequirements2;
    fns.vkBindBufferMemory2KHR                   = vkBindBufferMemory2;
    fns.vkBindImageMemory2KHR                    = vkBindImageMemory2;
    fns.vkGetPhysicalDeviceMemoryProperties2KHR  = vkGetPhysicalDeviceMemoryProperties2;
    fns.vkGetDeviceBufferMemoryRequirements      = vkGetDeviceBufferMemoryRequirements;
    fns.vkGetDeviceImageMemoryRequirements       = vkGetDeviceImageMemoryRequirements;

    VmaAllocatorCreateInfo aci{};
    aci.instance         = out.instance;
    aci.physicalDevice   = out.physicalDevice;
    aci.device           = out.device;
    aci.pVulkanFunctions = &fns;
    aci.vulkanApiVersion = VK_API_VERSION_1_3;
    aci.flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    VK_CHECK_RETURN(vmaCreateAllocator(&aci, &out.vma), false);
    return true;
}

// Phase-4 Security MEDIUM fix: previously this test used a
// `GfxAllocatorLayoutProbe` reinterpret_cast to smuggle a raw VmaAllocator
// into a gfx::Allocator&. That was a hard hazard if gfx::Allocator ever
// grew a member. Replaced with the public `Allocator::adopt()` factory —
// see src/gfx/AllocatorAdopt.cpp.

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------
fs::path locateDamagedHelmet(const char* argv0) {
    std::error_code ec;
    fs::path start = argv0 ? fs::absolute(argv0, ec).parent_path()
                           : fs::current_path(ec);
    if (ec) start = fs::current_path(ec);
    for (int i = 0; i < 6 && !start.empty(); ++i) {
        fs::path candidate = start / "assets" / "DamagedHelmet.glb";
        if (fs::exists(candidate, ec)) return candidate;
        if (start == start.parent_path()) break;
        start = start.parent_path();
    }
    fs::path cwd = fs::current_path(ec) / "assets" / "DamagedHelmet.glb";
    if (fs::exists(cwd, ec)) return cwd;
    return {};
}

fs::path tmpPath(const std::string& tag) {
    std::error_code ec;
    const fs::path base = fs::temp_directory_path(ec);
    if (ec || base.empty()) return fs::path{"."} / ("cold_cache_stress_" + tag + ".mpa");
    return base / ("cold_cache_stress_" + tag + ".mpa");
}

bool bakeDamagedHelmet(const fs::path& asset, const fs::path& out) {
    GltfIngest ingest;
    auto ingestRes = ingest.load(asset);
    if (!ingestRes.has_value()) return false;
    ClusterBuilder cb;
    auto clusterRes = cb.build(*ingestRes, ClusterBuildOptions{});
    if (!clusterRes.has_value()) return false;
    DagBuilder db;
    auto dagRes = db.build(std::span<const enigma::mpbake::ClusterData>(*clusterRes),
                           DagBuildOptions{});
    if (!dagRes.has_value()) return false;
    PageWriter writer;
    auto wrRes = writer.write(*dagRes, out, PageWriteOptions{});
    return wrRes.has_value();
}

// Staging helper identical to micropoly_streaming_test: write a
// RequestQueueHeader + pageIds into the queue's VkBuffer via a transient
// cmd buffer. This simulates the GPU compute emitter that M3 will provide.
struct Staging {
    VkBuffer       buffer = VK_NULL_HANDLE;
    VmaAllocation  alloc  = VK_NULL_HANDLE;
    void*          ptr    = nullptr;
    VmaAllocator   vma    = VK_NULL_HANDLE;
    ~Staging() {
        if (buffer != VK_NULL_HANDLE) vmaDestroyBuffer(vma, buffer, alloc);
    }
};

bool makeStaging(VmaAllocator vma, VkDeviceSize bytes, Staging& out) {
    VkBufferCreateInfo bci{};
    bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size        = bytes;
    bci.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    aci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT
              | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    VmaAllocationInfo info{};
    if (vmaCreateBuffer(vma, &bci, &aci, &out.buffer, &out.alloc, &info) != VK_SUCCESS) {
        return false;
    }
    out.ptr = info.pMappedData;
    out.vma = vma;
    return out.ptr != nullptr;
}

bool copyToQueue(VulkanBundle& b, VkBuffer dst, VkBuffer src, VkDeviceSize bytes) {
    VkCommandPoolCreateInfo pci{};
    pci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.queueFamilyIndex = b.graphicsFamily;
    pci.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    VkCommandPool pool = VK_NULL_HANDLE;
    if (vkCreateCommandPool(b.device, &pci, nullptr, &pool) != VK_SUCCESS) return false;

    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = pool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1u;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(b.device, &ai, &cb) != VK_SUCCESS) {
        vkDestroyCommandPool(b.device, pool, nullptr);
        return false;
    }

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);

    VkBufferCopy region{};
    region.size = bytes;
    vkCmdCopyBuffer(cb, src, dst, 1u, &region);

    vkEndCommandBuffer(cb);

    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    vkCreateFence(b.device, &fci, nullptr, &fence);

    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1u;
    si.pCommandBuffers    = &cb;
    vkQueueSubmit(b.graphicsQueue, 1u, &si, fence);

    const VkResult wr = vkWaitForFences(b.device, 1u, &fence, VK_TRUE,
                                        10ull * 1000ull * 1000ull * 1000ull);
    vkDestroyFence(b.device, fence, nullptr);
    vkDestroyCommandPool(b.device, pool, nullptr);
    return wr == VK_SUCCESS;
}

bool writeQueue(VulkanBundle& b,
                VkBuffer queueBuffer,
                const RequestQueueHeader& hdr,
                std::span<const u32> slots) {
    const VkDeviceSize bytes = sizeof(RequestQueueHeader) + slots.size() * sizeof(u32);
    Staging st;
    if (!makeStaging(b.vma, bytes, st)) return false;
    std::memcpy(st.ptr, &hdr, sizeof(hdr));
    if (!slots.empty()) {
        std::memcpy(static_cast<u8*>(st.ptr) + sizeof(hdr),
                    slots.data(), slots.size() * sizeof(u32));
    }
    return copyToQueue(b, queueBuffer, st.buffer, bytes);
}

// ---------------------------------------------------------------------------
// The cold-cache scenario: build a fresh MicropolyStreaming, request ALL
// pageIds in one burst, pump beginFrame() until they all land in residency.
// Returns the number of pageIds that completed + the final residency count.
// ---------------------------------------------------------------------------
struct CycleResult {
    u32 totalPages          = 0u;
    u32 completionsObserved = 0u;
    u32 residentCount       = 0u;
    u32 usedSlots           = 0u;
    u64 finalUploadCounter  = 0ull;
    u32 distinctSlotIds     = 0u;
};

bool runColdCacheCycle(VulkanBundle& bundle,
                       enigma::gfx::Device& device,
                       enigma::gfx::Allocator& allocator,
                       MpAssetReader& reader,
                       const fs::path& mpaPath,
                       CycleResult& out) {
    MicropolyStreamingOptions opts{};
    opts.mpaFilePath                 = mpaPath;
    opts.reader                      = &reader;
    opts.residency.capacityBytes     = 64ull * 1024ull * 1024ull;
    opts.pageCache.poolBytes         = 64ull * 1024ull * 1024ull;  // fits >>40 slots
    opts.pageCache.slotBytes         = 128u * 1024u;
    opts.requestQueue.capacity       = 1024u;
    opts.asyncIO.maxInflightRequests = 128u;
    opts.debugName                   = "cold_cache_stress_test";

    auto made = MicropolyStreaming::create(device, allocator, std::move(opts));
    if (!made) {
        std::fprintf(stderr, "[cold_cache_stress_test] create failed: %s / %s\n",
                     enigma::renderer::micropoly::micropolyStreamingErrorKindString(made.error().kind),
                     made.error().detail.c_str());
        return false;
    }
    auto& streaming = **made;

    const auto pageTable = reader.pageTable();
    const u32 total = static_cast<u32>(pageTable.size());
    out.totalPages = total;
    if (total == 0u) {
        std::fprintf(stderr, "[cold_cache_stress_test] .mpa has no pages\n");
        return false;
    }

    // Emit ALL pageIds into the request queue at once.
    RequestQueueHeader hdr{};
    hdr.count    = total;
    hdr.capacity = 1024u;
    std::vector<u32> ids;
    ids.reserve(total);
    for (u32 i = 0; i < total; ++i) ids.push_back(i);
    if (!writeQueue(bundle, streaming.requestQueue().buffer(), hdr,
                    std::span<const u32>(ids))) {
        std::fprintf(stderr, "[cold_cache_stress_test] writeQueue failed\n");
        return false;
    }

    // First beginFrame() drains + dispatches all requests.
    auto s0 = streaming.beginFrame();
    if (s0.drained != total || s0.dedupedFresh != total) {
        std::fprintf(stderr,
            "[cold_cache_stress_test] FAIL: first beginFrame drained=%u dedupedFresh=%u want %u\n",
            s0.drained, s0.dedupedFresh, total);
        return false;
    }
    if (s0.enqueueFailures != 0u || s0.lookupFailures != 0u) {
        std::fprintf(stderr,
            "[cold_cache_stress_test] FAIL: s0 enqueueFailures=%u lookupFailures=%u\n",
            s0.enqueueFailures, s0.lookupFailures);
        return false;
    }

    // Pump beginFrame() until every pageId is resident.
    u32 completed = 0u;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (completed < total &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
        auto s = streaming.beginFrame();
        completed += s.completed;
        if (s.uploadsFailed != 0u) {
            std::fprintf(stderr, "[cold_cache_stress_test] FAIL: uploadsFailed=%u\n",
                         s.uploadsFailed);
            return false;
        }
        if (s.slotAllocFailures != 0u) {
            std::fprintf(stderr, "[cold_cache_stress_test] FAIL: slotAllocFailures=%u\n",
                         s.slotAllocFailures);
            return false;
        }
    }
    out.completionsObserved = completed;
    if (completed < total) {
        std::fprintf(stderr, "[cold_cache_stress_test] FAIL: only %u/%u completions in 30s\n",
                     completed, total);
        return false;
    }

    // Verify residency.
    const auto rstats = streaming.residency().stats();
    out.residentCount = rstats.residentPageCount;
    if (rstats.residentPageCount < total) {
        std::fprintf(stderr,
            "[cold_cache_stress_test] FAIL: residentPageCount=%u want >=%u\n",
            rstats.residentPageCount, total);
        return false;
    }

    // Verify PageCache slot usage + per-page slot mapping. Collect into a
    // bitmap so duplicate slot assignments would surface.
    std::vector<u8> slotHit(streaming.pageCache().totalSlots(), 0u);
    u32 distinct = 0u;
    for (u32 i = 0; i < total; ++i) {
        const u32 slot = streaming.slotForPage(i);
        if (slot == UINT32_MAX) {
            std::fprintf(stderr, "[cold_cache_stress_test] FAIL: slotForPage(%u) missing\n", i);
            return false;
        }
        if (slot >= slotHit.size()) {
            std::fprintf(stderr, "[cold_cache_stress_test] FAIL: slot %u out of range\n", slot);
            return false;
        }
        if (slotHit[slot] != 0u) {
            std::fprintf(stderr,
                "[cold_cache_stress_test] FAIL: slot %u reused by pageId %u (duplicate assignment)\n",
                slot, i);
            return false;
        }
        slotHit[slot] = 1u;
        ++distinct;
    }
    out.distinctSlotIds = distinct;

    const auto pcStats = streaming.pageCache().stats();
    out.usedSlots = pcStats.usedSlots;
    if (pcStats.usedSlots < total) {
        std::fprintf(stderr,
            "[cold_cache_stress_test] FAIL: pageCache.usedSlots=%u want >=%u\n",
            pcStats.usedSlots, total);
        return false;
    }

    // Verify the timeline semaphore advanced at least once — means the
    // transfer queue actually executed a submit.
    out.finalUploadCounter = streaming.uploadCounter();
    if (streaming.uploadCounter() == 0ull) {
        std::fprintf(stderr,
            "[cold_cache_stress_test] FAIL: uploadCounter=0 (expected >=1 after %u completions)\n",
            completed);
        return false;
    }

    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const fs::path asset = locateDamagedHelmet(argc > 0 ? argv[0] : nullptr);
    if (asset.empty()) {
        std::fprintf(stderr, "[cold_cache_stress_test] could not locate assets/DamagedHelmet.glb\n");
        return EXIT_FAILURE;
    }

    VulkanBundle bundle;
    if (!createVulkanBundle(bundle)) {
        std::fprintf(stderr, "[cold_cache_stress_test] failed to init Vulkan bundle\n");
        return EXIT_FAILURE;
    }

    auto allocatorOwned = enigma::gfx::Allocator::adopt(bundle.vma);
    if (!allocatorOwned) {
        std::fprintf(stderr, "[cold_cache_stress_test] Allocator::adopt failed\n");
        return EXIT_FAILURE;
    }
    enigma::gfx::Allocator& allocator = *allocatorOwned;

    enigma::gfx::Device::AdoptDesc ad{};
    ad.physical            = bundle.physicalDevice;
    ad.device              = bundle.device;
    ad.graphicsQueue       = bundle.graphicsQueue;
    ad.graphicsQueueFamily = bundle.graphicsFamily;
    ad.hasTransferQueue    = false;  // unified queue — streaming falls back to graphics
    auto device = enigma::gfx::Device::adopt(ad);
    if (!device) {
        std::fprintf(stderr, "[cold_cache_stress_test] Device::adopt failed\n");
        return EXIT_FAILURE;
    }

    // Bake a fresh .mpa once; reused across all cycles.
    const fs::path mpaPath = tmpPath("bake");
    std::error_code ec;
    fs::remove(mpaPath, ec);
    if (!bakeDamagedHelmet(asset, mpaPath)) {
        std::fprintf(stderr, "[cold_cache_stress_test] bake failed\n");
        return EXIT_FAILURE;
    }

    MpAssetReader reader;
    auto openRes = reader.open(mpaPath);
    if (!openRes.has_value()) {
        std::fprintf(stderr, "[cold_cache_stress_test] reader.open failed\n");
        fs::remove(mpaPath, ec);
        return EXIT_FAILURE;
    }

    constexpr int kCycles = 3;
    bool ok = true;
    CycleResult firstResult{};
    for (int cycle = 0; cycle < kCycles; ++cycle) {
        CycleResult cr{};
        if (!runColdCacheCycle(bundle, *device, allocator, reader, mpaPath, cr)) {
            std::fprintf(stderr, "[cold_cache_stress_test] cycle %d FAIL\n", cycle);
            ok = false;
            break;
        }
        std::printf(
            "[cold_cache_stress_test] cycle %d: totalPages=%u completions=%u resident=%u "
            "usedSlots=%u distinctSlots=%u uploadCounter=%llu\n",
            cycle, cr.totalPages, cr.completionsObserved, cr.residentCount,
            cr.usedSlots, cr.distinctSlotIds,
            static_cast<unsigned long long>(cr.finalUploadCounter));

        // Idempotence: each cycle should produce the same totals (the
        // instance is fresh each time).
        if (cycle == 0) {
            firstResult = cr;
        } else {
            if (cr.totalPages != firstResult.totalPages ||
                cr.completionsObserved != firstResult.completionsObserved ||
                cr.residentCount != firstResult.residentCount ||
                cr.usedSlots != firstResult.usedSlots) {
                std::fprintf(stderr,
                    "[cold_cache_stress_test] FAIL: cycle %d disagrees with cycle 0 "
                    "(totalPages=%u/%u completions=%u/%u resident=%u/%u usedSlots=%u/%u)\n",
                    cycle,
                    cr.totalPages,          firstResult.totalPages,
                    cr.completionsObserved, firstResult.completionsObserved,
                    cr.residentCount,       firstResult.residentCount,
                    cr.usedSlots,           firstResult.usedSlots);
                ok = false;
                break;
            }
        }
    }

    // Make sure the transfer queue has retired before we tear down the
    // bundle (vkDestroyDevice would otherwise blow up on in-flight cmd bufs).
    vkDeviceWaitIdle(bundle.device);

    reader.close();
    fs::remove(mpaPath, ec);

    if (!ok) {
        std::fprintf(stderr, "[cold_cache_stress_test] FAIL\n");
        return EXIT_FAILURE;
    }
    std::printf("[cold_cache_stress_test] ALL %d CYCLES PASS (totalPages=%u per cycle)\n",
                kCycles, firstResult.totalPages);
    return EXIT_SUCCESS;
}
