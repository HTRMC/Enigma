// Unit test for MicropolyStreaming (M2.3).
// Five cases:
//   1) Creation with real .mpa file + valid options -> unique_ptr non-null,
//      sub-components reachable via accessors.
//   2) beginFrame with empty request queue -> all stats zero.
//   3) Manually poke request queue with 3 real pageIds -> beginFrame sees
//      drained=3, dedupedFresh=3, queuedForIO=3.
//   4) Wait for async IO completions; call beginFrame again. Verify
//      FrameStats.completed matches and residency manager has entries.
//   5) Dedup: same pageId twice in the drained queue -> dedupedFresh=1.
//
// M2.3 stub behavior: uploadsScheduled is always 0 (transfer-queue wiring
// is TODO(M2.4)). slotForPage() is populated for successfully-completed
// pages since we DO allocate PageCache slots.
//
// Shares the mpbake pipeline with async_io_worker_test to produce a real
// .mpa from DamagedHelmet.glb, then exercises the streaming orchestrator
// against it. Requires a headless Vulkan device (see page_cache_test for
// the bringup pattern).

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

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <system_error>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

using enigma::u32;
using enigma::u8;
using enigma::asset::MpAssetReader;
using enigma::asset::MpPageEntry;
using enigma::mpbake::ClusterBuildOptions;
using enigma::mpbake::ClusterBuilder;
using enigma::mpbake::DagBuildOptions;
using enigma::mpbake::DagBuilder;
using enigma::mpbake::GltfIngest;
using enigma::mpbake::PageWriter;
using enigma::mpbake::PageWriteOptions;
using enigma::renderer::micropoly::MicropolyStreaming;
using enigma::renderer::micropoly::MicropolyStreamingOptions;
using enigma::renderer::micropoly::PageCacheOptions;
using enigma::renderer::micropoly::RequestQueueHeader;
using enigma::renderer::micropoly::RequestQueueOptions;
using enigma::renderer::micropoly::ResidencyManagerOptions;

namespace {

// ---------------------------------------------------------------------------
// Headless Vulkan bundle (mirrored from page_cache_test.cpp).
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
             std::fprintf(stderr, "[micropoly_streaming_test] %s failed: %d\n", \
                          #expr, static_cast<int>(_vr)); \
             return rv; \
         } } while (0)

bool initVolk() {
    static bool ok = false;
    if (ok) return true;
    const VkResult r = volkInitialize();
    if (r != VK_SUCCESS) {
        std::fprintf(stderr, "[micropoly_streaming_test] volkInitialize: %d\n",
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
    app.pApplicationName   = "enigma-micropoly-streaming-test";
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
        std::fprintf(stderr, "[micropoly_streaming_test] no Vulkan device with graphics queue\n");
        return false;
    }

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = out.graphicsFamily;
    qci.queueCount       = 1u;
    qci.pQueuePriorities = &priority;

    // M2.4a: MicropolyStreaming creates a timeline semaphore. Enable both
    // BDA and timeline-semaphore features via VkPhysicalDeviceVulkan12Features.
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

// ---------------------------------------------------------------------------
// Adopt-based harness
// ---------------------------------------------------------------------------
// M2.4 replaced the old byte-aligned Device shim with Device::adopt(), and
// Phase-4 replaces the `GfxAllocatorLayoutProbe` Allocator shim with
// Allocator::adopt() — both are public test-only factories that wrap the
// already-created Vulkan + VMA handles in the real gfx types. This removes
// the undefined-behavior that the old layout probes would have exhibited
// as soon as MicropolyStreaming's transfer-queue path dereferenced
// device_->logical() / transferQueue() on zeroed bytes, or as soon as
// gfx::Allocator gained a second member.

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
    if (ec || base.empty()) return fs::path{"."} / ("micropoly_streaming_test_" + tag + ".mpa");
    return base / ("micropoly_streaming_test_" + tag + ".mpa");
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

// Helper: copy bytes into the RequestQueue's VkBuffer via a transient
// staging copy — same pattern as request_queue_test.cpp.
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

struct TestEnv {
    VulkanBundle                            bundle;
    std::unique_ptr<enigma::gfx::Allocator> allocator;  // adopted from bundle.vma
    std::unique_ptr<enigma::gfx::Device>    device;     // adopted from bundle
    fs::path                                mpaPath;
    MpAssetReader                           reader;

    ~TestEnv() {
        // Release the adopted Device BEFORE the VulkanBundle destructor runs
        // — adopt() owns logical device destruction.
        device.reset();
        allocator.reset();
        reader.close();
        if (!mpaPath.empty()) {
            std::error_code ec;
            fs::remove(mpaPath, ec);
        }
    }

    enigma::gfx::Allocator& allocatorRef() {
        return *allocator;
    }

    // Real gfx::Device constructed via Device::adopt(). MicropolyStreaming
    // reads device_->logical() / transferQueue*() in M2.4 — the old byte-
    // aligned shim would have been UB.
    enigma::gfx::Device& deviceRef() {
        return *device;
    }
};

bool makeEnv(TestEnv& env, const fs::path& asset) {
    if (!createVulkanBundle(env.bundle)) return false;
    env.allocator = enigma::gfx::Allocator::adopt(env.bundle.vma);
    if (!env.allocator) {
        std::fprintf(stderr, "[micropoly_streaming_test] Allocator::adopt failed\n");
        return false;
    }

    // M2.4: replace the old zeroed-bytes Device shim with Device::adopt().
    // MicropolyStreaming now dereferences device_->logical() and
    // transferQueue() during transfer-queue setup; the shim would be UB.
    enigma::gfx::Device::AdoptDesc ad{};
    ad.physical            = env.bundle.physicalDevice;
    ad.device              = env.bundle.device;
    ad.graphicsQueue       = env.bundle.graphicsQueue;
    ad.graphicsQueueFamily = env.bundle.graphicsFamily;
    // No dedicated transfer queue in the test bundle — streaming falls back
    // to the graphics queue, which matches production on unified GPUs.
    ad.hasTransferQueue    = false;
    env.device = enigma::gfx::Device::adopt(ad);
    if (!env.device) {
        std::fprintf(stderr, "[micropoly_streaming_test] Device::adopt failed\n");
        return false;
    }

    env.mpaPath = tmpPath("bake");
    std::error_code ec;
    fs::remove(env.mpaPath, ec);
    if (!bakeDamagedHelmet(asset, env.mpaPath)) {
        std::fprintf(stderr, "[micropoly_streaming_test] bakeDamagedHelmet failed\n");
        return false;
    }

    auto r = env.reader.open(env.mpaPath);
    if (!r.has_value()) {
        std::fprintf(stderr, "[micropoly_streaming_test] reader.open failed\n");
        return false;
    }
    if (env.reader.pageTable().empty()) {
        std::fprintf(stderr, "[micropoly_streaming_test] baked file has no pages\n");
        return false;
    }
    return true;
}

MicropolyStreamingOptions makeStreamingOpts(TestEnv& env) {
    MicropolyStreamingOptions opts{};
    opts.mpaFilePath = env.mpaPath;
    opts.reader      = &env.reader;

    opts.residency.capacityBytes = 64ull * 1024ull * 1024ull;  // plenty for tests

    opts.pageCache.poolBytes = 64ull * 1024ull * 1024ull;  // 64 MiB
    opts.pageCache.slotBytes = 128u * 1024u;               // 128 KiB per plan

    opts.requestQueue.capacity = 1024u;

    opts.asyncIO.maxInflightRequests = 32u;

    opts.debugName = "micropoly_streaming_test";
    return opts;
}

// ---------------------------------------------------------------------------
// Case 1: creation
// ---------------------------------------------------------------------------
bool testCreate(TestEnv& env) {
    auto opts = makeStreamingOpts(env);
    auto made = MicropolyStreaming::create(env.deviceRef(), env.allocatorRef(), opts);
    if (!made) {
        std::fprintf(stderr, "[micropoly_streaming_test] case 1 FAIL: create: %s / %s\n",
                     enigma::renderer::micropoly::micropolyStreamingErrorKindString(made.error().kind),
                     made.error().detail.c_str());
        return false;
    }
    if (!*made) {
        std::fprintf(stderr, "[micropoly_streaming_test] case 1 FAIL: null unique_ptr\n");
        return false;
    }
    if ((*made)->requestQueue().capacity() != 1024u) {
        std::fprintf(stderr, "[micropoly_streaming_test] case 1 FAIL: requestQueue.capacity()=%u\n",
                     (*made)->requestQueue().capacity());
        return false;
    }
    if ((*made)->pageCache().slotBytes() != 128u * 1024u) {
        std::fprintf(stderr, "[micropoly_streaming_test] case 1 FAIL: pageCache.slotBytes()=%u\n",
                     (*made)->pageCache().slotBytes());
        return false;
    }
    std::printf("[micropoly_streaming_test] case 1 PASS: create + accessors reachable.\n");
    return true;
}

// ---------------------------------------------------------------------------
// Case 2: empty beginFrame
// ---------------------------------------------------------------------------
bool testEmptyFrame(TestEnv& env) {
    auto opts = makeStreamingOpts(env);
    auto made = MicropolyStreaming::create(env.deviceRef(), env.allocatorRef(), opts);
    if (!made) { std::fprintf(stderr, "[micropoly_streaming_test] case 2 FAIL: create\n"); return false; }

    auto stats = (*made)->beginFrame();
    if (stats.drained != 0u || stats.dedupedFresh != 0u || stats.cacheHits != 0u ||
        stats.queuedForIO != 0u || stats.completed != 0u) {
        std::fprintf(stderr, "[micropoly_streaming_test] case 2 FAIL: stats drained=%u fresh=%u queued=%u comp=%u\n",
                     stats.drained, stats.dedupedFresh, stats.queuedForIO, stats.completed);
        return false;
    }
    std::printf("[micropoly_streaming_test] case 2 PASS: empty beginFrame -> all zero.\n");
    return true;
}

// ---------------------------------------------------------------------------
// Case 3: drained requests dispatched to async IO
// ---------------------------------------------------------------------------
bool testDispatch(TestEnv& env) {
    auto opts = makeStreamingOpts(env);
    auto made = MicropolyStreaming::create(env.deviceRef(), env.allocatorRef(), opts);
    if (!made) { std::fprintf(stderr, "[micropoly_streaming_test] case 3 FAIL: create\n"); return false; }

    const auto pageTbl = env.reader.pageTable();
    const u32 nPages = static_cast<u32>(std::min<std::size_t>(3u, pageTbl.size()));
    if (nPages == 0u) {
        std::fprintf(stderr, "[micropoly_streaming_test] case 3 FAIL: empty page table\n");
        return false;
    }

    // Poke the request queue with the first N pageIds.
    RequestQueueHeader hdr{};
    hdr.count    = nPages;
    hdr.capacity = 1024u;
    std::vector<u32> pageIds;
    pageIds.reserve(nPages);
    for (u32 i = 0; i < nPages; ++i) pageIds.push_back(i);

    if (!writeQueue(env.bundle, (*made)->requestQueue().buffer(), hdr,
                    std::span<const u32>(pageIds))) {
        std::fprintf(stderr, "[micropoly_streaming_test] case 3 FAIL: writeQueue\n");
        return false;
    }

    auto stats = (*made)->beginFrame();
    if (stats.drained != nPages) {
        std::fprintf(stderr, "[micropoly_streaming_test] case 3 FAIL: drained=%u want %u\n",
                     stats.drained, nPages);
        return false;
    }
    if (stats.dedupedFresh != nPages) {
        std::fprintf(stderr, "[micropoly_streaming_test] case 3 FAIL: dedupedFresh=%u\n",
                     stats.dedupedFresh);
        return false;
    }
    if (stats.queuedForIO != nPages) {
        std::fprintf(stderr, "[micropoly_streaming_test] case 3 FAIL: queuedForIO=%u enqueueFail=%u lookupFail=%u\n",
                     stats.queuedForIO, stats.enqueueFailures, stats.lookupFailures);
        return false;
    }
    // M2.4a: uploadsScheduled is now populated since we have a real device +
    // transfer-queue wiring. The value depends on how many completions landed
    // in the SAME beginFrame as the dispatch, which is async — completions
    // can arrive in subsequent frames. Assertion relaxed to "no failures";
    // case 4 below waits for completions + verifies the residency population.
    if (stats.uploadsFailed != 0u) {
        std::fprintf(stderr, "[micropoly_streaming_test] case 3 FAIL: uploadsFailed=%u (want 0)\n",
                     stats.uploadsFailed);
        return false;
    }
    std::printf("[micropoly_streaming_test] case 3 PASS: %u requests dispatched to async IO.\n", nPages);

    // Case 4 (completions) — wait for IO to finish + call beginFrame again.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    u32 totalCompleted = 0u;
    while (totalCompleted < nPages &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
        auto s = (*made)->beginFrame();
        totalCompleted += s.completed;
    }
    if (totalCompleted < nPages) {
        std::fprintf(stderr, "[micropoly_streaming_test] case 4 FAIL: only %u/%u completions within 10s\n",
                     totalCompleted, nPages);
        return false;
    }

    // Residency manager must now show at least nPages entries.
    const auto rstats = (*made)->residency().stats();
    if (rstats.residentPageCount < nPages) {
        std::fprintf(stderr, "[micropoly_streaming_test] case 4 FAIL: residentPageCount=%u want >=%u\n",
                     rstats.residentPageCount, nPages);
        return false;
    }

    // slotForPage must be valid for each requested pageId.
    for (u32 i = 0; i < nPages; ++i) {
        const u32 slot = (*made)->slotForPage(i);
        if (slot == UINT32_MAX) {
            std::fprintf(stderr, "[micropoly_streaming_test] case 4 FAIL: slotForPage(%u) missing\n", i);
            return false;
        }
    }
    std::printf("[micropoly_streaming_test] case 4 PASS: %u completions consumed, residency populated.\n",
                totalCompleted);
    return true;
}

// ---------------------------------------------------------------------------
// Case 5: dedup
// ---------------------------------------------------------------------------
bool testDedup(TestEnv& env) {
    auto opts = makeStreamingOpts(env);
    auto made = MicropolyStreaming::create(env.deviceRef(), env.allocatorRef(), opts);
    if (!made) { std::fprintf(stderr, "[micropoly_streaming_test] case 5 FAIL: create\n"); return false; }

    RequestQueueHeader hdr{};
    hdr.count    = 4u;
    hdr.capacity = 1024u;
    std::array<u32, 4> slots = {0u, 0u, 0u, 0u};  // same pageId four times
    if (!writeQueue(env.bundle, (*made)->requestQueue().buffer(), hdr,
                    std::span<const u32>(slots))) {
        std::fprintf(stderr, "[micropoly_streaming_test] case 5 FAIL: writeQueue\n");
        return false;
    }

    auto stats = (*made)->beginFrame();
    if (stats.drained != 4u) {
        std::fprintf(stderr, "[micropoly_streaming_test] case 5 FAIL: drained=%u\n", stats.drained);
        return false;
    }
    if (stats.dedupedFresh != 1u) {
        std::fprintf(stderr, "[micropoly_streaming_test] case 5 FAIL: dedupedFresh=%u want 1\n",
                     stats.dedupedFresh);
        return false;
    }
    if (stats.queuedForIO != 1u) {
        std::fprintf(stderr, "[micropoly_streaming_test] case 5 FAIL: queuedForIO=%u want 1\n",
                     stats.queuedForIO);
        return false;
    }
    std::printf("[micropoly_streaming_test] case 5 PASS: 4 duplicate IDs dedup to 1.\n");
    return true;
}

// ---------------------------------------------------------------------------
// Case 6: M4.5 multi-cluster page support — pageFirstDagNode attach ABI check
// ---------------------------------------------------------------------------
// Exercises the end-to-end attach path:
//   1) MpAssetReader::firstDagNodeIndices() returns one u32 per page.
//   2) MicropolyStreaming::attachPageFirstDagNodeBuffer() succeeds.
//   3) pageFirstDagNodeBuffer()/bufferBytes() are non-null/non-zero.
//   4) Entries mirror the baked page table's firstDagNodeIdx values.
// The post-attach bindless slot is set by the caller (Renderer); this test
// deliberately does NOT register a bindless slot so we only validate the
// buffer-side contract MicropolyStreaming owns.
bool testFirstDagNodeAttach(TestEnv& env) {
    auto opts = makeStreamingOpts(env);
    auto made = MicropolyStreaming::create(env.deviceRef(), env.allocatorRef(), opts);
    if (!made) {
        std::fprintf(stderr, "[micropoly_streaming_test] case 6 FAIL: create\n");
        return false;
    }

    const auto firstDagIdx = env.reader.firstDagNodeIndices();
    if (firstDagIdx.empty()) {
        std::fprintf(stderr, "[micropoly_streaming_test] case 6 FAIL: empty firstDagNodeIndices\n");
        return false;
    }
    const std::size_t pageCount = firstDagIdx.size();

    const bool ok = (*made)->attachPageFirstDagNodeBuffer(
        std::span<const u32>(firstDagIdx));
    if (!ok) {
        std::fprintf(stderr, "[micropoly_streaming_test] case 6 FAIL: attachPageFirstDagNodeBuffer\n");
        return false;
    }

    if ((*made)->pageFirstDagNodeBuffer() == VK_NULL_HANDLE) {
        std::fprintf(stderr, "[micropoly_streaming_test] case 6 FAIL: null VkBuffer post-attach\n");
        return false;
    }
    if ((*made)->pageFirstDagNodeBufferBytes() == 0u) {
        std::fprintf(stderr, "[micropoly_streaming_test] case 6 FAIL: zero bufferBytes post-attach\n");
        return false;
    }

    // Bindless slot defaults to UINT32_MAX until the caller (Renderer)
    // registers the buffer — we don't drive that side in this test.
    if ((*made)->pageFirstDagNodeBufferBindless() != UINT32_MAX) {
        std::fprintf(stderr,
                     "[micropoly_streaming_test] case 6 FAIL: unexpected bindless slot %u pre-register\n",
                     (*made)->pageFirstDagNodeBufferBindless());
        return false;
    }

    // Stamp + read-back round trip — mirrors the Renderer's
    // setPageFirstDagNodeBindless() path.
    (*made)->setPageFirstDagNodeBindless(42u);
    if ((*made)->pageFirstDagNodeBufferBindless() != 42u) {
        std::fprintf(stderr, "[micropoly_streaming_test] case 6 FAIL: bindless setter round-trip\n");
        return false;
    }
    (*made)->setPageFirstDagNodeBindless(UINT32_MAX);

    std::printf("[micropoly_streaming_test] case 6 PASS: pageFirstDagNode attach ABI (pageCount=%zu).\n",
                pageCount);
    return true;
}

// ---------------------------------------------------------------------------
// Case 7: M3.3-deferred DAG SSBO — attachDagNodeBuffer ABI check
// ---------------------------------------------------------------------------
// Exercises the runtime-format DAG attach path:
//   1) MpAssetReader::assembleRuntimeDagNodes() yields dagNodeCount × 80 B
//      (M4: widened from 3 → 4 float4 to carry maxError + parentMaxError;
//       M4-fix: widened from 4 → 5 float4 to carry parentCenter for the
//       group-coherent SSE LOD anchor that prevents cross-sibling flicker).
//   2) MicropolyStreaming::attachDagNodeBuffer() succeeds on the staging
//      upload + DEVICE_LOCAL fill.
//   3) dagNodeBuffer() / dagNodeBufferBytes() are non-null/non-zero.
//   4) Bindless slot round-trips through the setter just like the other
//      attach paths — the Renderer owns DescriptorAllocator registration.
bool testDagNodeAttach(TestEnv& env) {
    auto opts = makeStreamingOpts(env);
    auto made = MicropolyStreaming::create(env.deviceRef(), env.allocatorRef(), opts);
    if (!made) {
        std::fprintf(stderr, "[micropoly_streaming_test] case 7 FAIL: create\n");
        return false;
    }

    auto runtimeDag = env.reader.assembleRuntimeDagNodes();
    if (!runtimeDag.has_value()) {
        std::fprintf(stderr, "[micropoly_streaming_test] case 7 FAIL: assembleRuntimeDagNodes: %s / %s\n",
                     enigma::asset::mpReadErrorKindString(runtimeDag.error().kind),
                     runtimeDag.error().detail.c_str());
        return false;
    }
    if (runtimeDag->empty()) {
        std::fprintf(stderr, "[micropoly_streaming_test] case 7 FAIL: empty runtimeDag\n");
        return false;
    }

    // Each runtime node is 80 bytes = 5×float4 (M4-fix widening: m4 carries
    // parentCenter.xyz so the SSE LOD test uses a group-shared anchor —
    // eliminates cross-sibling flicker at cut boundaries). Sanity-check
    // the type size so a future header tweak doesn't silently break the
    // upload contract.
    static_assert(sizeof(MpAssetReader::RuntimeDagNode) == 80u,
                  "RuntimeDagNode must be 80 bytes (5×float4) to match shader");

    const u8*  dagBytes = reinterpret_cast<const u8*>(runtimeDag->data());
    const auto dagByteCount = runtimeDag->size() * sizeof(MpAssetReader::RuntimeDagNode);

    const bool ok = (*made)->attachDagNodeBuffer(
        std::span<const u8>(dagBytes, dagByteCount));
    if (!ok) {
        std::fprintf(stderr, "[micropoly_streaming_test] case 7 FAIL: attachDagNodeBuffer\n");
        return false;
    }

    if ((*made)->dagNodeBuffer() == VK_NULL_HANDLE) {
        std::fprintf(stderr, "[micropoly_streaming_test] case 7 FAIL: null VkBuffer post-attach\n");
        return false;
    }
    if ((*made)->dagNodeBufferBytes() == 0u) {
        std::fprintf(stderr, "[micropoly_streaming_test] case 7 FAIL: zero bufferBytes post-attach\n");
        return false;
    }

    // Padded-up to 16 B stride so a StructuredBuffer<float4> view stays legal.
    const auto expectedEntryBytes  = dagByteCount;
    const auto expectedPaddedBytes = (expectedEntryBytes + 15ull) & ~15ull;
    if ((*made)->dagNodeBufferBytes() != expectedPaddedBytes) {
        std::fprintf(stderr,
                     "[micropoly_streaming_test] case 7 FAIL: bufferBytes=%llu want %llu (padded)\n",
                     static_cast<unsigned long long>((*made)->dagNodeBufferBytes()),
                     static_cast<unsigned long long>(expectedPaddedBytes));
        return false;
    }

    // Bindless slot defaults to UINT32_MAX until the caller (Renderer)
    // registers the buffer — we don't drive that side in this test.
    if ((*made)->dagNodeBufferBindless() != UINT32_MAX) {
        std::fprintf(stderr,
                     "[micropoly_streaming_test] case 7 FAIL: unexpected bindless slot %u pre-register\n",
                     (*made)->dagNodeBufferBindless());
        return false;
    }

    // Stamp + read-back round trip — mirrors the Renderer's
    // setDagNodeBufferBindless() path.
    (*made)->setDagNodeBufferBindless(123u);
    if ((*made)->dagNodeBufferBindless() != 123u) {
        std::fprintf(stderr, "[micropoly_streaming_test] case 7 FAIL: bindless setter round-trip\n");
        return false;
    }
    (*made)->setDagNodeBufferBindless(UINT32_MAX);

    // Re-attach at the SAME size must be idempotent (no leak, same buffer bytes).
    const VkBuffer       firstBuffer = (*made)->dagNodeBuffer();
    const enigma::u64    firstBytes  = (*made)->dagNodeBufferBytes();
    const bool ok2 = (*made)->attachDagNodeBuffer(
        std::span<const u8>(dagBytes, dagByteCount));
    if (!ok2) {
        std::fprintf(stderr, "[micropoly_streaming_test] case 7 FAIL: re-attach same-size\n");
        return false;
    }
    if ((*made)->dagNodeBuffer() != firstBuffer ||
        (*made)->dagNodeBufferBytes() != firstBytes) {
        std::fprintf(stderr, "[micropoly_streaming_test] case 7 FAIL: same-size re-attach changed buffer\n");
        return false;
    }

    std::printf("[micropoly_streaming_test] case 7 PASS: DAG node attach ABI (nodes=%zu, bytes=%zu).\n",
                runtimeDag->size(), static_cast<std::size_t>(dagByteCount));
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const fs::path asset = locateDamagedHelmet(argc > 0 ? argv[0] : nullptr);
    if (asset.empty()) {
        std::fprintf(stderr, "[micropoly_streaming_test] could not locate assets/DamagedHelmet.glb\n");
        return EXIT_FAILURE;
    }

    TestEnv env;
    if (!makeEnv(env, asset)) {
        std::fprintf(stderr, "[micropoly_streaming_test] failed to init env\n");
        return EXIT_FAILURE;
    }

    bool ok = true;
    ok &= testCreate(env);
    ok &= testEmptyFrame(env);
    ok &= testDispatch(env);   // covers cases 3 + 4
    ok &= testDedup(env);
    ok &= testFirstDagNodeAttach(env);  // M4.5 multi-cluster ABI check
    ok &= testDagNodeAttach(env);       // M3.3-deferred DAG SSBO ABI check

    if (!ok) {
        std::fprintf(stderr, "[micropoly_streaming_test] FAIL\n");
        return EXIT_FAILURE;
    }
    std::printf("[micropoly_streaming_test] ALL CASES PASS\n");
    return EXIT_SUCCESS;
}
