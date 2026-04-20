// Unit test for RequestQueue (M2.3).
// Four cases:
//   1) Creation + stats           : capacity=1024 -> buffer allocated, stats zeroed.
//   2) Manual write + drain       : poke header.count=5 + 5 slots, drain() returns them,
//                                   count resets, stats update.
//   3) Drain into small span      : 5 queued, drain into 3-element span -> returns 3,
//                                   remaining 2 stay in the buffer before reset... actually
//                                   drain always resets count per the contract; so the
//                                   test just checks the small-span path returns copyCount
//                                   and stats.lastDrainCount reflects the clamped value.
//   4) Overflow detection         : header.overflowed=1 -> stats.overflowEvents++.
//
// Mirrors page_cache_test.cpp bringup pattern for a headless Vulkan
// device + VMA instance.

#include "renderer/micropoly/RequestQueue.h"

#include "core/Types.h"
#include "gfx/Allocator.h"

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
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

using enigma::u32;
using enigma::u8;
using enigma::renderer::micropoly::RequestQueue;
using enigma::renderer::micropoly::RequestQueueError;
using enigma::renderer::micropoly::RequestQueueErrorKind;
using enigma::renderer::micropoly::RequestQueueHeader;
using enigma::renderer::micropoly::RequestQueueOptions;
using enigma::renderer::micropoly::requestQueueErrorKindString;

namespace {

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
             std::fprintf(stderr, "[request_queue_test] %s failed: %d\n", \
                          #expr, static_cast<int>(_vr)); \
             return rv; \
         } } while (0)

bool initVolk() {
    static bool ok = false;
    if (ok) return true;
    const VkResult r = volkInitialize();
    if (r != VK_SUCCESS) {
        std::fprintf(stderr, "[request_queue_test] volkInitialize: %d\n", static_cast<int>(r));
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
    app.pApplicationName   = "enigma-request-queue-test";
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
        std::fprintf(stderr, "[request_queue_test] no Vulkan device with graphics queue\n");
        return false;
    }

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = out.graphicsFamily;
    qci.queueCount       = 1u;
    qci.pQueuePriorities = &priority;

    VkPhysicalDeviceBufferDeviceAddressFeatures bdaFeatures{};
    bdaFeatures.sType               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
    bdaFeatures.bufferDeviceAddress = VK_TRUE;

    VkDeviceCreateInfo dci{};
    dci.sType                = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1u;
    dci.pQueueCreateInfos    = &qci;
    dci.pNext                = &bdaFeatures;
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

// Phase-4 Security MEDIUM fix: replaced `GfxAllocatorLayoutProbe` +
// reinterpret_cast shim with `Allocator::adopt()` (public test-only factory).
struct TestEnv {
    VulkanBundle                            bundle;
    std::unique_ptr<enigma::gfx::Allocator> allocator;
    enigma::gfx::Allocator& allocatorRef() {
        return *allocator;
    }
};

bool makeEnv(TestEnv& env) {
    if (!createVulkanBundle(env.bundle)) return false;
    env.allocator = enigma::gfx::Allocator::adopt(env.bundle.vma);
    return env.allocator != nullptr;
}

// Friend-ish helper: poke the mapped buffer directly for test setup. We do
// NOT have access to RequestQueue's private mappedPtr_, but we can map the
// buffer ourselves via VMA — except the buffer is already mapped by VMA
// with HOST_ACCESS_RANDOM + MAPPED. The cleanest path for the test is to
// create the queue, then look up the mapped pointer via vmaGetAllocationInfo
// on the queue's allocation. That requires access to the allocation handle,
// which is private. So instead we emit a friendly pattern: the test
// workflow is "build the queue, then call queue.drain() after GPU-side
// writes happen". For a unit test without a GPU compute shader, we route
// the write through a small staging buffer + vkCmdCopyBuffer — but even
// simpler: we expose a plain test-only helper by re-mapping the buffer via
// vmaMapMemory on a freshly-allocated VMA allocation on our side.
//
// Simpler still: just keep a raw secondary map. We know the buffer is
// HOST_VISIBLE (that's exactly what RequestQueue requires for its own
// persistent map). So here in the test we create our OWN staging buffer,
// copy its bytes into RequestQueue via vkCmdCopyBuffer, submit, wait, and
// then call drain(). That avoids poking the internal mapped pointer.

// Helper: allocate a host-visible staging buffer; return mapped ptr.
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

// Copy `bytes` from staging into the queue's VkBuffer via a transient
// command buffer + fence. Returns true on success.
bool copyStagingToQueue(VulkanBundle& b, VkBuffer dst, VkBuffer src, VkDeviceSize bytes) {
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

    const VkResult wr = vkWaitForFences(b.device, 1u, &fence, VK_TRUE, 10ull * 1000ull * 1000ull * 1000ull);
    vkDestroyFence(b.device, fence, nullptr);
    vkDestroyCommandPool(b.device, pool, nullptr);
    return wr == VK_SUCCESS;
}

// Write a RequestQueueHeader + slot payload into the queue's VkBuffer.
// Used to simulate GPU-side InterlockedAdd without an actual compute pass.
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

    return copyStagingToQueue(b, queueBuffer, st.buffer, bytes);
}

// ---------------------------------------------------------------------------
// Case 1: creation + stats
// ---------------------------------------------------------------------------
bool testCreateStats(TestEnv& env) {
    RequestQueueOptions opts{};
    opts.capacity = 1024u;
    opts.debugName = "request_queue_test.case1";

    auto made = RequestQueue::create(env.allocatorRef(), opts);
    if (!made) {
        std::fprintf(stderr, "[request_queue_test] case 1 FAIL: create %s / %s\n",
                     requestQueueErrorKindString(made.error().kind),
                     made.error().detail.c_str());
        return false;
    }
    if (made->capacity() != 1024u) {
        std::fprintf(stderr, "[request_queue_test] case 1 FAIL: capacity=%u\n", made->capacity());
        return false;
    }
    if (made->buffer() == VK_NULL_HANDLE) {
        std::fprintf(stderr, "[request_queue_test] case 1 FAIL: null VkBuffer\n");
        return false;
    }
    const auto s = made->stats();
    if (s.totalDrained != 0u || s.overflowEvents != 0u || s.lastDrainCount != 0u) {
        std::fprintf(stderr, "[request_queue_test] case 1 FAIL: stats not zero\n");
        return false;
    }
    std::printf("[request_queue_test] case 1 PASS: creation + zero stats (capacity=1024).\n");
    return true;
}

// ---------------------------------------------------------------------------
// Case 2: manual header write + drain
// ---------------------------------------------------------------------------
bool testManualDrain(TestEnv& env) {
    RequestQueueOptions opts{};
    opts.capacity = 1024u;
    auto made = RequestQueue::create(env.allocatorRef(), opts);
    if (!made) {
        std::fprintf(stderr, "[request_queue_test] case 2 FAIL: create\n");
        return false;
    }

    RequestQueueHeader hdr{};
    hdr.count      = 5u;
    hdr.capacity   = 1024u;
    hdr.overflowed = 0u;
    hdr._pad       = 0u;

    std::array<u32, 5> slots = {101u, 202u, 303u, 404u, 505u};
    if (!writeQueue(env.bundle, made->buffer(), hdr, std::span<const u32>(slots))) {
        std::fprintf(stderr, "[request_queue_test] case 2 FAIL: writeQueue\n");
        return false;
    }

    std::array<u32, 8> out{};
    const u32 n = made->drain(std::span<u32>(out));
    if (n != 5u) {
        std::fprintf(stderr, "[request_queue_test] case 2 FAIL: drain returned %u\n", n);
        return false;
    }
    for (u32 i = 0; i < 5u; ++i) {
        if (out[i] != slots[i]) {
            std::fprintf(stderr, "[request_queue_test] case 2 FAIL: slot %u got %u want %u\n",
                         i, out[i], slots[i]);
            return false;
        }
    }

    const auto s = made->stats();
    if (s.totalDrained != 5u || s.lastDrainCount != 5u || s.overflowEvents != 0u) {
        std::fprintf(stderr, "[request_queue_test] case 2 FAIL: stats drained=%llu last=%u overflow=%llu\n",
                     static_cast<unsigned long long>(s.totalDrained),
                     s.lastDrainCount,
                     static_cast<unsigned long long>(s.overflowEvents));
        return false;
    }

    // Second drain should return 0 — the buffer was reset by the first drain.
    std::array<u32, 8> out2{};
    const u32 n2 = made->drain(std::span<u32>(out2));
    if (n2 != 0u) {
        std::fprintf(stderr, "[request_queue_test] case 2 FAIL: second drain returned %u (want 0)\n", n2);
        return false;
    }
    std::printf("[request_queue_test] case 2 PASS: drained 5 slots, count reset, stats updated.\n");
    return true;
}

// ---------------------------------------------------------------------------
// Case 3: drain with too-small output span
// ---------------------------------------------------------------------------
bool testSmallSpan(TestEnv& env) {
    RequestQueueOptions opts{};
    opts.capacity = 1024u;
    auto made = RequestQueue::create(env.allocatorRef(), opts);
    if (!made) {
        std::fprintf(stderr, "[request_queue_test] case 3 FAIL: create\n");
        return false;
    }

    RequestQueueHeader hdr{};
    hdr.count    = 5u;
    hdr.capacity = 1024u;
    std::array<u32, 5> slots = {1u, 2u, 3u, 4u, 5u};
    if (!writeQueue(env.bundle, made->buffer(), hdr, std::span<const u32>(slots))) {
        std::fprintf(stderr, "[request_queue_test] case 3 FAIL: writeQueue\n");
        return false;
    }

    std::array<u32, 3> out{};
    // Use drainEx() to verify the M2.3 Security MEDIUM-1 closeout: dropped
    // entries are surfaced via DrainResult.dropped, no longer silently lost.
    const auto dr = made->drainEx(std::span<u32>(out));
    if (dr.copied != 3u) {
        std::fprintf(stderr, "[request_queue_test] case 3 FAIL: drainEx.copied=%u (want 3)\n", dr.copied);
        return false;
    }
    if (dr.dropped != 2u) {
        std::fprintf(stderr, "[request_queue_test] case 3 FAIL: drainEx.dropped=%u (want 2)\n", dr.dropped);
        return false;
    }
    if (out[0] != 1u || out[1] != 2u || out[2] != 3u) {
        std::fprintf(stderr, "[request_queue_test] case 3 FAIL: contents [%u,%u,%u]\n",
                     out[0], out[1], out[2]);
        return false;
    }
    const auto s = made->stats();
    if (s.lastDrainCount != 3u || s.totalDrained != 3u) {
        std::fprintf(stderr, "[request_queue_test] case 3 FAIL: stats last=%u total=%llu\n",
                     s.lastDrainCount, static_cast<unsigned long long>(s.totalDrained));
        return false;
    }
    std::printf("[request_queue_test] case 3 PASS: drainEx returns copied=3 dropped=2, stats reflect clamp.\n");
    return true;
}

// ---------------------------------------------------------------------------
// Case 4: overflow detection
// ---------------------------------------------------------------------------
bool testOverflow(TestEnv& env) {
    RequestQueueOptions opts{};
    opts.capacity = 1024u;
    auto made = RequestQueue::create(env.allocatorRef(), opts);
    if (!made) {
        std::fprintf(stderr, "[request_queue_test] case 4 FAIL: create\n");
        return false;
    }

    RequestQueueHeader hdr{};
    hdr.count      = 2u;
    hdr.capacity   = 1024u;
    hdr.overflowed = 1u;
    std::array<u32, 2> slots = {77u, 88u};
    if (!writeQueue(env.bundle, made->buffer(), hdr, std::span<const u32>(slots))) {
        std::fprintf(stderr, "[request_queue_test] case 4 FAIL: writeQueue\n");
        return false;
    }

    std::array<u32, 8> out{};
    const u32 n = made->drain(std::span<u32>(out));
    if (n != 2u) {
        std::fprintf(stderr, "[request_queue_test] case 4 FAIL: drain returned %u\n", n);
        return false;
    }
    const auto s = made->stats();
    if (s.overflowEvents != 1u) {
        std::fprintf(stderr, "[request_queue_test] case 4 FAIL: overflowEvents=%llu\n",
                     static_cast<unsigned long long>(s.overflowEvents));
        return false;
    }
    std::printf("[request_queue_test] case 4 PASS: overflow flag bumps stats.overflowEvents.\n");
    return true;
}

// ---------------------------------------------------------------------------
// Extra sanity: options validation
// ---------------------------------------------------------------------------
bool testOptionsValidation(TestEnv& env) {
    {
        RequestQueueOptions opts{};
        opts.capacity = 4u;  // below minimum 16
        auto r = RequestQueue::create(env.allocatorRef(), opts);
        if (r || r.error().kind != RequestQueueErrorKind::CapacityTooSmall) {
            std::fprintf(stderr, "[request_queue_test] options FAIL: capacity=4 accepted\n");
            return false;
        }
    }
    {
        RequestQueueOptions opts{};
        opts.capacity = (1u << 20) + 1u;  // above max
        auto r = RequestQueue::create(env.allocatorRef(), opts);
        if (r || r.error().kind != RequestQueueErrorKind::CapacityTooLarge) {
            std::fprintf(stderr, "[request_queue_test] options FAIL: capacity too large accepted\n");
            return false;
        }
    }
    std::printf("[request_queue_test] options PASS: capacity bounds enforced.\n");
    return true;
}

}  // namespace

int main(int /*argc*/, char** /*argv*/) {
    TestEnv env;
    if (!makeEnv(env)) {
        std::fprintf(stderr, "[request_queue_test] failed to init Vulkan bundle\n");
        return EXIT_FAILURE;
    }

    bool ok = true;
    ok &= testCreateStats(env);
    ok &= testManualDrain(env);
    ok &= testSmallSpan(env);
    ok &= testOverflow(env);
    ok &= testOptionsValidation(env);

    if (!ok) {
        std::fprintf(stderr, "[request_queue_test] FAIL\n");
        return EXIT_FAILURE;
    }
    std::printf("[request_queue_test] ALL CASES PASS\n");
    return EXIT_SUCCESS;
}
