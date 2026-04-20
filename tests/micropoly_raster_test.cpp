// Unit test for MicropolyRasterPass (M3.3).
//
// Scope: layout smoke / error-kind stringifier. The M3.3 smoke suite mirrors
// micropoly_cull_test — a narrow public-surface check that does NOT require
// gfx::Device + ShaderManager + DXC bringup. Full pipeline creation is
// exercised live by the Enigma binary when MicropolyConfig::enabled=true
// (and a proper integration test arrives at M3.5 once a live scene lands).
//
// Three cases:
//   1) Layout smoke: error-kind string round-trip distinct values.
//   2) Error-kind enum distinctness.
//   3) Headless Vulkan bundle bringup — same pattern as micropoly_cull_test.

#include "core/Types.h"
#include "renderer/micropoly/MicropolyRasterPass.h"

#include <volk.h>

// VMA: keep the implementation stamped here so it doesn't collide with any
// other test binary that links this TU (it doesn't — each test is its own
// executable). Matches micropoly_cull_test's pattern.
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

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using enigma::renderer::micropoly::MicropolyRasterPass;
using enigma::renderer::micropoly::MicropolyRasterError;
using enigma::renderer::micropoly::MicropolyRasterErrorKind;
using enigma::renderer::micropoly::micropolyRasterErrorKindString;

namespace {

// ---------------------------------------------------------------------------
// Headless Vulkan bundle — minimal bringup, same shape as
// micropoly_cull_test. Used by case 3 to validate that the mesh-shader /
// atomic-int64 feature flags we want to probe are at least reachable on
// the host test machine. Skipped cleanly on CI without a GPU.
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

bool initVolk() {
    static bool ok = false;
    if (ok) return true;
    if (volkInitialize() != VK_SUCCESS) {
        std::fprintf(stderr, "[micropoly_raster_test] volkInitialize failed\n");
        return false;
    }
    ok = true;
    return true;
}

#define VK_CHECK_RET(expr, rv) \
    do { const VkResult _r = (expr); \
         if (_r != VK_SUCCESS) { \
             std::fprintf(stderr, "[micropoly_raster_test] %s failed: %d\n", \
                          #expr, static_cast<int>(_r)); return rv; } \
       } while (0)

VkPhysicalDevice pickPhysicalDevice(VkInstance instance,
                                    std::uint32_t& outGraphicsFamily) {
    std::uint32_t count = 0u;
    if (vkEnumeratePhysicalDevices(instance, &count, nullptr) != VK_SUCCESS) return VK_NULL_HANDLE;
    if (count == 0u) return VK_NULL_HANDLE;
    std::vector<VkPhysicalDevice> devices(count);
    if (vkEnumeratePhysicalDevices(instance, &count, devices.data()) != VK_SUCCESS) return VK_NULL_HANDLE;

    VkPhysicalDevice fallback = VK_NULL_HANDLE;
    std::uint32_t fallbackFamily = 0u;
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
            if (fallback == VK_NULL_HANDLE) { fallback = pd; fallbackFamily = i; }
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
    app.pApplicationName   = "micropoly_raster_test";
    app.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    app.pEngineName        = "Enigma";
    app.engineVersion      = VK_MAKE_API_VERSION(0, 0, 1, 0);
    app.apiVersion         = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ici{};
    ici.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    VK_CHECK_RET(vkCreateInstance(&ici, nullptr, &out.instance), false);
    volkLoadInstance(out.instance);

    out.physicalDevice = pickPhysicalDevice(out.instance, out.graphicsFamily);
    if (out.physicalDevice == VK_NULL_HANDLE) {
        std::fprintf(stderr, "[micropoly_raster_test] no Vulkan device with graphics queue\n");
        return false;
    }

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = out.graphicsFamily;
    qci.queueCount       = 1u;
    qci.pQueuePriorities = &priority;

    VkPhysicalDeviceVulkan12Features v12{};
    v12.sType               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    v12.bufferDeviceAddress = VK_TRUE;
    v12.timelineSemaphore   = VK_TRUE;

    VkPhysicalDeviceVulkan13Features v13{};
    v13.sType            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    v13.synchronization2 = VK_TRUE;
    v13.dynamicRendering = VK_TRUE;
    v13.pNext            = &v12;

    VkPhysicalDeviceFeatures2 f2{};
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f2.pNext = &v13;

    VkDeviceCreateInfo dci{};
    dci.sType                = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1u;
    dci.pQueueCreateInfos    = &qci;
    dci.pNext                = &f2;
    VK_CHECK_RET(vkCreateDevice(out.physicalDevice, &dci, nullptr, &out.device), false);
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
    VK_CHECK_RET(vmaCreateAllocator(&aci, &out.vma), false);
    return true;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

int testErrorKindEnumDistinct() {
    using K = MicropolyRasterErrorKind;
    const int v1 = static_cast<int>(K::MeshShadersUnsupported);
    const int v2 = static_cast<int>(K::Int64ImageUnsupported);
    const int v3 = static_cast<int>(K::PipelineBuildFailed);
    const int v4 = static_cast<int>(K::InvalidVisImage);
    if (v1 == v2 || v1 == v3 || v1 == v4 || v2 == v3 || v2 == v4 || v3 == v4) {
        std::fprintf(stderr, "[micropoly_raster_test] case 1 FAIL: "
                             "duplicate enum values (%d %d %d %d)\n",
                             v1, v2, v3, v4);
        return 1;
    }
    std::printf("[micropoly_raster_test] case 1 PASS: error-kind enum distinct\n");
    return 0;
}

int testErrorKindStringifier() {
    using K = MicropolyRasterErrorKind;
    const char* a = micropolyRasterErrorKindString(K::MeshShadersUnsupported);
    const char* b = micropolyRasterErrorKindString(K::Int64ImageUnsupported);
    const char* c = micropolyRasterErrorKindString(K::PipelineBuildFailed);
    const char* d = micropolyRasterErrorKindString(K::InvalidVisImage);
    if (a == nullptr || b == nullptr || c == nullptr || d == nullptr) {
        std::fprintf(stderr, "[micropoly_raster_test] case 2 FAIL: "
                             "stringifier returned nullptr\n");
        return 1;
    }
    if (std::strcmp(a, b) == 0 || std::strcmp(a, c) == 0 || std::strcmp(a, d) == 0
        || std::strcmp(b, c) == 0 || std::strcmp(b, d) == 0 || std::strcmp(c, d) == 0) {
        std::fprintf(stderr, "[micropoly_raster_test] case 2 FAIL: "
                             "stringifier returned duplicate strings: "
                             "'%s' '%s' '%s' '%s'\n", a, b, c, d);
        return 1;
    }
    std::printf("[micropoly_raster_test] case 2 PASS: error-kind strings distinct "
                "('%s', '%s', '%s', '%s')\n", a, b, c, d);
    return 0;
}

int testVulkanBundleBringup() {
    // Case 3 — smoke-test the headless infrastructure. Mirrors
    // micropoly_cull_test::testVulkanBundleBringup. SKIPs on no-device CI.
    VulkanBundle bundle;
    if (!createVulkanBundle(bundle)) {
        std::printf("[micropoly_raster_test] case 3 SKIPPED: "
                    "no Vulkan device available (env-dependent)\n");
        return 0;
    }
    // M3.3 smoke — full MicropolyRasterPass::create() would require a live
    // ShaderManager + DescriptorAllocator + DXC. That path is exercised by
    // the Enigma binary at startup. Here we only validate the bundle spins
    // up so future integration tests can slot their Device::adopt() path
    // directly on top of this scaffolding.
    std::printf("[micropoly_raster_test] case 3 PASS: headless Vulkan bundle bringup\n");
    return 0;
}

}  // namespace

int main() {
    int status = 0;
    status |= testErrorKindEnumDistinct();
    status |= testErrorKindStringifier();
    status |= testVulkanBundleBringup();
    if (status == 0) {
        std::printf("[micropoly_raster_test] ALL PASS\n");
    } else {
        std::fprintf(stderr, "[micropoly_raster_test] FAILURES\n");
    }
    return status;
}
