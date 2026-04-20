// Unit test for MicropolySwRasterPass (M4.2 — binning).
//
// Mirrors micropoly_raster_test. Three cases:
//   1) Error-kind enum distinctness.
//   2) Error-kind stringifier round-trips to distinct strings.
//   3) Headless Vulkan bundle bringup (smoke — ensures the test harness is
//      at least reachable on this host; full pipeline creation needs a
//      live ShaderManager/DXC and is exercised by the Enigma binary).
//
// Plain main, printf output. Exit 0 on pass.

#include "core/Types.h"
#include "renderer/micropoly/MicropolySwRasterPass.h"

#include <volk.h>

// VMA implementation lives in this TU — no other file in this test binary
// stamps VMA_IMPLEMENTATION.
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

using enigma::renderer::micropoly::MicropolySwRasterPass;
using enigma::renderer::micropoly::MicropolySwRasterError;
using enigma::renderer::micropoly::MicropolySwRasterErrorKind;
using enigma::renderer::micropoly::micropolySwRasterErrorKindString;

namespace {

// ---------------------------------------------------------------------------
// Headless Vulkan bundle — identical shape to micropoly_raster_test.
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
        std::fprintf(stderr, "[micropoly_sw_raster_bin_test] volkInitialize failed\n");
        return false;
    }
    ok = true;
    return true;
}

#define VK_CHECK_RET(expr, rv) \
    do { const VkResult _r = (expr); \
         if (_r != VK_SUCCESS) { \
             std::fprintf(stderr, "[micropoly_sw_raster_bin_test] %s failed: %d\n", \
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
    app.pApplicationName   = "micropoly_sw_raster_bin_test";
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
        std::fprintf(stderr, "[micropoly_sw_raster_bin_test] no Vulkan device with graphics queue\n");
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
    using K = MicropolySwRasterErrorKind;
    const int v[] = {
        static_cast<int>(K::MeshShadersUnsupported),
        static_cast<int>(K::PipelineBuildFailed),
        static_cast<int>(K::BufferAllocFailed),
        static_cast<int>(K::BindlessRegistrationFailed),
        static_cast<int>(K::RasterPipelineBuildFailed),
        static_cast<int>(K::Int64ImageUnsupported),
    };
    for (size_t i = 0; i < sizeof(v) / sizeof(v[0]); ++i) {
        for (size_t j = i + 1; j < sizeof(v) / sizeof(v[0]); ++j) {
            if (v[i] == v[j]) {
                std::fprintf(stderr, "[micropoly_sw_raster_bin_test] case 1 FAIL: "
                                     "duplicate enum values at [%zu]=[%zu]=%d\n",
                                     i, j, v[i]);
                return 1;
            }
        }
    }
    std::printf("[micropoly_sw_raster_bin_test] case 1 PASS: error-kind enum distinct\n");
    return 0;
}

int testErrorKindStringifier() {
    using K = MicropolySwRasterErrorKind;
    const char* s[] = {
        micropolySwRasterErrorKindString(K::MeshShadersUnsupported),
        micropolySwRasterErrorKindString(K::PipelineBuildFailed),
        micropolySwRasterErrorKindString(K::BufferAllocFailed),
        micropolySwRasterErrorKindString(K::BindlessRegistrationFailed),
        micropolySwRasterErrorKindString(K::RasterPipelineBuildFailed),
        micropolySwRasterErrorKindString(K::Int64ImageUnsupported),
    };
    for (auto* p : s) {
        if (p == nullptr) {
            std::fprintf(stderr, "[micropoly_sw_raster_bin_test] case 2 FAIL: "
                                 "stringifier returned nullptr\n");
            return 1;
        }
    }
    for (size_t i = 0; i < sizeof(s) / sizeof(s[0]); ++i) {
        for (size_t j = i + 1; j < sizeof(s) / sizeof(s[0]); ++j) {
            if (std::strcmp(s[i], s[j]) == 0) {
                std::fprintf(stderr, "[micropoly_sw_raster_bin_test] case 2 FAIL: "
                                     "duplicate strings '%s' at [%zu]=[%zu]\n",
                                     s[i], i, j);
                return 1;
            }
        }
    }
    std::printf("[micropoly_sw_raster_bin_test] case 2 PASS: error-kind strings distinct "
                "('%s', '%s', '%s', '%s', '%s', '%s')\n",
                s[0], s[1], s[2], s[3], s[4], s[5]);
    return 0;
}

int testVulkanBundleBringup() {
    VulkanBundle bundle;
    if (!createVulkanBundle(bundle)) {
        std::printf("[micropoly_sw_raster_bin_test] case 3 SKIPPED: "
                    "no Vulkan device available (env-dependent)\n");
        return 0;
    }
    // M4.2 smoke — full MicropolySwRasterPass::create() would require a live
    // ShaderManager + DescriptorAllocator + DXC. That path is exercised by
    // the Enigma binary at startup. Here we only validate the bundle spins
    // up so future integration tests can slot their Device::adopt() path
    // directly on top of this scaffolding.
    std::printf("[micropoly_sw_raster_bin_test] case 3 PASS: headless Vulkan bundle bringup\n");
    return 0;
}

// Case 4 — M4.3: verify that the DispatchInputs struct carries the new
// visImage64Bindless field the fragment raster pipeline needs. Full
// create + raster pipeline non-null verification needs DXC/ShaderManager
// + a live device with descriptor pool, exercised at engine bringup
// (see log line "pipelines built (prep+bin+raster)"). The point of this
// test is to catch build regressions where the DispatchInputs field
// gets accidentally removed — a compile-time guard the runtime
// assertion reinforces.
int testDispatchInputsVisImageField() {
    MicropolySwRasterPass::DispatchInputs sin{};
    // Default-init MUST be UINT32_MAX per the struct's default member
    // initializer; otherwise the fragment pipeline would latch a bogus
    // bindless slot and trip VVL. Mirrors every other bindless-slot
    // field in the struct.
    if (sin.visImage64Bindless != UINT32_MAX) {
        std::fprintf(stderr, "[micropoly_sw_raster_bin_test] case 4 FAIL: "
                             "DispatchInputs::visImage64Bindless default is %u, expected UINT32_MAX\n",
                             sin.visImage64Bindless);
        return 1;
    }
    sin.visImage64Bindless = 0xAAu;
    if (sin.visImage64Bindless != 0xAAu) {
        std::fprintf(stderr, "[micropoly_sw_raster_bin_test] case 4 FAIL: "
                             "DispatchInputs::visImage64Bindless not writable\n");
        return 1;
    }
    std::printf("[micropoly_sw_raster_bin_test] case 4 PASS: DispatchInputs::visImage64Bindless "
                "present and default-init to UINT32_MAX\n");
    return 0;
}

}  // namespace

int main() {
    int status = 0;
    status |= testErrorKindEnumDistinct();
    status |= testErrorKindStringifier();
    status |= testVulkanBundleBringup();
    status |= testDispatchInputsVisImageField();
    if (status == 0) {
        std::printf("[micropoly_sw_raster_bin_test] ALL PASS\n");
    } else {
        std::fprintf(stderr, "[micropoly_sw_raster_bin_test] FAILURES\n");
    }
    return status;
}
