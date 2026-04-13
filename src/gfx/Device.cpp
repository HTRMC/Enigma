#include "gfx/Device.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "gfx/Instance.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <string>
#include <vector>

namespace enigma::gfx {

namespace {

constexpr std::array<const char*, 4> kRequiredDeviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
    VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
    VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
};

bool extensionsSupported(VkPhysicalDevice phys) {
    u32 count = 0;
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> exts(count);
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &count, exts.data());
    for (const char* req : kRequiredDeviceExtensions) {
        bool found = false;
        for (const auto& e : exts) {
            if (std::strcmp(e.extensionName, req) == 0) {
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }
    return true;
}

bool findGraphicsQueueFamily(VkPhysicalDevice phys, u32* outFamily) {
    u32 count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &count, families.data());
    for (u32 i = 0; i < count; ++i) {
        if ((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0) {
            if (outFamily != nullptr) {
                *outFamily = i;
            }
            return true;
        }
    }
    return false;
}

u32 scoreDevice(VkPhysicalDevice phys) {
    if (!extensionsSupported(phys)) {
        return 0;
    }
    if (!findGraphicsQueueFamily(phys, nullptr)) {
        return 0;
    }

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(phys, &props);
    switch (props.deviceType) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   return 1000;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 500;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    return 100;
        case VK_PHYSICAL_DEVICE_TYPE_CPU:            return 50;
        default:                                      return 10;
    }
}

} // namespace

void RequiredFeatures::requestAllRequired() {
    v13 = {};
    v13.sType                          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    v13.dynamicRendering               = VK_TRUE;
    v13.synchronization2               = VK_TRUE;
    v13.shaderDemoteToHelperInvocation = VK_TRUE;

    v12 = {};
    v12.sType                                         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    v12.descriptorIndexing                            = VK_TRUE;
    v12.runtimeDescriptorArray                        = VK_TRUE;
    v12.descriptorBindingPartiallyBound               = VK_TRUE;
    v12.descriptorBindingVariableDescriptorCount      = VK_TRUE;
    v12.descriptorBindingSampledImageUpdateAfterBind  = VK_TRUE;
    v12.descriptorBindingStorageImageUpdateAfterBind  = VK_TRUE;
    v12.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
    v12.descriptorBindingUpdateUnusedWhilePending     = VK_TRUE;
    v12.shaderSampledImageArrayNonUniformIndexing     = VK_TRUE;
    v12.shaderStorageBufferArrayNonUniformIndexing    = VK_TRUE;
    v12.shaderStorageImageArrayNonUniformIndexing     = VK_TRUE;
    v12.timelineSemaphore                             = VK_TRUE;
    v12.bufferDeviceAddress                           = VK_TRUE;

    v11 = {};
    v11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;

    features2 = {};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.features.samplerAnisotropy                    = VK_TRUE;
    // Required for DXC-compiled compute shaders: RWTexture2D<float4> emits
    // SPIR-V OpTypeImage with Unknown format; without these features, storage
    // image reads/writes are undefined (LUTs silently stay zeroed → black sky).
    features2.features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
    features2.features.shaderStorageImageReadWithoutFormat  = VK_TRUE;
    // DXC emits Capability Geometry in some VS shaders (e.g. clustered_forward.hlsl);
    // enabling this suppresses the spurious validation error.
    features2.features.geometryShader                       = VK_TRUE;

    features2.pNext = &v11;
    v11.pNext       = &v12;
    v12.pNext       = &v13;
    v13.pNext       = nullptr;
}

Device::Device(Instance& instance) {
    pickPhysicalDevice(instance.handle());
    findGraphicsQueueFamily(m_physical, &m_graphicsQueueFamily);

    ENIGMA_LOG_INFO("[gfx] picked physical device '{}' (queue family {})",
                    m_properties.deviceName, m_graphicsQueueFamily);

    // Feature verification: query what the device actually supports and
    // compare with what Enigma requires. Abort with a clear diagnostic
    // when anything is missing. (This is the step 24 verification path.)
    RequiredFeatures have{};
    have.requestAllRequired(); // sets up sType + pNext chain with all bits 0
    have.v13.dynamicRendering        = VK_FALSE;
    have.v13.synchronization2        = VK_FALSE;
    have.v12.descriptorIndexing      = VK_FALSE;
    have.v12.runtimeDescriptorArray  = VK_FALSE;
    have.v12.descriptorBindingPartiallyBound               = VK_FALSE;
    have.v12.descriptorBindingVariableDescriptorCount      = VK_FALSE;
    have.v12.descriptorBindingSampledImageUpdateAfterBind  = VK_FALSE;
    have.v12.descriptorBindingStorageImageUpdateAfterBind  = VK_FALSE;
    have.v12.descriptorBindingStorageBufferUpdateAfterBind = VK_FALSE;
    have.v12.descriptorBindingUpdateUnusedWhilePending     = VK_FALSE;
    have.v12.shaderSampledImageArrayNonUniformIndexing     = VK_FALSE;
    have.v12.shaderStorageBufferArrayNonUniformIndexing    = VK_FALSE;
    have.v12.shaderStorageImageArrayNonUniformIndexing     = VK_FALSE;
    have.v12.timelineSemaphore                             = VK_FALSE;
    have.v12.bufferDeviceAddress                           = VK_FALSE;
    vkGetPhysicalDeviceFeatures2(m_physical, &have.features2);

    const auto check = [](VkBool32 v, const char* name) {
        if (v != VK_TRUE) {
            ENIGMA_LOG_ERROR("[gfx] missing required feature: {}", name);
            return false;
        }
        return true;
    };
    bool ok = true;
    ok &= check(have.v13.dynamicRendering,                         "Vulkan13.dynamicRendering");
    ok &= check(have.v13.synchronization2,                         "Vulkan13.synchronization2");
    ok &= check(have.v12.descriptorIndexing,                       "Vulkan12.descriptorIndexing");
    ok &= check(have.v12.runtimeDescriptorArray,                   "Vulkan12.runtimeDescriptorArray");
    ok &= check(have.v12.descriptorBindingPartiallyBound,          "Vulkan12.descriptorBindingPartiallyBound");
    ok &= check(have.v12.descriptorBindingVariableDescriptorCount, "Vulkan12.descriptorBindingVariableDescriptorCount");
    ok &= check(have.v12.descriptorBindingSampledImageUpdateAfterBind,  "Vulkan12.descriptorBindingSampledImageUpdateAfterBind");
    ok &= check(have.v12.descriptorBindingStorageImageUpdateAfterBind,  "Vulkan12.descriptorBindingStorageImageUpdateAfterBind");
    ok &= check(have.v12.descriptorBindingStorageBufferUpdateAfterBind, "Vulkan12.descriptorBindingStorageBufferUpdateAfterBind");
    ok &= check(have.v12.timelineSemaphore,                        "Vulkan12.timelineSemaphore");
    ok &= check(have.v12.bufferDeviceAddress,                     "Vulkan12.bufferDeviceAddress");
    ok &= check(have.features2.features.samplerAnisotropy,                    "features.samplerAnisotropy");
    ok &= check(have.features2.features.shaderStorageImageWriteWithoutFormat, "features.shaderStorageImageWriteWithoutFormat");
    ok &= check(have.features2.features.shaderStorageImageReadWithoutFormat,  "features.shaderStorageImageReadWithoutFormat");
    if (!ok) {
        ENIGMA_ASSERT(false);
        return;
    }

    // Build the actual "request" struct for device creation.
    RequiredFeatures want{};
    want.requestAllRequired();

    // Enumerate device extensions to add the spec-mandated
    // VK_KHR_portability_subset if present on the chosen device.
    u32 extCount = 0;
    vkEnumerateDeviceExtensionProperties(m_physical, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> extProps(extCount);
    vkEnumerateDeviceExtensionProperties(m_physical, nullptr, &extCount, extProps.data());

    std::vector<const char*> enabledExts;
    enabledExts.reserve(kRequiredDeviceExtensions.size() + 1);
    for (const char* e : kRequiredDeviceExtensions) {
        enabledExts.push_back(e);
    }
    for (const auto& ep : extProps) {
        if (std::strcmp(ep.extensionName, "VK_KHR_portability_subset") == 0) {
            enabledExts.push_back("VK_KHR_portability_subset");
            ENIGMA_LOG_INFO("[gfx] enabling VK_KHR_portability_subset on device");
            break;
        }
    }

    // Feature structs declared at constructor scope: they must remain alive
    // until vkCreateDevice (pNext chain references them by pointer).
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelStructFeatures{};
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR    rtPipelineFeatures{};
    VkPhysicalDeviceRayQueryFeaturesKHR              rayQueryFeatures{};
    VkPhysicalDeviceMeshShaderFeaturesEXT            meshShaderFeatures{};

    // Conditionally enable RT extensions when hardware supports them.
    {
        bool hasAccelStruct = false;
        bool hasRTPipeline  = false;
        bool hasDeferredOps = false;
        bool hasRayQuery    = false;
        for (const auto& ep : extProps) {
            if (std::strcmp(ep.extensionName, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) == 0)
                hasAccelStruct = true;
            if (std::strcmp(ep.extensionName, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) == 0)
                hasRTPipeline = true;
            if (std::strcmp(ep.extensionName, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME) == 0)
                hasDeferredOps = true;
            if (std::strcmp(ep.extensionName, VK_KHR_RAY_QUERY_EXTENSION_NAME) == 0)
                hasRayQuery = true;
        }
        if (hasAccelStruct && hasRTPipeline && hasDeferredOps) {
            enabledExts.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
            enabledExts.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
            enabledExts.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);

            accelStructFeatures.sType                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
            accelStructFeatures.accelerationStructure = VK_TRUE;
            rtPipelineFeatures.sType                  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
            rtPipelineFeatures.rayTracingPipeline     = VK_TRUE;

            // Chain RT features after v13 in want's pNext.
            want.v13.pNext            = &accelStructFeatures;
            accelStructFeatures.pNext = &rtPipelineFeatures;

            if (hasRayQuery) {
                enabledExts.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
                rayQueryFeatures.sType    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
                rayQueryFeatures.rayQuery = VK_TRUE;
                rtPipelineFeatures.pNext  = &rayQueryFeatures;
            }

            ENIGMA_LOG_INFO("[gfx] enabling RT extensions (acceleration_structure + ray_tracing_pipeline + deferred_host_operations)");
        }
    }

    // Conditionally enable VK_EXT_mesh_shader (task + mesh shaders).
    {
        bool hasMeshShader = false;
        for (const auto& ep : extProps) {
            if (std::strcmp(ep.extensionName, VK_EXT_MESH_SHADER_EXTENSION_NAME) == 0) {
                hasMeshShader = true;
                break;
            }
        }
        if (hasMeshShader) {
            enabledExts.push_back(VK_EXT_MESH_SHADER_EXTENSION_NAME);

            meshShaderFeatures.sType      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT;
            meshShaderFeatures.taskShader = VK_TRUE;
            meshShaderFeatures.meshShader = VK_TRUE;

            // Append to the tail of the existing pNext chain rooted at want.v13.
            auto* tail = reinterpret_cast<VkBaseOutStructure*>(&want.v13);
            while (tail->pNext != nullptr) tail = tail->pNext;
            tail->pNext = reinterpret_cast<VkBaseOutStructure*>(&meshShaderFeatures);

            m_meshShadersEnabled = true;
            ENIGMA_LOG_INFO("[gfx] enabling mesh shaders (VK_EXT_mesh_shader)");
        }
    }

    // Discover optional async compute and dedicated transfer queue families.
    // Compute-only: VK_QUEUE_COMPUTE_BIT set, VK_QUEUE_GRAPHICS_BIT clear.
    // Transfer-only: VK_QUEUE_TRANSFER_BIT set, GRAPHICS and COMPUTE both clear.
    {
        u32 famCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(m_physical, &famCount, nullptr);
        std::vector<VkQueueFamilyProperties> fams(famCount);
        vkGetPhysicalDeviceQueueFamilyProperties(m_physical, &famCount, fams.data());

        for (u32 i = 0; i < famCount; ++i) {
            const VkQueueFlags f = fams[i].queueFlags;
            if (!m_computeQueueFamily.has_value() &&
                (f & VK_QUEUE_COMPUTE_BIT) &&
                !(f & VK_QUEUE_GRAPHICS_BIT) &&
                i != m_graphicsQueueFamily) {
                m_computeQueueFamily = i;
            }
            if (!m_transferQueueFamily.has_value() &&
                (f & VK_QUEUE_TRANSFER_BIT) &&
                !(f & VK_QUEUE_GRAPHICS_BIT) &&
                !(f & VK_QUEUE_COMPUTE_BIT) &&
                i != m_graphicsQueueFamily) {
                m_transferQueueFamily = i;
            }
        }
    }

    const float queuePriority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> queueInfos;
    queueInfos.reserve(3);

    {
        VkDeviceQueueCreateInfo qi{};
        qi.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qi.queueFamilyIndex = m_graphicsQueueFamily;
        qi.queueCount       = 1;
        qi.pQueuePriorities = &queuePriority;
        queueInfos.push_back(qi);
    }
    if (m_computeQueueFamily.has_value()) {
        VkDeviceQueueCreateInfo qi{};
        qi.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qi.queueFamilyIndex = *m_computeQueueFamily;
        qi.queueCount       = 1;
        qi.pQueuePriorities = &queuePriority;
        queueInfos.push_back(qi);
    }
    if (m_transferQueueFamily.has_value()) {
        VkDeviceQueueCreateInfo qi{};
        qi.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qi.queueFamilyIndex = *m_transferQueueFamily;
        qi.queueCount       = 1;
        qi.pQueuePriorities = &queuePriority;
        queueInfos.push_back(qi);
    }

    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.pNext                   = &want.features2;
    deviceInfo.queueCreateInfoCount    = static_cast<u32>(queueInfos.size());
    deviceInfo.pQueueCreateInfos       = queueInfos.data();
    deviceInfo.enabledExtensionCount   = static_cast<u32>(enabledExts.size());
    deviceInfo.ppEnabledExtensionNames = enabledExts.data();

    ENIGMA_VK_CHECK(vkCreateDevice(m_physical, &deviceInfo, nullptr, &m_device));
    ENIGMA_ASSERT(m_device != VK_NULL_HANDLE);

    // Populate device-level function pointers for volk.
    volkLoadDevice(m_device);

    vkGetDeviceQueue(m_device, m_graphicsQueueFamily, 0, &m_graphicsQueue);

    if (m_computeQueueFamily.has_value()) {
        VkQueue q = VK_NULL_HANDLE;
        vkGetDeviceQueue(m_device, *m_computeQueueFamily, 0, &q);
        m_computeQueue = q;
        ENIGMA_LOG_INFO("[gfx] async compute queue: family {}", *m_computeQueueFamily);
    } else {
        ENIGMA_LOG_INFO("[gfx] no dedicated compute family — sharing graphics queue");
    }

    if (m_transferQueueFamily.has_value()) {
        VkQueue q = VK_NULL_HANDLE;
        vkGetDeviceQueue(m_device, *m_transferQueueFamily, 0, &q);
        m_transferQueue = q;
        ENIGMA_LOG_INFO("[gfx] dedicated transfer queue: family {}", *m_transferQueueFamily);
    } else {
        ENIGMA_LOG_INFO("[gfx] no dedicated transfer family — sharing graphics queue");
    }

    m_gpuTier = detectTier();
    ENIGMA_LOG_INFO("[gfx] VkDevice created (extensions = {}, queues = {}, tier = {})",
                    enabledExts.size(), queueInfos.size(), static_cast<int>(m_gpuTier));
}

Device::~Device() {
    if (m_device != VK_NULL_HANDLE) {
        vkDestroyDevice(m_device, nullptr);
        m_device = VK_NULL_HANDLE;
    }
}

void Device::pickPhysicalDevice(VkInstance instance) {
    u32 count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (count == 0) {
        ENIGMA_LOG_ERROR("[gfx] no Vulkan physical devices found");
        ENIGMA_ASSERT(false);
        return;
    }
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());

    VkPhysicalDevice best = VK_NULL_HANDLE;
    u32 bestScore = 0;
    for (VkPhysicalDevice d : devices) {
        const u32 s = scoreDevice(d);
        if (s > bestScore) {
            bestScore = s;
            best      = d;
        }
    }

    if (best == VK_NULL_HANDLE) {
        ENIGMA_LOG_ERROR("[gfx] no physical device satisfies required extensions");
        ENIGMA_ASSERT(false);
        return;
    }

    m_physical = best;
    vkGetPhysicalDeviceProperties(m_physical, &m_properties);
}

GpuTier Device::detectTier() const {
    // Check for hardware RT support (VK_KHR_ray_tracing_pipeline).
    u32 extCount = 0;
    vkEnumerateDeviceExtensionProperties(m_physical, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> exts(extCount);
    vkEnumerateDeviceExtensionProperties(m_physical, nullptr, &extCount, exts.data());

    bool hasRT = false;
    for (const auto& e : exts) {
        if (std::strcmp(e.extensionName, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) == 0) {
            hasRT = true;
            break;
        }
    }

    if (!hasRT) {
        return GpuTier::Min;
    }

    // Measure device-local VRAM by summing DEVICE_LOCAL heaps.
    VkPhysicalDeviceMemoryProperties memProps{};
    vkGetPhysicalDeviceMemoryProperties(m_physical, &memProps);

    VkDeviceSize vramBytes = 0;
    for (u32 i = 0; i < memProps.memoryHeapCount; ++i) {
        if (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            vramBytes += memProps.memoryHeaps[i].size;
        }
    }

    constexpr VkDeviceSize k8GB  = 8ULL  * 1024 * 1024 * 1024;
    constexpr VkDeviceSize k16GB = 16ULL * 1024 * 1024 * 1024;

    if (vramBytes >= k16GB) {
        return GpuTier::ExtremeRT;
    }
    if (vramBytes >= k8GB) {
        return GpuTier::Extreme;
    }
    return GpuTier::Recommended;
}

} // namespace enigma::gfx
