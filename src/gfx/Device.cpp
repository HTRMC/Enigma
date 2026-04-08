#include "gfx/Device.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "gfx/Instance.h"

#include <array>
#include <cstring>
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
    v13.sType             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    v13.dynamicRendering  = VK_TRUE;
    v13.synchronization2  = VK_TRUE;

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
    // bufferDeviceAddress intentionally NOT enabled (ADR).

    v11 = {};
    v11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;

    features2 = {};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

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

    // One graphics queue.
    const float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{};
    queueInfo.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = m_graphicsQueueFamily;
    queueInfo.queueCount       = 1;
    queueInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.pNext                   = &want.features2;
    deviceInfo.queueCreateInfoCount    = 1;
    deviceInfo.pQueueCreateInfos       = &queueInfo;
    deviceInfo.enabledExtensionCount   = static_cast<u32>(enabledExts.size());
    deviceInfo.ppEnabledExtensionNames = enabledExts.data();

    ENIGMA_VK_CHECK(vkCreateDevice(m_physical, &deviceInfo, nullptr, &m_device));
    ENIGMA_ASSERT(m_device != VK_NULL_HANDLE);

    // Populate device-level function pointers for volk.
    volkLoadDevice(m_device);

    vkGetDeviceQueue(m_device, m_graphicsQueueFamily, 0, &m_graphicsQueue);

    ENIGMA_LOG_INFO("[gfx] VkDevice created (extensions = {})", enabledExts.size());
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

} // namespace enigma::gfx
