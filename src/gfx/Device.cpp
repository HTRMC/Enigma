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

    // Logical device creation arrives at step 24. For now we only have the
    // selected physical device + properties + graphics queue family.
    findGraphicsQueueFamily(m_physical, &m_graphicsQueueFamily);

    ENIGMA_LOG_INFO("[gfx] picked physical device '{}' (queue family {})",
                    m_properties.deviceName, m_graphicsQueueFamily);
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
