#pragma once

#include "core/Types.h"

#include <volk.h>

namespace enigma::gfx {

class Instance;

// Bundle of required Vulkan feature structs chained via pNext. Declared
// as a self-contained unit at step 23 so step 24 can zero-initialize,
// populate the features we require, and pass it straight to
// `vkGetPhysicalDeviceFeatures2` (for verification) and then to
// `VkDeviceCreateInfo::pNext` (for device creation).
//
// BDA (`bufferDeviceAddress`) is deliberately omitted per ADR — scaffold-
// without-usage violates Principle 3 at this milestone.
struct RequiredFeatures {
    VkPhysicalDeviceFeatures2         features2;
    VkPhysicalDeviceVulkan11Features  v11;
    VkPhysicalDeviceVulkan12Features  v12;
    VkPhysicalDeviceVulkan13Features  v13;

    // Populate the structs and wire the pNext chain. After this call
    // `features2.pNext` points at `v11`, `v11.pNext` at `v12`, etc.
    // The booleans flip VK_TRUE on the features Enigma requires.
    void requestAllRequired();
};

// Owns both physical-device selection and the logical `VkDevice`. The
// selection algorithm prefers discrete GPUs; the feature set requested
// matches `AC15` (dynamic rendering, sync2, descriptor indexing,
// timeline semaphores). BDA is deliberately omitted per ADR.
class Device {
public:
    explicit Device(Instance& instance);
    ~Device();

    Device(const Device&)            = delete;
    Device& operator=(const Device&) = delete;
    Device(Device&&)                 = delete;
    Device& operator=(Device&&)      = delete;

    VkPhysicalDevice physical() const { return m_physical; }
    VkDevice         logical()  const { return m_device;   }
    u32              graphicsQueueFamily() const { return m_graphicsQueueFamily; }
    VkQueue          graphicsQueue() const { return m_graphicsQueue; }

    const VkPhysicalDeviceProperties& properties() const { return m_properties; }

private:
    void pickPhysicalDevice(VkInstance instance);

    VkPhysicalDevice            m_physical            = VK_NULL_HANDLE;
    VkDevice                    m_device              = VK_NULL_HANDLE;
    VkQueue                     m_graphicsQueue       = VK_NULL_HANDLE;
    u32                         m_graphicsQueueFamily = 0;
    VkPhysicalDeviceProperties  m_properties{};
};

} // namespace enigma::gfx
