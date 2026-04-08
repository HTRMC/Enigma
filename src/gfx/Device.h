#pragma once

#include "core/Types.h"

#include <volk.h>

namespace enigma::gfx {

class Instance;

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
