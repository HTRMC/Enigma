// DeviceAdopt.cpp
// ================
// Test-only factory for wrapping an already-created VkDevice into a
// gfx::Device. See Device.h for the contract. Split from Device.cpp so
// micropoly test TUs can link against it WITHOUT pulling in core/Log.h
// (ENIGMA_LOG_*), gfx/Instance.h, or the full feature-probe machinery
// that Device's main ctor uses.
//
// The full Device.cpp still compiles with Instance.h and Log.h; nothing
// here supplants that TU. It is purely additive.

#include "gfx/Device.h"

namespace enigma::gfx {

// Destructor — lives here (rather than in Device.cpp) so test TUs can link
// this single small file without also pulling in the full Device.cpp +
// Instance.h + Log.h surface. Behavior is identical to the old destructor:
// vkDestroyDevice runs iff the logical device was created by this class's
// normal ctor. The `m_externallyOwnedDevice` flag (set only by the AdoptDesc
// ctor) suppresses that call on the adopt path.
Device::~Device() {
    if (m_device != VK_NULL_HANDLE && !m_externallyOwnedDevice) {
        vkDestroyDevice(m_device, nullptr);
    }
    m_device = VK_NULL_HANDLE;
}

std::unique_ptr<Device> Device::adopt(const AdoptDesc& desc) {
    return std::unique_ptr<Device>(new Device(desc));
}

Device::Device(const AdoptDesc& d)
    : m_physical(d.physical),
      m_device(d.device),
      m_graphicsQueue(d.graphicsQueue),
      m_graphicsQueueFamily(d.graphicsQueueFamily),
      m_externallyOwnedDevice(true) {
    if (d.hasTransferQueue) {
        m_transferQueue       = d.transferQueue;
        m_transferQueueFamily = d.transferQueueFamily;
    }
    if (d.physical != VK_NULL_HANDLE) {
        vkGetPhysicalDeviceProperties(d.physical, &m_properties);
    }
    // Leave RT / mesh-shader / atomic probes at their default (false). Adopt
    // is a test vehicle — tests opt into whatever capabilities they need.
}

} // namespace enigma::gfx
