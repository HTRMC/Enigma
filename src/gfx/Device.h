#pragma once

#include "core/Types.h"

#include <volk.h>

#include <optional>

namespace enigma::gfx {

class Instance;

// GPU capability tier, detected at construction from RT extension support
// and device-local VRAM size. Drives renderer feature selection: RT
// settings are grayed-out on Min tier; upscaling is unlocked on ExtremeRT.
enum class GpuTier {
    Min,         // No RT hardware (e.g. GTX 1650). Rasterization-only path.
    Recommended, // Hardware RT, ~4-8 GB VRAM (e.g. RTX 2060 / 2070).
    Extreme,     // Hardware RT, ~8-16 GB VRAM (e.g. RTX 3080 / 4070).
    ExtremeRT,   // Hardware RT, 16+ GB VRAM (e.g. RTX 4090). Upscaling unlocked.
};

// Bundle of required Vulkan feature structs chained via pNext. Declared
// as a self-contained unit so it can be populated once, verified against
// the physical device, and then passed straight to VkDeviceCreateInfo::pNext.
struct RequiredFeatures {
    VkPhysicalDeviceFeatures2         features2;
    VkPhysicalDeviceVulkan11Features  v11;
    VkPhysicalDeviceVulkan12Features  v12;
    VkPhysicalDeviceVulkan13Features  v13;

    // Populate structs and wire the pNext chain. After this call
    // features2.pNext points at v11, v11.pNext at v12, etc.
    void requestAllRequired();
};

// Owns both physical-device selection and the logical VkDevice.
// Exposes optional async compute and transfer queues when the physical
// device has dedicated queue families for them.
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

    // Async compute queue — present only when the device has a dedicated
    // compute-only family (VK_QUEUE_COMPUTE_BIT without GRAPHICS bit).
    // Returns std::nullopt on devices with a unified graphics+compute family;
    // callers must fall back to the graphics queue in that case.
    std::optional<VkQueue> computeQueue()       const { return m_computeQueue;       }
    std::optional<u32>     computeQueueFamily() const { return m_computeQueueFamily; }

    // Dedicated DMA/transfer queue — present only when the device has a
    // transfer-only family (TRANSFER bit, no GRAPHICS or COMPUTE bits).
    std::optional<VkQueue> transferQueue()       const { return m_transferQueue;       }
    std::optional<u32>     transferQueueFamily() const { return m_transferQueueFamily; }

    // GPU capability tier detected at construction time.
    GpuTier gpuTier() const { return m_gpuTier; }

    // True only when VK_EXT_mesh_shader was found and enabled at device creation.
    bool supportsMeshShaders() const { return m_meshShadersEnabled; }

    // True when VkPhysicalDeviceFeatures::fillModeNonSolid was enabled (wireframe debug).
    bool fillModeNonSolidSupported() const { return m_fillModeNonSolidSupported; }

    const VkPhysicalDeviceProperties& properties() const { return m_properties; }

private:
    void    pickPhysicalDevice(VkInstance instance);
    GpuTier detectTier() const;

    VkPhysicalDevice            m_physical            = VK_NULL_HANDLE;
    VkDevice                    m_device              = VK_NULL_HANDLE;
    VkQueue                     m_graphicsQueue       = VK_NULL_HANDLE;
    u32                         m_graphicsQueueFamily = 0;
    std::optional<VkQueue>      m_computeQueue;
    std::optional<u32>          m_computeQueueFamily;
    std::optional<VkQueue>      m_transferQueue;
    std::optional<u32>          m_transferQueueFamily;
    GpuTier                     m_gpuTier             = GpuTier::Min;
    bool                        m_meshShadersEnabled  = false;
    bool                        m_fillModeNonSolidSupported = false;
    VkPhysicalDeviceProperties  m_properties{};
};

} // namespace enigma::gfx
