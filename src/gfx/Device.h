#pragma once

#include "core/Types.h"

#include <volk.h>

#include <memory>
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
    // features2.pNext points at v11, v11.pNext at v12, etc. The Vulkan 1.2
    // promoted fields (e.g. shaderBufferInt64Atomics) are surfaced through
    // v12 directly — do NOT chain the legacy VkPhysicalDeviceShaderAtomicInt64Features
    // struct here, VVL rejects that as duplicate feature coverage.
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

    // --- Test-only adopt path (M2.4 micropoly streaming tests) ---
    //
    // Creates a Device instance from already-existing Vulkan handles. Used by
    // micropoly unit/integration tests that bring up their own headless bundle
    // and want to pass it into streaming code as a real `gfx::Device` reference
    // (see Allocator::adopt precedent in M2.2).
    //
    // Ownership: the adopted Device does NOT destroy `physical` OR the logical
    // `device` — the caller retains full ownership of both handles. This lets
    // test harnesses keep their own VulkanBundle destructor logic unchanged
    // (destroy instance/device/vma in their established order) while still
    // giving streaming code a real `gfx::Device` reference. `transferQueueFamily`
    // is optional; when set it also populates transferQueue/transferQueueFamily
    // accessors so MicropolyStreaming's transfer-queue wiring can exercise a
    // real queue.
    //
    // No feature probes are performed — the adopter is responsible for having
    // enabled whatever features it needs (e.g. timelineSemaphore).
    struct AdoptDesc {
        VkPhysicalDevice physical            = VK_NULL_HANDLE;
        VkDevice         device              = VK_NULL_HANDLE;
        VkQueue          graphicsQueue       = VK_NULL_HANDLE;
        u32              graphicsQueueFamily = 0u;
        VkQueue          transferQueue       = VK_NULL_HANDLE;
        u32              transferQueueFamily = 0u;
        bool             hasTransferQueue    = false;
    };
    static std::unique_ptr<Device> adopt(const AdoptDesc& desc);

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

    // True when BOTH VkPhysicalDeviceVulkan12Features::shaderBufferInt64Atomics
    // AND shaderSharedInt64Atomics were advertised and enabled. Both are
    // required for the micropoly software-raster path: buffer atomics for
    // the vis-buffer atomic-min write, shared atomics for workgroup-level
    // tile reductions. Probed-only here, not mandatory for device creation.
    // Gate lives at Device.cpp §305-314.
    bool supportsShaderAtomicInt64() const { return m_shaderAtomicInt64Supported; }

    // True when VkPhysicalDeviceFeatures::sparseBinding AND sparseResidencyImage2D
    // were advertised. Required for VSM shadow pages and micropoly page-streamed
    // geometry residency; optional.
    bool supportsSparseResidency() const { return m_sparseResidencySupported; }

    // True when VK_EXT_shader_image_atomic_int64 was advertised (SPV_EXT_shader_image_int64).
    // Required for packed 64-bit vis image atomic-min on R32G32_UINT alias when
    // R64_UINT is unavailable.
    bool supportsShaderImageInt64() const { return m_shaderImageInt64Supported; }

    // True when the RT extension trio (acceleration_structure + ray_tracing_pipeline
    // + deferred_host_operations) was enabled at device creation. Redundant with
    // `gpuTier() >= GpuTier::Recommended` today but kept explicit so micropoly
    // code paths do not couple to the VRAM-driven tier heuristic.
    bool supportsRayTracing() const { return m_rtEnabled; }

    const VkPhysicalDeviceProperties& properties() const { return m_properties; }

private:
    void    pickPhysicalDevice(VkInstance instance);
    GpuTier detectTier() const;

    // Private ctor used by adopt() — fills members from an externally-owned
    // VkDevice bundle. See Device::adopt() in .cpp for the contract.
    explicit Device(const AdoptDesc& desc);

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
    bool                        m_shaderAtomicInt64Supported = false;
    bool                        m_sparseResidencySupported   = false;
    bool                        m_shaderImageInt64Supported  = false;
    bool                        m_rtEnabled                  = false;
    VkPhysicalDeviceProperties  m_properties{};

    // When true, the destructor does NOT call vkDestroyDevice — the logical
    // device handle was adopted from a caller who retains ownership. Only
    // set by the AdoptDesc private ctor.
    bool m_externallyOwnedDevice = false;
};

} // namespace enigma::gfx
