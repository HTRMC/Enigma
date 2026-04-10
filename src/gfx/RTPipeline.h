#pragma once

#include "core/Types.h"

#include <volk.h>

struct VmaAllocation_T;
using VmaAllocation = VmaAllocation_T*;

namespace enigma::gfx {

class Allocator;
class Device;

// RTPipeline
// ==========
// Wraps a VkPipeline (ray tracing), its layout, and the Shader Binding
// Table (SBT). The SBT is a single contiguous GPU buffer covering
// [raygen | miss | hitGroup] with proper alignment from
// VkPhysicalDeviceRayTracingPipelinePropertiesKHR.
class RTPipeline {
public:
    struct CreateInfo {
        VkShaderModule        raygenModule;
        const char*           raygenEntry;
        VkShaderModule        missModule;
        const char*           missEntry;
        VkShaderModule        closestHitModule;
        const char*           closestHitEntry;
        VkDescriptorSetLayout globalSetLayout;
        u32                   pushConstantSize  = 0;
        u32                   maxRecursionDepth = 1;
    };

    RTPipeline(Device& device, Allocator& allocator, const CreateInfo& info);
    ~RTPipeline();

    RTPipeline(const RTPipeline&)            = delete;
    RTPipeline& operator=(const RTPipeline&) = delete;
    RTPipeline(RTPipeline&&)                 = delete;
    RTPipeline& operator=(RTPipeline&&)      = delete;

    VkPipeline       handle() const { return m_pipeline; }
    VkPipelineLayout layout() const { return m_layout; }

    // SBT regions for vkCmdTraceRaysKHR.
    const VkStridedDeviceAddressRegionKHR& raygenRegion()   const { return m_raygenRegion; }
    const VkStridedDeviceAddressRegionKHR& missRegion()     const { return m_missRegion; }
    const VkStridedDeviceAddressRegionKHR& hitGroupRegion() const { return m_hitGroupRegion; }
    const VkStridedDeviceAddressRegionKHR& callableRegion() const { return m_callableRegion; }

private:
    void buildSBT(Device& device, Allocator& allocator);

    Device*          m_device   = nullptr;
    Allocator*       m_allocator = nullptr;
    VkPipeline       m_pipeline = VK_NULL_HANDLE;
    VkPipelineLayout m_layout   = VK_NULL_HANDLE;

    // SBT buffer (single allocation covering raygen + miss + hit groups).
    VkBuffer      m_sbtBuffer = VK_NULL_HANDLE;
    VmaAllocation m_sbtAlloc  = nullptr;

    VkStridedDeviceAddressRegionKHR m_raygenRegion{};
    VkStridedDeviceAddressRegionKHR m_missRegion{};
    VkStridedDeviceAddressRegionKHR m_hitGroupRegion{};
    VkStridedDeviceAddressRegionKHR m_callableRegion{};

    u32 m_groupCount = 0;
};

} // namespace enigma::gfx
