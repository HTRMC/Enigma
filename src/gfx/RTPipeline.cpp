#include "gfx/RTPipeline.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "gfx/Allocator.h"
#include "gfx/Device.h"

#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#include <vk_mem_alloc.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

namespace enigma::gfx {

RTPipeline::RTPipeline(Device& device, Allocator& allocator, const CreateInfo& info)
    : m_device(&device)
    , m_allocator(&allocator) {
    ENIGMA_ASSERT(info.raygenModule      != VK_NULL_HANDLE);
    ENIGMA_ASSERT(info.missModule        != VK_NULL_HANDLE);
    ENIGMA_ASSERT(info.closestHitModule  != VK_NULL_HANDLE);
    ENIGMA_ASSERT(info.globalSetLayout   != VK_NULL_HANDLE);

    VkDevice dev = device.logical();

    // Pipeline layout: set=0 is the global bindless set.
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR
                         | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR
                         | VK_SHADER_STAGE_MISS_BIT_KHR;
    pushRange.offset     = 0;
    pushRange.size       = info.pushConstantSize;

    VkPipelineLayoutCreateInfo layoutCI{};
    layoutCI.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutCI.setLayoutCount         = 1;
    layoutCI.pSetLayouts            = &info.globalSetLayout;
    layoutCI.pushConstantRangeCount = info.pushConstantSize > 0 ? 1u : 0u;
    layoutCI.pPushConstantRanges    = info.pushConstantSize > 0 ? &pushRange : nullptr;

    ENIGMA_VK_CHECK(vkCreatePipelineLayout(dev, &layoutCI, nullptr, &m_layout));

    // Shader stages: raygen(0), miss(1), closest-hit(2).
    std::array<VkPipelineShaderStageCreateInfo, 3> stages{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[0].module = info.raygenModule;
    stages[0].pName  = info.raygenEntry;

    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[1].module = info.missModule;
    stages[1].pName  = info.missEntry;

    stages[2].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[2].stage  = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stages[2].module = info.closestHitModule;
    stages[2].pName  = info.closestHitEntry;

    // Shader groups: general(raygen), general(miss), triangles-hit-group(closest-hit).
    std::array<VkRayTracingShaderGroupCreateInfoKHR, 3> groups{};

    // Raygen group.
    groups[0].sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[0].type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[0].generalShader      = 0;
    groups[0].closestHitShader   = VK_SHADER_UNUSED_KHR;
    groups[0].anyHitShader       = VK_SHADER_UNUSED_KHR;
    groups[0].intersectionShader = VK_SHADER_UNUSED_KHR;

    // Miss group.
    groups[1].sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[1].type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[1].generalShader      = 1;
    groups[1].closestHitShader   = VK_SHADER_UNUSED_KHR;
    groups[1].anyHitShader       = VK_SHADER_UNUSED_KHR;
    groups[1].intersectionShader = VK_SHADER_UNUSED_KHR;

    // Closest-hit group.
    groups[2].sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[2].type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    groups[2].generalShader      = VK_SHADER_UNUSED_KHR;
    groups[2].closestHitShader   = 2;
    groups[2].anyHitShader       = VK_SHADER_UNUSED_KHR;
    groups[2].intersectionShader = VK_SHADER_UNUSED_KHR;

    m_groupCount = static_cast<u32>(groups.size());

    VkRayTracingPipelineCreateInfoKHR pipelineCI{};
    pipelineCI.sType                        = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    pipelineCI.stageCount                   = static_cast<u32>(stages.size());
    pipelineCI.pStages                      = stages.data();
    pipelineCI.groupCount                   = m_groupCount;
    pipelineCI.pGroups                      = groups.data();
    pipelineCI.maxPipelineRayRecursionDepth = info.maxRecursionDepth;
    pipelineCI.layout                       = m_layout;

    ENIGMA_VK_CHECK(vkCreateRayTracingPipelinesKHR(
        dev, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &pipelineCI, nullptr, &m_pipeline));

    buildSBT(device, allocator);

    ENIGMA_LOG_INFO("[rt] pipeline created (recursion={})", info.maxRecursionDepth);
}

RTPipeline::~RTPipeline() {
    if (m_device == nullptr) return;
    VkDevice dev = m_device->logical();

    if (m_sbtBuffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_allocator->handle(), m_sbtBuffer, m_sbtAlloc);
    }
    if (m_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(dev, m_pipeline, nullptr);
    }
    if (m_layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(dev, m_layout, nullptr);
    }
}

void RTPipeline::buildSBT(Device& device, Allocator& allocator) {
    VkDevice dev = device.logical();

    // Query RT pipeline properties for handle size and alignment.
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProps{};
    rtProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;

    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &rtProps;
    vkGetPhysicalDeviceProperties2(device.physical(), &props2);

    const u32 handleSize      = rtProps.shaderGroupHandleSize;
    const u32 handleAlignment = rtProps.shaderGroupHandleAlignment;
    const u32 baseAlignment   = rtProps.shaderGroupBaseAlignment;

    // Aligned handle size (each entry in the SBT).
    auto alignUp = [](u32 val, u32 align) -> u32 {
        return (val + align - 1) & ~(align - 1);
    };
    const u32 handleSizeAligned = alignUp(handleSize, handleAlignment);

    // Region sizes (aligned to base alignment).
    const u32 raygenRegionSize   = alignUp(handleSizeAligned, baseAlignment);
    const u32 missRegionSize     = alignUp(handleSizeAligned, baseAlignment);
    const u32 hitGroupRegionSize = alignUp(handleSizeAligned, baseAlignment);
    const u32 totalSBTSize       = raygenRegionSize + missRegionSize + hitGroupRegionSize;

    // Get shader group handles from the pipeline.
    const u32 handleDataSize = m_groupCount * handleSize;
    std::vector<u8> handleData(handleDataSize);
    ENIGMA_VK_CHECK(vkGetRayTracingShaderGroupHandlesKHR(
        dev, m_pipeline, 0, m_groupCount, handleDataSize, handleData.data()));

    // Create the SBT buffer.
    VkBufferCreateInfo bufCI{};
    bufCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufCI.size  = totalSBTSize;
    bufCI.usage = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR
                | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_AUTO;
    allocCI.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                  | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo allocResult{};
    ENIGMA_VK_CHECK(vmaCreateBuffer(allocator.handle(), &bufCI, &allocCI,
                                    &m_sbtBuffer, &m_sbtAlloc, &allocResult));

    // Write handles into the SBT buffer with proper alignment.
    auto* mapped = static_cast<u8*>(allocResult.pMappedData);
    ENIGMA_ASSERT(mapped != nullptr);
    std::memset(mapped, 0, totalSBTSize);

    // Group 0 = raygen, Group 1 = miss, Group 2 = hit.
    std::memcpy(mapped, handleData.data(), handleSize);
    std::memcpy(mapped + raygenRegionSize, handleData.data() + handleSize, handleSize);
    std::memcpy(mapped + raygenRegionSize + missRegionSize,
                handleData.data() + 2 * handleSize, handleSize);

    // Query buffer device address.
    VkBufferDeviceAddressInfo addrInfo{};
    addrInfo.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    addrInfo.buffer = m_sbtBuffer;
    const VkDeviceAddress sbtAddress = vkGetBufferDeviceAddress(dev, &addrInfo);

    m_raygenRegion.deviceAddress = sbtAddress;
    m_raygenRegion.stride        = handleSizeAligned;
    m_raygenRegion.size          = raygenRegionSize;

    m_missRegion.deviceAddress = sbtAddress + raygenRegionSize;
    m_missRegion.stride        = handleSizeAligned;
    m_missRegion.size          = missRegionSize;

    m_hitGroupRegion.deviceAddress = sbtAddress + raygenRegionSize + missRegionSize;
    m_hitGroupRegion.stride        = handleSizeAligned;
    m_hitGroupRegion.size          = hitGroupRegionSize;

    // Callable region is unused.
    m_callableRegion = {};
}

} // namespace enigma::gfx
