#include "gfx/DescriptorAllocator.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "gfx/Device.h"

#include <algorithm>
#include <array>
#include <cstdint>

namespace enigma::gfx {

namespace {

// Binding indices — must match the shader declarations in shaders/*.
constexpr u32 kBindingSampledImage  = 0;
constexpr u32 kBindingStorageImage  = 1;
constexpr u32 kBindingStorageBuffer = 2;
constexpr u32 kBindingSampler       = 3;

// Query descriptor-indexing limits from the physical device and clamp
// the caller's requested per-type caps down to what the driver actually
// supports. Defensive against devices that report lower UAB limits than
// their regular limits (rare but possible on older mobile silicon).
DescriptorAllocator::Caps clampCapsToDevice(VkPhysicalDevice physical,
                                            const DescriptorAllocator::Caps& requested) {
    VkPhysicalDeviceDescriptorIndexingProperties indexingProps{};
    indexingProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES;

    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &indexingProps;

    vkGetPhysicalDeviceProperties2(physical, &props2);

    DescriptorAllocator::Caps clamped;
    clamped.sampledImages  = std::min(requested.sampledImages,
                                      indexingProps.maxDescriptorSetUpdateAfterBindSampledImages);
    clamped.storageImages  = std::min(requested.storageImages,
                                      indexingProps.maxDescriptorSetUpdateAfterBindStorageImages);
    clamped.storageBuffers = std::min(requested.storageBuffers,
                                      indexingProps.maxDescriptorSetUpdateAfterBindStorageBuffers);
    clamped.samplers       = std::min(requested.samplers,
                                      indexingProps.maxDescriptorSetUpdateAfterBindSamplers);
    return clamped;
}

} // namespace

DescriptorAllocator::DescriptorAllocator(Device& device, Caps caps)
    : m_device(&device) {
    m_caps = clampCapsToDevice(device.physical(), caps);

    ENIGMA_LOG_INFO("[renderer] bindless layout: sampledImages={} storageImages={} "
                    "storageBuffers={} samplers={}",
                    m_caps.sampledImages, m_caps.storageImages,
                    m_caps.storageBuffers, m_caps.samplers);

    // -------------------------------------------------------------------
    // Layout: 4 bindings, UPDATE_AFTER_BIND + PARTIALLY_BOUND on all four.
    // VARIABLE_DESCRIPTOR_COUNT on binding 3 only (last binding rule).
    // -------------------------------------------------------------------
    const std::array<VkDescriptorSetLayoutBinding, 4> bindings = {{
        {
            kBindingSampledImage,
            VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            m_caps.sampledImages,
            VK_SHADER_STAGE_ALL,
            nullptr,
        },
        {
            kBindingStorageImage,
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            m_caps.storageImages,
            VK_SHADER_STAGE_ALL,
            nullptr,
        },
        {
            kBindingStorageBuffer,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            m_caps.storageBuffers,
            VK_SHADER_STAGE_ALL,
            nullptr,
        },
        {
            kBindingSampler,
            VK_DESCRIPTOR_TYPE_SAMPLER,
            m_caps.samplers,
            VK_SHADER_STAGE_ALL,
            nullptr,
        },
    }};

    constexpr VkDescriptorBindingFlags kCommonFlags =
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;

    const std::array<VkDescriptorBindingFlags, 4> bindingFlags = {{
        kCommonFlags,
        kCommonFlags,
        kCommonFlags,
        kCommonFlags | VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT,
    }};

    VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{};
    bindingFlagsInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    bindingFlagsInfo.bindingCount  = static_cast<u32>(bindingFlags.size());
    bindingFlagsInfo.pBindingFlags = bindingFlags.data();

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.pNext        = &bindingFlagsInfo;
    layoutInfo.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    layoutInfo.bindingCount = static_cast<u32>(bindings.size());
    layoutInfo.pBindings    = bindings.data();

    ENIGMA_VK_CHECK(vkCreateDescriptorSetLayout(m_device->logical(), &layoutInfo, nullptr, &m_layout));

    // -------------------------------------------------------------------
    // Pool sized to hold exactly the four type arrays. We allocate one
    // set (the global set); there are no per-draw sets in this engine.
    // -------------------------------------------------------------------
    const std::array<VkDescriptorPoolSize, 4> poolSizes = {{
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  m_caps.sampledImages  },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  m_caps.storageImages  },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, m_caps.storageBuffers },
        { VK_DESCRIPTOR_TYPE_SAMPLER,        m_caps.samplers       },
    }};

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags         = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    poolInfo.maxSets       = 1;
    poolInfo.poolSizeCount = static_cast<u32>(poolSizes.size());
    poolInfo.pPoolSizes    = poolSizes.data();

    ENIGMA_VK_CHECK(vkCreateDescriptorPool(m_device->logical(), &poolInfo, nullptr, &m_pool));

    // -------------------------------------------------------------------
    // Allocate the single global set. The VARIABLE_COUNT descriptor on
    // binding 3 (samplers) needs a companion count struct telling the
    // driver the actual array length to allocate for that binding.
    // -------------------------------------------------------------------
    const u32 samplerVariableCount = m_caps.samplers;

    VkDescriptorSetVariableDescriptorCountAllocateInfo variableCountInfo{};
    variableCountInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO;
    variableCountInfo.descriptorSetCount = 1;
    variableCountInfo.pDescriptorCounts  = &samplerVariableCount;

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.pNext              = &variableCountInfo;
    allocInfo.descriptorPool     = m_pool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts        = &m_layout;

    ENIGMA_VK_CHECK(vkAllocateDescriptorSets(m_device->logical(), &allocInfo, &m_globalSet));
}

DescriptorAllocator::~DescriptorAllocator() {
    if (m_device == nullptr) return;
    VkDevice dev = m_device->logical();
    if (m_pool   != VK_NULL_HANDLE) vkDestroyDescriptorPool(dev, m_pool, nullptr);
    if (m_layout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(dev, m_layout, nullptr);
}

u32 DescriptorAllocator::registerStorageBuffer(VkBuffer buffer, VkDeviceSize size) {
    ENIGMA_ASSERT(m_nextStorageBuffer < m_caps.storageBuffers);
    const u32 slot = m_nextStorageBuffer++;

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = buffer;
    bufferInfo.offset = 0;
    bufferInfo.range  = size;

    VkWriteDescriptorSet write{};
    write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet          = m_globalSet;
    write.dstBinding      = kBindingStorageBuffer;
    write.dstArrayElement = slot;
    write.descriptorCount = 1;
    write.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo     = &bufferInfo;

    vkUpdateDescriptorSets(m_device->logical(), 1, &write, 0, nullptr);
    return slot;
}

// Milestone-2 implementations. Pattern mirrors registerStorageBuffer
// above: descriptor image/buffer info + single vkUpdateDescriptorSets
// write per call. UPDATE_AFTER_BIND + PARTIALLY_BOUND semantics make
// these legal at any time, including after the set has been bound to
// a command buffer.
u32 DescriptorAllocator::registerSampledImage(VkImageView view, VkImageLayout layout) {
    ENIGMA_ASSERT(m_nextSampledImage < m_caps.sampledImages);
    const u32 slot = m_nextSampledImage++;

    VkDescriptorImageInfo imageInfo{};
    imageInfo.sampler     = VK_NULL_HANDLE;
    imageInfo.imageView   = view;
    imageInfo.imageLayout = layout;

    VkWriteDescriptorSet write{};
    write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet          = m_globalSet;
    write.dstBinding      = kBindingSampledImage;
    write.dstArrayElement = slot;
    write.descriptorCount = 1;
    write.descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    write.pImageInfo      = &imageInfo;

    vkUpdateDescriptorSets(m_device->logical(), 1, &write, 0, nullptr);
    return slot;
}

// Remaining stub: storage image lands at a later milestone 2 step
// (compute pass writing into a procedural storage image). Kept
// asserting so any premature caller lights up immediately.
u32 DescriptorAllocator::registerStorageImage(VkImageView) {
    ENIGMA_ASSERT(false && "registerStorageImage: not implemented yet");
    return UINT32_MAX;
}

u32 DescriptorAllocator::registerSampler(VkSampler sampler) {
    ENIGMA_ASSERT(m_nextSampler < m_caps.samplers);
    const u32 slot = m_nextSampler++;

    VkDescriptorImageInfo imageInfo{};
    imageInfo.sampler     = sampler;
    imageInfo.imageView   = VK_NULL_HANDLE;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkWriteDescriptorSet write{};
    write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet          = m_globalSet;
    write.dstBinding      = kBindingSampler;
    write.dstArrayElement = slot;
    write.descriptorCount = 1;
    write.descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLER;
    write.pImageInfo      = &imageInfo;

    vkUpdateDescriptorSets(m_device->logical(), 1, &write, 0, nullptr);
    return slot;
}

} // namespace enigma::gfx
