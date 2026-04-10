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
constexpr u32 kBindingAccelStruct   = 4;

constexpr u32 kAccelStructCount = 1; // single TLAS slot

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
    // Layout: 5 bindings. UPDATE_AFTER_BIND + PARTIALLY_BOUND on 0-3.
    // Binding 4 is the acceleration structure (fixed count 1).
    // The Vulkan spec (VUID-VkDescriptorSetLayoutBindingFlagsCreateInfo-
    // pBindingFlags-03004) requires VARIABLE_DESCRIPTOR_COUNT_BIT on the
    // binding with the HIGHEST binding number. Since binding 4 (AS) is the
    // last binding, NO binding can have VARIABLE_DESCRIPTOR_COUNT — all
    // bindings use fixed descriptor counts.
    // -------------------------------------------------------------------
    const std::array<VkDescriptorSetLayoutBinding, 5> bindings = {{
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
        {
            kBindingAccelStruct,
            VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
            kAccelStructCount,
            VK_SHADER_STAGE_ALL,
            nullptr,
        },
    }};

    constexpr VkDescriptorBindingFlags kCommonFlags =
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;

    const std::array<VkDescriptorBindingFlags, 5> bindingFlags = {{
        kCommonFlags,
        kCommonFlags,
        kCommonFlags,
        kCommonFlags,                               // sampler: fixed count
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT, // AS: no UAB needed
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
    // Pool sized to hold the five type arrays.
    // -------------------------------------------------------------------
    const std::array<VkDescriptorPoolSize, 5> poolSizes = {{
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,               m_caps.sampledImages  },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,               m_caps.storageImages  },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,              m_caps.storageBuffers },
        { VK_DESCRIPTOR_TYPE_SAMPLER,                     m_caps.samplers       },
        { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,  kAccelStructCount     },
    }};

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags         = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    poolInfo.maxSets       = 1;
    poolInfo.poolSizeCount = static_cast<u32>(poolSizes.size());
    poolInfo.pPoolSizes    = poolSizes.data();

    ENIGMA_VK_CHECK(vkCreateDescriptorPool(m_device->logical(), &poolInfo, nullptr, &m_pool));

    // -------------------------------------------------------------------
    // Allocate the single global set. All bindings have fixed counts so
    // no VkDescriptorSetVariableDescriptorCountAllocateInfo is needed.
    // -------------------------------------------------------------------
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
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
    u32 slot = 0;
    if (!m_freeStorageBuffers.empty()) {
        slot = m_freeStorageBuffers.back();
        m_freeStorageBuffers.pop_back();
    } else {
        ENIGMA_ASSERT(m_nextStorageBuffer < m_caps.storageBuffers);
        slot = m_nextStorageBuffer++;
    }

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

u32 DescriptorAllocator::registerSampledImage(VkImageView view, VkImageLayout layout) {
    u32 slot = 0;
    if (!m_freeSampledImages.empty()) {
        slot = m_freeSampledImages.back();
        m_freeSampledImages.pop_back();
    } else {
        ENIGMA_ASSERT(m_nextSampledImage < m_caps.sampledImages);
        slot = m_nextSampledImage++;
    }

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

void DescriptorAllocator::updateSampledImage(u32 slot, VkImageView view, VkImageLayout layout) {
    ENIGMA_ASSERT(slot < m_nextSampledImage && "slot was never registered");

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
}

void DescriptorAllocator::updateStorageImage(u32 slot, VkImageView view) {
    ENIGMA_ASSERT(slot < m_nextStorageImage && "slot was never registered");

    VkDescriptorImageInfo imageInfo{};
    imageInfo.sampler     = VK_NULL_HANDLE;
    imageInfo.imageView   = view;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet write{};
    write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet          = m_globalSet;
    write.dstBinding      = kBindingStorageImage;
    write.dstArrayElement = slot;
    write.descriptorCount = 1;
    write.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write.pImageInfo      = &imageInfo;

    vkUpdateDescriptorSets(m_device->logical(), 1, &write, 0, nullptr);
}

u32 DescriptorAllocator::registerStorageImage(VkImageView view) {
    u32 slot = 0;
    if (!m_freeStorageImages.empty()) {
        slot = m_freeStorageImages.back();
        m_freeStorageImages.pop_back();
    } else {
        ENIGMA_ASSERT(m_nextStorageImage < m_caps.storageImages);
        slot = m_nextStorageImage++;
    }

    VkDescriptorImageInfo imageInfo{};
    imageInfo.sampler     = VK_NULL_HANDLE;
    imageInfo.imageView   = view;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet write{};
    write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet          = m_globalSet;
    write.dstBinding      = kBindingStorageImage;
    write.dstArrayElement = slot;
    write.descriptorCount = 1;
    write.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write.pImageInfo      = &imageInfo;

    vkUpdateDescriptorSets(m_device->logical(), 1, &write, 0, nullptr);
    return slot;
}

u32 DescriptorAllocator::registerSampler(VkSampler sampler) {
    u32 slot = 0;
    if (!m_freeSamplers.empty()) {
        slot = m_freeSamplers.back();
        m_freeSamplers.pop_back();
    } else {
        ENIGMA_ASSERT(m_nextSampler < m_caps.samplers);
        slot = m_nextSampler++;
    }

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

u32 DescriptorAllocator::registerAccelerationStructure(VkAccelerationStructureKHR as) {
    VkWriteDescriptorSetAccelerationStructureKHR asWrite{};
    asWrite.sType                      = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    asWrite.accelerationStructureCount = 1;
    asWrite.pAccelerationStructures    = &as;

    VkWriteDescriptorSet write{};
    write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.pNext           = &asWrite;
    write.dstSet          = m_globalSet;
    write.dstBinding      = kBindingAccelStruct;
    write.dstArrayElement = 0;
    write.descriptorCount = 1;
    write.descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

    vkUpdateDescriptorSets(m_device->logical(), 1, &write, 0, nullptr);
    return 0; // single slot, always index 0
}

void DescriptorAllocator::releaseSampledImage(u32 slot) {
    m_freeSampledImages.push_back(slot);
}

void DescriptorAllocator::releaseStorageImage(u32 slot) {
    m_freeStorageImages.push_back(slot);
}

void DescriptorAllocator::releaseStorageBuffer(u32 slot) {
    m_freeStorageBuffers.push_back(slot);
}

void DescriptorAllocator::releaseSampler(u32 slot) {
    m_freeSamplers.push_back(slot);
}

} // namespace enigma::gfx
