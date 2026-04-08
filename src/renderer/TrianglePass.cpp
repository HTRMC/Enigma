#include "renderer/TrianglePass.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "gfx/Allocator.h"
#include "gfx/DescriptorAllocator.h"
#include "gfx/Device.h"

#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#include <vk_mem_alloc.h>

#include <array>
#include <cstring>

namespace enigma {

namespace {

// NDC-space centered triangle, alpha=1 for padding. Matches the
// coordinates documented in the plan (step 36) so the executor and
// the plan agree byte-for-byte.
constexpr std::array<float, 12> kTriangleVertices = {
    -0.5f, -0.5f, 0.0f, 1.0f,
     0.5f, -0.5f, 0.0f, 1.0f,
     0.0f,  0.5f, 0.0f, 1.0f,
};

} // namespace

TrianglePass::TrianglePass(gfx::Device& device,
                           gfx::Allocator& allocator,
                           gfx::DescriptorAllocator& descriptorAllocator)
    : m_device(&device),
      m_allocator(&allocator) {

    // Create a host-visible SSBO and map it long enough to write the
    // three vertex positions. Host-visible is fine for 48 bytes at a
    // single draw call; a real mesh would use a device-local buffer
    // plus a staging upload.
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size        = sizeof(kTriangleVertices);
    bufferInfo.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage          = VMA_MEMORY_USAGE_AUTO;
    allocInfo.flags          = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                               | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    allocInfo.requiredFlags  = 0;

    VmaAllocationInfo allocationInfo{};
    ENIGMA_VK_CHECK(vmaCreateBuffer(allocator.handle(), &bufferInfo, &allocInfo,
                                    &m_vertexBuffer, &m_vertexAllocation, &allocationInfo));
    ENIGMA_ASSERT(allocationInfo.pMappedData != nullptr);
    std::memcpy(allocationInfo.pMappedData, kTriangleVertices.data(), sizeof(kTriangleVertices));

    // Register in the bindless descriptor set at binding 2.
    m_bindlessSlot = descriptorAllocator.registerStorageBuffer(
        m_vertexBuffer, sizeof(kTriangleVertices));

    ENIGMA_LOG_INFO("[triangle] ssbo created: size={} bytes, bindless slot={}",
                    sizeof(kTriangleVertices), m_bindlessSlot);
}

TrianglePass::~TrianglePass() {
    if (m_allocator != nullptr && m_vertexBuffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_allocator->handle(), m_vertexBuffer, m_vertexAllocation);
    }
}

void TrianglePass::buildPipeline(gfx::ShaderManager&,
                                 VkDescriptorSetLayout,
                                 VkFormat) {
    // Implemented at plan step 37.
}

void TrianglePass::record(VkCommandBuffer, VkDescriptorSet, VkExtent2D) {
    // Implemented at plan step 37.
}

} // namespace enigma
