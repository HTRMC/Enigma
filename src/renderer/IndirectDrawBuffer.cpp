#include "renderer/IndirectDrawBuffer.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "gfx/Allocator.h"
#include "gfx/DescriptorAllocator.h"
#include "gfx/Device.h"

#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#include <vk_mem_alloc.h>

namespace enigma {

IndirectDrawBuffer::IndirectDrawBuffer(gfx::Device& device, gfx::Allocator& allocator,
                                       gfx::DescriptorAllocator& descriptors)
    : m_device(&device)
    , m_allocator(&allocator)
    , m_descriptors(&descriptors)
{
}

IndirectDrawBuffer::~IndirectDrawBuffer() {
    if (m_commands_slot  != 0) m_descriptors->releaseUavBuffer(m_commands_slot);
    if (m_count_slot     != 0) m_descriptors->releaseUavBuffer(m_count_slot);
    if (m_surviving_slot != 0) m_descriptors->releaseUavBuffer(m_surviving_slot);

    if (m_buffer           != VK_NULL_HANDLE)
        vmaDestroyBuffer(m_allocator->handle(), m_buffer, m_alloc);
    if (m_count_buffer     != VK_NULL_HANDLE)
        vmaDestroyBuffer(m_allocator->handle(), m_count_buffer, m_count_alloc);
    if (m_surviving_buffer != VK_NULL_HANDLE)
        vmaDestroyBuffer(m_allocator->handle(), m_surviving_buffer, m_surviving_alloc);
}

void IndirectDrawBuffer::resize(size_t max_meshlets) {
    if (max_meshlets <= m_capacity) return;

    vkDeviceWaitIdle(m_device->logical());

    // Release old UAV slots.
    if (m_commands_slot  != 0) { m_descriptors->releaseUavBuffer(m_commands_slot);  m_commands_slot  = 0; }
    if (m_count_slot     != 0) { m_descriptors->releaseUavBuffer(m_count_slot);     m_count_slot     = 0; }
    if (m_surviving_slot != 0) { m_descriptors->releaseUavBuffer(m_surviving_slot); m_surviving_slot = 0; }

    // Destroy old buffers.
    if (m_buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_allocator->handle(), m_buffer, m_alloc);
        m_buffer = VK_NULL_HANDLE; m_alloc = nullptr;
    }
    if (m_count_buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_allocator->handle(), m_count_buffer, m_count_alloc);
        m_count_buffer = VK_NULL_HANDLE; m_count_alloc = nullptr;
    }
    if (m_surviving_buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_allocator->handle(), m_surviving_buffer, m_surviving_alloc);
        m_surviving_buffer = VK_NULL_HANDLE; m_surviving_alloc = nullptr;
    }

    // All three buffers need INDIRECT_BUFFER (for vkCmdDrawMeshTasksIndirectCountEXT),
    // STORAGE_BUFFER (for compute UAV), and TRANSFER_DST (for vkCmdFillBuffer reset).
    const VkBufferUsageFlags usage = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT
                                   | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                   | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    // Commands buffer: one DrawMeshTasksCommand per surviving meshlet.
    {
        const VkDeviceSize size = max_meshlets * sizeof(DrawMeshTasksCommand);
        VkBufferCreateInfo bufCI{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bufCI.size = size; bufCI.usage = usage; bufCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ENIGMA_VK_CHECK(vmaCreateBuffer(m_allocator->handle(), &bufCI, &allocCI,
                                        &m_buffer, &m_alloc, nullptr));
        m_commands_slot = m_descriptors->registerUavBuffer(m_buffer, size);
    }

    // Count buffer: single u32 atomic counter.
    {
        const VkDeviceSize size = sizeof(u32);
        VkBufferCreateInfo bufCI{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bufCI.size = size; bufCI.usage = usage; bufCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ENIGMA_VK_CHECK(vmaCreateBuffer(m_allocator->handle(), &bufCI, &allocCI,
                                        &m_count_buffer, &m_count_alloc, nullptr));
        m_count_slot = m_descriptors->registerUavBuffer(m_count_buffer, size);
    }

    // Surviving IDs buffer: two u32s per surviving meshlet — [globalMeshletId, instanceId].
    // Written by GpuCullPass, read by the task shader via survivingIdsSlot.
    // Stride is 8 bytes so instanceId is available without re-running findInstanceAndLocal.
    {
        const VkDeviceSize size = max_meshlets * sizeof(u32) * 2;
        VkBufferCreateInfo bufCI{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bufCI.size = size; bufCI.usage = usage; bufCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ENIGMA_VK_CHECK(vmaCreateBuffer(m_allocator->handle(), &bufCI, &allocCI,
                                        &m_surviving_buffer, &m_surviving_alloc, nullptr));
        m_surviving_slot = m_descriptors->registerUavBuffer(m_surviving_buffer, size);
    }

    m_capacity = max_meshlets;

    ENIGMA_LOG_INFO("[indirect_draw] resized to {} slots (cmd={} count={} surviving={})",
                    max_meshlets, m_commands_slot, m_count_slot, m_surviving_slot);
}

void IndirectDrawBuffer::reset_count(VkCommandBuffer cmd) {
    ENIGMA_ASSERT(m_count_buffer     != VK_NULL_HANDLE);
    ENIGMA_ASSERT(m_surviving_buffer != VK_NULL_HANDLE);

    // Zero both counters in a single transfer batch.
    vkCmdFillBuffer(cmd, m_count_buffer,     0, sizeof(u32), 0);
    vkCmdFillBuffer(cmd, m_surviving_buffer, 0, sizeof(u32), 0); // only first u32 used as count

    // Barrier: transfer write -> compute shader read/write.
    const VkBufferMemoryBarrier2 barriers[2] = {
        {
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2, nullptr,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            m_count_buffer, 0, sizeof(u32)
        },
        {
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2, nullptr,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            m_surviving_buffer, 0, VK_WHOLE_SIZE
        },
    };

    VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dep.bufferMemoryBarrierCount = 2;
    dep.pBufferMemoryBarriers    = barriers;
    vkCmdPipelineBarrier2(cmd, &dep);
}

} // namespace enigma
