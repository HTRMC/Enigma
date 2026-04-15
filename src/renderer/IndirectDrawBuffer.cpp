#include "renderer/IndirectDrawBuffer.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "gfx/Allocator.h"
#include "gfx/DescriptorAllocator.h"
#include "gfx/Device.h"

#include <array>
#include <cstring>

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
    if (m_readback_buffer  != VK_NULL_HANDLE)
        vmaDestroyBuffer(m_allocator->handle(), m_readback_buffer, m_readback_alloc);
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
    if (m_readback_buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_allocator->handle(), m_readback_buffer, m_readback_alloc);
        m_readback_buffer = VK_NULL_HANDLE; m_readback_alloc = nullptr;
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

    // Count buffer: u32[0] = surviving meshlet count; u32[1..5] = per-plane cull
    // counters for DIAG_PER_PLANE_CULL in gpu_cull.comp.hlsl (offsets 4,8,12,16,20).
    // 6 u32s total = 24 bytes. Extra slots are unused when diagnostics are off.
    {
        const VkDeviceSize size = sizeof(u32) * 6;
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

    // Readback buffer: HOST_VISIBLE copy of count_buffer (24 bytes).
    // Used by log_diag_plane_counts() to read per-plane cull counters on the CPU.
    {
        const VkDeviceSize size = sizeof(u32) * 6;
        VkBufferCreateInfo bufCI{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bufCI.size        = size;
        bufCI.usage       = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo rbAllocCI{};
        rbAllocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        rbAllocCI.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
        ENIGMA_VK_CHECK(vmaCreateBuffer(m_allocator->handle(), &bufCI, &rbAllocCI,
                                        &m_readback_buffer, &m_readback_alloc, nullptr));
    }

    m_capacity = max_meshlets;

    ENIGMA_LOG_INFO("[indirect_draw] resized to {} slots (cmd={} count={} surviving={})",
                    max_meshlets, m_commands_slot, m_count_slot, m_surviving_slot);
}

void IndirectDrawBuffer::reset_count(VkCommandBuffer cmd) {
    ENIGMA_ASSERT(m_count_buffer     != VK_NULL_HANDLE);
    ENIGMA_ASSERT(m_surviving_buffer != VK_NULL_HANDLE);

    // Zero the count buffer (all 6 u32s: surviving count + 5 per-plane diagnostic counters)
    // and the first u32 of the surviving buffer (unused sentinel).
    vkCmdFillBuffer(cmd, m_count_buffer,     0, sizeof(u32) * 6, 0);
    vkCmdFillBuffer(cmd, m_surviving_buffer, 0, sizeof(u32),     0);

    // Barrier: transfer write -> compute shader read/write.
    const VkBufferMemoryBarrier2 barriers[2] = {
        {
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2, nullptr,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            m_count_buffer, 0, sizeof(u32) * 6
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

void IndirectDrawBuffer::record_count_readback(VkCommandBuffer cmd) {
    ENIGMA_ASSERT(m_count_buffer    != VK_NULL_HANDLE);
    ENIGMA_ASSERT(m_readback_buffer != VK_NULL_HANDLE);

    // Barrier: compute write -> transfer read.
    const VkBufferMemoryBarrier2 barrier{
        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2, nullptr,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COPY_BIT,           VK_ACCESS_2_TRANSFER_READ_BIT,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        m_count_buffer, 0, sizeof(u32) * 6
    };
    VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dep.bufferMemoryBarrierCount = 1;
    dep.pBufferMemoryBarriers    = &barrier;
    vkCmdPipelineBarrier2(cmd, &dep);

    VkBufferCopy region{};
    region.srcOffset = 0;
    region.dstOffset = 0;
    region.size      = sizeof(u32) * 6;
    vkCmdCopyBuffer(cmd, m_count_buffer, m_readback_buffer, 1, &region);
}

void IndirectDrawBuffer::log_diag_plane_counts() {
    ENIGMA_ASSERT(m_readback_buffer != VK_NULL_HANDLE);

    // Invalidate before reading on non-coherent allocations.
    vmaInvalidateAllocation(m_allocator->handle(), m_readback_alloc, 0, sizeof(u32) * 6);

    void* mapped = nullptr;
    ENIGMA_VK_CHECK(vmaMapMemory(m_allocator->handle(), m_readback_alloc, &mapped));
    std::array<u32, 6> counts{};
    std::memcpy(counts.data(), mapped, sizeof(u32) * 6);
    vmaUnmapMemory(m_allocator->handle(), m_readback_alloc);

    ENIGMA_LOG_INFO("[cull_diag] surviving={} plane[L]={} plane[R]={} plane[B]={} plane[T]={} plane[N]={}",
                    counts[0], counts[1], counts[2], counts[3], counts[4], counts[5]);
}

} // namespace enigma
