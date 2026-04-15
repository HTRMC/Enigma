#include "renderer/GpuSceneBuffer.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "gfx/Allocator.h"
#include "gfx/DescriptorAllocator.h"
#include "gfx/Device.h"
#include "gfx/FrameContext.h"

#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#include <vk_mem_alloc.h>

#include <algorithm>
#include <cstring>

namespace enigma {

// Initial capacity: 256 instances.
static constexpr size_t kInitialCapacity = 256 * sizeof(GpuInstance);

GpuSceneBuffer::GpuSceneBuffer(gfx::Device& device, gfx::Allocator& allocator,
                               gfx::DescriptorAllocator& descriptors)
    : m_device(&device)
    , m_allocator(&allocator)
    , m_descriptors(&descriptors)
{
    ensure_capacity(kInitialCapacity);
    ENIGMA_LOG_INFO("[gpu_scene] created ({} bytes initial capacity, {} staging buffers)",
                    kInitialCapacity, gfx::MAX_FRAMES_IN_FLIGHT);
}

GpuSceneBuffer::~GpuSceneBuffer() {
    if (m_slot != 0) {
        m_descriptors->releaseStorageBuffer(m_slot);
    }
    for (u32 i = 0; i < gfx::MAX_FRAMES_IN_FLIGHT; ++i) {
        if (m_staging[i] != VK_NULL_HANDLE) {
            vmaDestroyBuffer(m_allocator->handle(), m_staging[i], m_staging_alloc[i]);
        }
    }
    if (m_gpu_buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_allocator->handle(), m_gpu_buffer, m_gpu_alloc);
    }
}

void GpuSceneBuffer::begin_frame() {
    m_instances.clear();
}

u32 GpuSceneBuffer::add_instance(const GpuInstance& inst) {
    const u32 index = static_cast<u32>(m_instances.size());
    m_instances.push_back(inst);
    return index;
}

void GpuSceneBuffer::upload(VkCommandBuffer cmd, u32 frameIndex) {
    ENIGMA_ASSERT(frameIndex < gfx::MAX_FRAMES_IN_FLIGHT);
    if (m_instances.empty()) return;

    const size_t required = m_instances.size() * sizeof(GpuInstance);
    ensure_capacity(required);

    // Write into this frame's staging buffer — safe because the GPU is at most
    // MAX_FRAMES_IN_FLIGHT-1 frames behind, so the previous use of this slot
    // has completed before we get back to it.
    void* mapped = nullptr;
    ENIGMA_VK_CHECK(vmaMapMemory(m_allocator->handle(), m_staging_alloc[frameIndex], &mapped));
    std::memcpy(mapped, m_instances.data(), required);
    vmaUnmapMemory(m_allocator->handle(), m_staging_alloc[frameIndex]);
    vmaFlushAllocation(m_allocator->handle(), m_staging_alloc[frameIndex], 0, required);

    // Copy staging -> GPU.
    VkBufferCopy region{};
    region.srcOffset = 0;
    region.dstOffset = 0;
    region.size      = required;
    vkCmdCopyBuffer(cmd, m_staging[frameIndex], m_gpu_buffer, 1, &region);

    // Barrier: transfer write -> shader read (compute cull + mesh shader).
    VkBufferMemoryBarrier2 barrier{};
    barrier.sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    barrier.srcStageMask  = VK_PIPELINE_STAGE_2_COPY_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barrier.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                          | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT
                          | VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    barrier.buffer        = m_gpu_buffer;
    barrier.offset        = 0;
    barrier.size          = required;

    VkDependencyInfo dep{};
    dep.sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.bufferMemoryBarrierCount = 1;
    dep.pBufferMemoryBarriers    = &barrier;
    vkCmdPipelineBarrier2(cmd, &dep);
}

void GpuSceneBuffer::ensure_capacity(size_t required) {
    if (required <= m_gpu_capacity) return;

    // Grow to at least double or the required size, whichever is larger.
    const size_t new_capacity = std::max(required, m_gpu_capacity * 2);

    // Wait for any in-flight GPU work before destroying old buffers.
    vkDeviceWaitIdle(m_device->logical());

    // Release old descriptor slot.
    if (m_slot != 0) {
        m_descriptors->releaseStorageBuffer(m_slot);
        m_slot = 0;
    }

    // Destroy old per-frame staging buffers.
    for (u32 i = 0; i < gfx::MAX_FRAMES_IN_FLIGHT; ++i) {
        if (m_staging[i] != VK_NULL_HANDLE) {
            vmaDestroyBuffer(m_allocator->handle(), m_staging[i], m_staging_alloc[i]);
            m_staging[i]       = VK_NULL_HANDLE;
            m_staging_alloc[i] = nullptr;
        }
    }

    // Destroy old GPU buffer.
    if (m_gpu_buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_allocator->handle(), m_gpu_buffer, m_gpu_alloc);
        m_gpu_buffer = VK_NULL_HANDLE;
        m_gpu_alloc  = nullptr;
    }

    // Create GPU SSBO.
    VkBufferCreateInfo gpuBufCI{};
    gpuBufCI.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    gpuBufCI.size        = new_capacity;
    gpuBufCI.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    gpuBufCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo gpuAllocCI{};
    gpuAllocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    ENIGMA_VK_CHECK(vmaCreateBuffer(m_allocator->handle(), &gpuBufCI, &gpuAllocCI,
                                    &m_gpu_buffer, &m_gpu_alloc, nullptr));

    // Create one staging buffer per frame.
    VkBufferCreateInfo stagingCI{};
    stagingCI.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingCI.size        = new_capacity;
    stagingCI.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagingCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo stagingAllocCI{};
    stagingAllocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    stagingAllocCI.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    for (u32 i = 0; i < gfx::MAX_FRAMES_IN_FLIGHT; ++i) {
        ENIGMA_VK_CHECK(vmaCreateBuffer(m_allocator->handle(), &stagingCI, &stagingAllocCI,
                                        &m_staging[i], &m_staging_alloc[i], nullptr));
    }

    m_gpu_capacity = new_capacity;

    // Register the new GPU buffer in the bindless descriptor set.
    m_slot = m_descriptors->registerStorageBuffer(
        m_gpu_buffer, static_cast<VkDeviceSize>(new_capacity));

    ENIGMA_LOG_INFO("[gpu_scene] resized to {} bytes (slot {}, {} staging buffers)",
                    new_capacity, m_slot, gfx::MAX_FRAMES_IN_FLIGHT);
}

} // namespace enigma
