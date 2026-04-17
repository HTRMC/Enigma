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

// Initial reverse-lookup capacity: 4096 u32s = 16 KB. Grows via
// ensure_lookup_capacity() as meshlet-id high-water expands.
static constexpr size_t kInitialLookupCapacity = 4096 * sizeof(u32);

GpuSceneBuffer::GpuSceneBuffer(gfx::Device& device, gfx::Allocator& allocator,
                               gfx::DescriptorAllocator& descriptors)
    : m_device(&device)
    , m_allocator(&allocator)
    , m_descriptors(&descriptors)
{
    ensure_capacity(kInitialCapacity);
    ensure_lookup_capacity(kInitialLookupCapacity);
    ENIGMA_LOG_INFO("[gpu_scene] created ({} bytes instance capacity, {} bytes lookup capacity, {} staging buffers)",
                    kInitialCapacity, kInitialLookupCapacity, gfx::MAX_FRAMES_IN_FLIGHT);
}

GpuSceneBuffer::~GpuSceneBuffer() {
    if (m_slot != 0) {
        m_descriptors->releaseStorageBuffer(m_slot);
    }
    if (m_lookup_slot != 0) {
        m_descriptors->releaseStorageBuffer(m_lookup_slot);
    }
    for (u32 i = 0; i < gfx::MAX_FRAMES_IN_FLIGHT; ++i) {
        if (m_staging[i] != VK_NULL_HANDLE) {
            vmaDestroyBuffer(m_allocator->handle(), m_staging[i], m_staging_alloc[i]);
        }
        if (m_lookup_staging[i] != VK_NULL_HANDLE) {
            vmaDestroyBuffer(m_allocator->handle(), m_lookup_staging[i], m_lookup_staging_alloc[i]);
        }
    }
    if (m_gpu_buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_allocator->handle(), m_gpu_buffer, m_gpu_alloc);
    }
    if (m_lookup_gpu_buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_allocator->handle(), m_lookup_gpu_buffer, m_lookup_gpu_alloc);
    }
}

void GpuSceneBuffer::begin_frame() {
    m_instances.clear();
    // Reset all lookup entries to "orphaned". Retain vector size (high-water)
    // so we avoid reallocating storage each frame. add_instance() fills the
    // owned ranges; any id not claimed stays 0xFFFFFFFFu.
    std::fill(m_meshletToInstance.begin(), m_meshletToInstance.end(), 0xFFFFFFFFu);
}

u32 GpuSceneBuffer::add_instance(const GpuInstance& inst) {
    const u32 index = static_cast<u32>(m_instances.size());
    m_instances.push_back(inst);

    // Populate the meshlet→instance reverse lookup for this instance's range.
    const u32 end = inst.meshlet_offset + inst.meshlet_count;
    if (end > m_meshletToInstance.size()) {
        m_meshletToInstance.resize(end, 0xFFFFFFFFu);
    }
    for (u32 i = inst.meshlet_offset; i < end; ++i) {
        m_meshletToInstance[i] = index;
    }
    return index;
}

void GpuSceneBuffer::upload(VkCommandBuffer cmd, u32 frameIndex, u32 meshletLookupCount) {
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

    // Grow the lookup vector to cover every meshlet id the cull pass might
    // query (retired terrain ranges can extend past the add_instance() claims).
    // Newly grown entries default to 0xFFFFFFFFu (orphaned).
    if (meshletLookupCount > m_meshletToInstance.size()) {
        m_meshletToInstance.resize(meshletLookupCount, 0xFFFFFFFFu);
    }

    const size_t lookupBytes = m_meshletToInstance.size() * sizeof(u32);
    ensure_lookup_capacity(lookupBytes);

    void* lookupMapped = nullptr;
    ENIGMA_VK_CHECK(vmaMapMemory(m_allocator->handle(), m_lookup_staging_alloc[frameIndex], &lookupMapped));
    std::memcpy(lookupMapped, m_meshletToInstance.data(), lookupBytes);
    vmaUnmapMemory(m_allocator->handle(), m_lookup_staging_alloc[frameIndex]);
    vmaFlushAllocation(m_allocator->handle(), m_lookup_staging_alloc[frameIndex], 0, lookupBytes);

    // Copy staging -> GPU for both instance and lookup buffers.
    VkBufferCopy region{};
    region.srcOffset = 0;
    region.dstOffset = 0;
    region.size      = required;
    vkCmdCopyBuffer(cmd, m_staging[frameIndex], m_gpu_buffer, 1, &region);

    VkBufferCopy lookupRegion{};
    lookupRegion.srcOffset = 0;
    lookupRegion.dstOffset = 0;
    lookupRegion.size      = lookupBytes;
    vkCmdCopyBuffer(cmd, m_lookup_staging[frameIndex], m_lookup_gpu_buffer, 1, &lookupRegion);

    // Barrier: transfer write -> shader read for both buffers.
    const VkBufferMemoryBarrier2 barriers[2] = {
        {
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2, nullptr,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
            | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT
            | VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            m_gpu_buffer, 0, required
        },
        {
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2, nullptr,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
            | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT
            | VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            m_lookup_gpu_buffer, 0, lookupBytes
        },
    };

    VkDependencyInfo dep{};
    dep.sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.bufferMemoryBarrierCount = 2;
    dep.pBufferMemoryBarriers    = barriers;
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

void GpuSceneBuffer::ensure_lookup_capacity(size_t required_bytes) {
    if (required_bytes <= m_lookup_gpu_capacity) return;

    const size_t new_capacity = std::max(required_bytes, m_lookup_gpu_capacity * 2);

    vkDeviceWaitIdle(m_device->logical());

    if (m_lookup_slot != 0) {
        m_descriptors->releaseStorageBuffer(m_lookup_slot);
        m_lookup_slot = 0;
    }

    for (u32 i = 0; i < gfx::MAX_FRAMES_IN_FLIGHT; ++i) {
        if (m_lookup_staging[i] != VK_NULL_HANDLE) {
            vmaDestroyBuffer(m_allocator->handle(), m_lookup_staging[i], m_lookup_staging_alloc[i]);
            m_lookup_staging[i]       = VK_NULL_HANDLE;
            m_lookup_staging_alloc[i] = nullptr;
        }
    }
    if (m_lookup_gpu_buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_allocator->handle(), m_lookup_gpu_buffer, m_lookup_gpu_alloc);
        m_lookup_gpu_buffer = VK_NULL_HANDLE;
        m_lookup_gpu_alloc  = nullptr;
    }

    VkBufferCreateInfo gpuBufCI{};
    gpuBufCI.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    gpuBufCI.size        = new_capacity;
    gpuBufCI.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    gpuBufCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo gpuAllocCI{};
    gpuAllocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    ENIGMA_VK_CHECK(vmaCreateBuffer(m_allocator->handle(), &gpuBufCI, &gpuAllocCI,
                                    &m_lookup_gpu_buffer, &m_lookup_gpu_alloc, nullptr));

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
                                        &m_lookup_staging[i], &m_lookup_staging_alloc[i], nullptr));
    }

    m_lookup_gpu_capacity = new_capacity;
    m_lookup_slot = m_descriptors->registerStorageBuffer(
        m_lookup_gpu_buffer, static_cast<VkDeviceSize>(new_capacity));

    ENIGMA_LOG_INFO("[gpu_scene] lookup resized to {} bytes (slot {})",
                    new_capacity, m_lookup_slot);
}

} // namespace enigma
