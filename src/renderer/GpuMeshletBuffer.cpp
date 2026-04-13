#include "renderer/GpuMeshletBuffer.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "gfx/Allocator.h"
#include "gfx/DescriptorAllocator.h"
#include "gfx/Device.h"

#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#include <vk_mem_alloc.h>

#include <cstring>

namespace enigma {

GpuMeshletBuffer::GpuMeshletBuffer(gfx::Device& device, gfx::Allocator& allocator,
                                   gfx::DescriptorAllocator& descriptors)
    : m_device(&device)
    , m_allocator(&allocator)
    , m_descriptors(&descriptors)
{}

GpuMeshletBuffer::~GpuMeshletBuffer() {
    flush_staging();

    if (m_meshlets_slot  != 0) m_descriptors->releaseStorageBuffer(m_meshlets_slot);
    if (m_vertices_slot  != 0) m_descriptors->releaseStorageBuffer(m_vertices_slot);
    if (m_triangles_slot != 0) m_descriptors->releaseUavBuffer(m_triangles_slot);

    if (m_meshlets_buf   != VK_NULL_HANDLE) vmaDestroyBuffer(m_allocator->handle(), m_meshlets_buf,  m_meshlets_alloc);
    if (m_vertices_buf   != VK_NULL_HANDLE) vmaDestroyBuffer(m_allocator->handle(), m_vertices_buf,  m_vertices_alloc);
    if (m_triangles_buf  != VK_NULL_HANDLE) vmaDestroyBuffer(m_allocator->handle(), m_triangles_buf, m_triangles_alloc);
}

u32 GpuMeshletBuffer::append(const MeshletData& data) {
    const u32 offset = static_cast<u32>(m_meshlets.size());

    // Append meshlets, adjusting offsets to be relative to the global arrays.
    const u32 vertex_base   = static_cast<u32>(m_vertices.size());
    const u32 triangle_base = static_cast<u32>(m_triangles.size());

    for (const Meshlet& src : data.meshlets) {
        Meshlet m = src;
        m.vertex_offset   += vertex_base;
        m.triangle_offset += triangle_base;
        m_meshlets.push_back(m);
    }

    m_vertices.insert(m_vertices.end(),
                      data.meshlet_vertices.begin(),
                      data.meshlet_vertices.end());
    m_triangles.insert(m_triangles.end(),
                       data.meshlet_triangles.begin(),
                       data.meshlet_triangles.end());

    return offset;
}

void GpuMeshletBuffer::create_and_upload(VkCommandBuffer    cmd,
                                         const void*        data,
                                         VkDeviceSize       size,
                                         VkBufferUsageFlags extra_usage,
                                         VkBuffer&          out_buffer,
                                         VmaAllocation&     out_alloc,
                                         VkBuffer&          out_staging,
                                         VmaAllocation&     out_staging_alloc) {
    // Staging buffer (CPU-visible).
    {
        VkBufferCreateInfo bufCI{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bufCI.size        = size;
        bufCI.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bufCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo allocCI{};
        allocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        allocCI.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT
                      | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

        VmaAllocationInfo info{};
        ENIGMA_VK_CHECK(vmaCreateBuffer(m_allocator->handle(), &bufCI, &allocCI,
                                        &out_staging, &out_staging_alloc, &info));
        std::memcpy(info.pMappedData, data, static_cast<size_t>(size));
        vmaFlushAllocation(m_allocator->handle(), out_staging_alloc, 0, VK_WHOLE_SIZE);
    }

    // Device-local buffer.
    {
        VkBufferCreateInfo bufCI{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bufCI.size        = size;
        bufCI.usage       = VK_BUFFER_USAGE_TRANSFER_DST_BIT | extra_usage;
        bufCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo allocCI{};
        allocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

        ENIGMA_VK_CHECK(vmaCreateBuffer(m_allocator->handle(), &bufCI, &allocCI,
                                        &out_buffer, &out_alloc, nullptr));
    }

    VkBufferCopy region{ 0, 0, size };
    vkCmdCopyBuffer(cmd, out_staging, out_buffer, 1, &region);
}

void GpuMeshletBuffer::upload(VkCommandBuffer cmd) {
    ENIGMA_ASSERT(!m_meshlets.empty() && "GpuMeshletBuffer::upload called with no data");

    // ------------------------------------------------------------------
    // Create device-local buffers and record copy commands.
    // ------------------------------------------------------------------
    const VkDeviceSize meshlets_size  = m_meshlets.size()  * sizeof(Meshlet);
    const VkDeviceSize vertices_size  = m_vertices.size()  * sizeof(u32);
    const VkDeviceSize triangles_size = m_triangles.size() * sizeof(u8);

    // meshlets and vertices → binding 2 (STORAGE_BUFFER, StructuredBuffer<float4>)
    create_and_upload(cmd, m_meshlets.data(), meshlets_size,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      m_meshlets_buf, m_meshlets_alloc,
                      m_staging_meshlets, m_staging_meshlets_alloc);

    create_and_upload(cmd, m_vertices.data(), vertices_size,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      m_vertices_buf, m_vertices_alloc,
                      m_staging_vertices, m_staging_vertices_alloc);

    // triangles → binding 5 (STORAGE_BUFFER used as RWByteAddressBuffer UAV)
    create_and_upload(cmd, m_triangles.data(), triangles_size,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      m_triangles_buf, m_triangles_alloc,
                      m_staging_triangles, m_staging_triangles_alloc);

    // ------------------------------------------------------------------
    // Barrier: transfer write → compute/graphics shader read.
    // ------------------------------------------------------------------
    const VkBufferMemoryBarrier2 barriers[3] = {
        {
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2, nullptr,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT
            | VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            m_meshlets_buf, 0, VK_WHOLE_SIZE
        },
        {
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2, nullptr,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT
            | VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            m_vertices_buf, 0, VK_WHOLE_SIZE
        },
        {
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2, nullptr,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            m_triangles_buf, 0, VK_WHOLE_SIZE
        },
    };

    VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dep.bufferMemoryBarrierCount = 3;
    dep.pBufferMemoryBarriers    = barriers;
    vkCmdPipelineBarrier2(cmd, &dep);

    // Register bindless descriptors.
    m_meshlets_slot  = m_descriptors->registerStorageBuffer(m_meshlets_buf,  meshlets_size);
    m_vertices_slot  = m_descriptors->registerStorageBuffer(m_vertices_buf,  vertices_size);
    m_triangles_slot = m_descriptors->registerUavBuffer(m_triangles_buf, triangles_size);

    ENIGMA_LOG_INFO("[meshlet_buffer] uploaded {} meshlets / {} verts / {} tri-bytes "
                    "(slots: meshlets={} verts={} tris={})",
                    m_meshlets.size(), m_vertices.size(), m_triangles.size(),
                    m_meshlets_slot, m_vertices_slot, m_triangles_slot);
}

void GpuMeshletBuffer::flush_staging() {
    if (m_staging_meshlets != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_allocator->handle(), m_staging_meshlets, m_staging_meshlets_alloc);
        m_staging_meshlets = VK_NULL_HANDLE; m_staging_meshlets_alloc = nullptr;
    }
    if (m_staging_vertices != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_allocator->handle(), m_staging_vertices, m_staging_vertices_alloc);
        m_staging_vertices = VK_NULL_HANDLE; m_staging_vertices_alloc = nullptr;
    }
    if (m_staging_triangles != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_allocator->handle(), m_staging_triangles, m_staging_triangles_alloc);
        m_staging_triangles = VK_NULL_HANDLE; m_staging_triangles_alloc = nullptr;
    }
}

} // namespace enigma
