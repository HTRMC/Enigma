#include "renderer/GpuMeshletBuffer.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "gfx/Allocator.h"
#include "gfx/DescriptorAllocator.h"
#include "gfx/Device.h"

#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#include <vk_mem_alloc.h>

#include <algorithm>
#include <cstring>

namespace enigma {

// Staging ring-buffer ceiling for incremental meshlet appends. Budget is
// 16 patches/frame × 18 meshlets/patch — comfortably covers the default
// CdlodConfig::activationBudget with headroom for burst activations.
static constexpr u32 kStagingRingMeshletsPerSlot = 16u * 40u; // 16 activations/frame × 40 meshlets/patch (32×32 grid → 34 + headroom)

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

    // Incremental-append ring-buffer staging (CDLOD path).
    for (u32 i = 0; i < gfx::MAX_FRAMES_IN_FLIGHT; ++i) {
        if (m_stagingRingBuf[i] != VK_NULL_HANDLE) {
            vmaDestroyBuffer(m_allocator->handle(), m_stagingRingBuf[i], m_stagingRingAlloc[i]);
            m_stagingRingBuf[i]   = VK_NULL_HANDLE;
            m_stagingRingAlloc[i] = nullptr;
        }
    }

    // Per-LOD shared topology SSBOs.
    for (u32 slot : m_topoVertSlots) if (slot != 0) m_descriptors->releaseStorageBuffer(slot);
    for (u32 slot : m_topoTriSlots)  if (slot != 0) m_descriptors->releaseUavBuffer(slot);
    for (usize i = 0; i < m_topoVertBufs.size(); ++i) {
        if (m_topoVertBufs[i] != VK_NULL_HANDLE)
            vmaDestroyBuffer(m_allocator->handle(), m_topoVertBufs[i], m_topoVertAllocs[i]);
    }
    for (usize i = 0; i < m_topoTriBufs.size(); ++i) {
        if (m_topoTriBufs[i] != VK_NULL_HANDLE)
            vmaDestroyBuffer(m_allocator->handle(), m_topoTriBufs[i], m_topoTriAllocs[i]);
    }
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

    // Freeze the scene-only meshlet count so scene_meshlet_count() stays
    // correct even if appendIncremental() later grows m_meshlets.
    m_sceneMeshletCount = static_cast<u32>(m_meshlets.size());

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

    // Release the per-uploadSharedTopology staging buffers accumulated during
    // the one-shot upload cmd buffer (caller has now waited on the submit).
    for (usize i = 0; i < m_topoStagingBufs.size(); ++i) {
        if (m_topoStagingBufs[i] != VK_NULL_HANDLE) {
            vmaDestroyBuffer(m_allocator->handle(),
                             m_topoStagingBufs[i], m_topoStagingAllocs[i]);
        }
    }
    m_topoStagingBufs.clear();
    m_topoStagingAllocs.clear();
}

// ---------------------------------------------------------------------------
// CDLOD: pre-allocated meshlet buffer + per-LOD topology SSBOs +
// per-frame incremental append ring.
// ---------------------------------------------------------------------------

void GpuMeshletBuffer::reserveCapacity(VkCommandBuffer cmd, u32 totalMeshlets) {
    ENIGMA_ASSERT(m_meshlets_buf == VK_NULL_HANDLE &&
                  "GpuMeshletBuffer::reserveCapacity called twice or after upload()");
    ENIGMA_ASSERT(totalMeshlets > 0 && totalMeshlets >= m_meshlets.size() &&
                  "reserveCapacity must be >= current appended meshlet count");

    const VkDeviceSize bufferSize = static_cast<VkDeviceSize>(totalMeshlets) * sizeof(Meshlet);
    const VkDeviceSize initialBytes = m_meshlets.size() * sizeof(Meshlet);

    // Device-local meshlet buffer sized to full capacity.
    {
        VkBufferCreateInfo bufCI{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bufCI.size        = bufferSize;
        bufCI.usage       = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bufCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo allocCI{};
        allocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

        ENIGMA_VK_CHECK(vmaCreateBuffer(m_allocator->handle(), &bufCI, &allocCI,
                                        &m_meshlets_buf, &m_meshlets_alloc, nullptr));
    }

    // Seed the first initialBytes bytes with the scene-mesh Meshlet records
    // already appended by append(). Uses a one-time staging buffer (retained
    // until flush_staging() runs post-submit), matching upload() semantics.
    if (initialBytes > 0) {
        VkBufferCreateInfo stageCI{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        stageCI.size        = initialBytes;
        stageCI.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        stageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo allocCI{};
        allocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        allocCI.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT
                      | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

        VmaAllocationInfo info{};
        ENIGMA_VK_CHECK(vmaCreateBuffer(m_allocator->handle(), &stageCI, &allocCI,
                                        &m_staging_meshlets, &m_staging_meshlets_alloc, &info));
        std::memcpy(info.pMappedData, m_meshlets.data(), static_cast<size_t>(initialBytes));
        vmaFlushAllocation(m_allocator->handle(), m_staging_meshlets_alloc, 0, VK_WHOLE_SIZE);

        VkBufferCopy region{ 0, 0, initialBytes };
        vkCmdCopyBuffer(cmd, m_staging_meshlets, m_meshlets_buf, 1, &region);
    }

    // Upload vertices and triangles (static — built by append() calls before init).
    // reserveCapacity() replaces upload() in the CDLOD path, so it must also
    // seed the vertex-remapping and triangle-index buffers. Without this,
    // m_vertices_slot and m_triangles_slot stay at 0 and the mesh shader
    // reads vertex indices and triangle data from bindless slot 0 (a wrong buffer).
    const VkDeviceSize vertices_size  = m_vertices.size()  * sizeof(u32);
    const VkDeviceSize triangles_size = m_triangles.size() * sizeof(u8);

    if (vertices_size > 0) {
        create_and_upload(cmd, m_vertices.data(), vertices_size,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          m_vertices_buf, m_vertices_alloc,
                          m_staging_vertices, m_staging_vertices_alloc);
    }
    if (triangles_size > 0) {
        create_and_upload(cmd, m_triangles.data(), triangles_size,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          m_triangles_buf, m_triangles_alloc,
                          m_staging_triangles, m_staging_triangles_alloc);
    }

    // Combined barrier: transfer writes → shader reads for all seeded buffers.
    {
        VkBufferMemoryBarrier2 barriers[3];
        u32 n = 0;
        auto makeBarrier = [](VkBuffer buf) -> VkBufferMemoryBarrier2 {
            VkBufferMemoryBarrier2 b{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2 };
            b.srcStageMask        = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            b.srcAccessMask       = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            b.dstStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                                  | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT
                                  | VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT;
            b.dstAccessMask       = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.buffer              = buf;
            b.offset              = 0;
            b.size                = VK_WHOLE_SIZE;
            return b;
        };
        if (initialBytes   > 0) barriers[n++] = makeBarrier(m_meshlets_buf);
        if (vertices_size  > 0) barriers[n++] = makeBarrier(m_vertices_buf);
        if (triangles_size > 0) barriers[n++] = makeBarrier(m_triangles_buf);
        if (n > 0) {
            VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            dep.bufferMemoryBarrierCount = n;
            dep.pBufferMemoryBarriers    = barriers;
            vkCmdPipelineBarrier2(cmd, &dep);
        }
    }

    // Register bindless slots — stable for program lifetime.
    m_meshlets_slot  = m_descriptors->registerStorageBuffer(m_meshlets_buf, bufferSize);
    if (m_vertices_buf  != VK_NULL_HANDLE)
        m_vertices_slot  = m_descriptors->registerStorageBuffer(m_vertices_buf,  vertices_size);
    if (m_triangles_buf != VK_NULL_HANDLE)
        m_triangles_slot = m_descriptors->registerUavBuffer(m_triangles_buf, triangles_size);
    m_reservedCapacity  = totalMeshlets;
    // Freeze the scene-only count before any terrain appendIncremental() calls.
    m_sceneMeshletCount = static_cast<u32>(m_meshlets.size());

    // Pre-allocate the per-frame staging ring. Each slot is sized to the
    // conservative ceiling (16 patches × 18 meshlets per patch).
    const VkDeviceSize slotSize =
        static_cast<VkDeviceSize>(kStagingRingMeshletsPerSlot) * sizeof(Meshlet);
    m_stagingRingSize = static_cast<u32>(slotSize);

    VkBufferCreateInfo ringCI{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    ringCI.size        = slotSize;
    ringCI.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    ringCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo ringAllocCI{};
    ringAllocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    ringAllocCI.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT
                      | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    for (u32 i = 0; i < gfx::MAX_FRAMES_IN_FLIGHT; ++i) {
        ENIGMA_VK_CHECK(vmaCreateBuffer(m_allocator->handle(), &ringCI, &ringAllocCI,
                                        &m_stagingRingBuf[i], &m_stagingRingAlloc[i], nullptr));
    }

    ENIGMA_LOG_INFO("[meshlet_buffer] reserved capacity for {} meshlets ({} bytes), "
                    "slots: meshlets={} verts={} tris={}, staging ring {}x{} bytes",
                    totalMeshlets, static_cast<u64>(bufferSize),
                    m_meshlets_slot, m_vertices_slot, m_triangles_slot,
                    gfx::MAX_FRAMES_IN_FLIGHT, static_cast<u64>(slotSize));
}

GpuMeshletBuffer::SharedTopologyHandle GpuMeshletBuffer::uploadSharedTopology(
    VkCommandBuffer cmd,
    const std::vector<u32>& vertexIndices,
    const std::vector<u8>&  triangleBytes)
{
    ENIGMA_ASSERT(!vertexIndices.empty() && !triangleBytes.empty());

    auto makeStaging = [&](const void* data, VkDeviceSize size,
                           VkBuffer& outBuf, VmaAllocation& outAlloc) {
        VkBufferCreateInfo stageCI{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        stageCI.size        = size;
        stageCI.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        stageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo allocCI{};
        allocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        allocCI.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT
                      | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

        VmaAllocationInfo info{};
        ENIGMA_VK_CHECK(vmaCreateBuffer(m_allocator->handle(), &stageCI, &allocCI,
                                        &outBuf, &outAlloc, &info));
        std::memcpy(info.pMappedData, data, static_cast<size_t>(size));
        vmaFlushAllocation(m_allocator->handle(), outAlloc, 0, VK_WHOLE_SIZE);
    };

    auto makeDevice = [&](VkDeviceSize size, VkBuffer& outBuf, VmaAllocation& outAlloc) {
        VkBufferCreateInfo bufCI{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bufCI.size        = size;
        bufCI.usage       = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bufCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo allocCI{};
        allocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

        ENIGMA_VK_CHECK(vmaCreateBuffer(m_allocator->handle(), &bufCI, &allocCI,
                                        &outBuf, &outAlloc, nullptr));
    };

    // Vertex-index SSBO (binding 2 storage buffer).
    const VkDeviceSize vertSize = vertexIndices.size() * sizeof(u32);
    VkBuffer vertBuf = VK_NULL_HANDLE;
    VmaAllocation vertAlloc = nullptr;
    VkBuffer vertStaging = VK_NULL_HANDLE;
    VmaAllocation vertStagingAlloc = nullptr;
    makeDevice(vertSize, vertBuf, vertAlloc);
    makeStaging(vertexIndices.data(), vertSize, vertStaging, vertStagingAlloc);

    // Triangle-byte SSBO (binding 5 UAV RWByteAddressBuffer).
    const VkDeviceSize triSize = triangleBytes.size() * sizeof(u8);
    VkBuffer triBuf = VK_NULL_HANDLE;
    VmaAllocation triAlloc = nullptr;
    VkBuffer triStaging = VK_NULL_HANDLE;
    VmaAllocation triStagingAlloc = nullptr;
    makeDevice(triSize, triBuf, triAlloc);
    makeStaging(triangleBytes.data(), triSize, triStaging, triStagingAlloc);

    // Copy staging -> device.
    {
        VkBufferCopy region{ 0, 0, vertSize };
        vkCmdCopyBuffer(cmd, vertStaging, vertBuf, 1, &region);
    }
    {
        VkBufferCopy region{ 0, 0, triSize };
        vkCmdCopyBuffer(cmd, triStaging, triBuf, 1, &region);
    }

    // Barrier: transfer write -> shader read.
    VkBufferMemoryBarrier2 barriers[2] = {
        {
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2, nullptr,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT
            | VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            vertBuf, 0, VK_WHOLE_SIZE
        },
        {
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2, nullptr,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT
            | VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            triBuf, 0, VK_WHOLE_SIZE
        },
    };
    VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dep.bufferMemoryBarrierCount = 2;
    dep.pBufferMemoryBarriers    = barriers;
    vkCmdPipelineBarrier2(cmd, &dep);

    // Register bindless slots.
    const u32 vertSlot = m_descriptors->registerStorageBuffer(vertBuf, vertSize);
    const u32 triSlot  = m_descriptors->registerUavBuffer(triBuf,  triSize);

    // Retain device buffers for program lifetime; stash staging so we can
    // destroy them on the same flush as the legacy upload() stagings.
    m_topoVertBufs.push_back(vertBuf);
    m_topoVertAllocs.push_back(vertAlloc);
    m_topoTriBufs.push_back(triBuf);
    m_topoTriAllocs.push_back(triAlloc);
    m_topoVertSlots.push_back(vertSlot);
    m_topoTriSlots.push_back(triSlot);

    // Defer staging destruction until flush_staging() — the caller submits a
    // single one-shot command buffer that records many uploadSharedTopology()
    // copies; destroying the staging inline would race with the pending copy.
    m_topoStagingBufs.push_back(vertStaging);
    m_topoStagingAllocs.push_back(vertStagingAlloc);
    m_topoStagingBufs.push_back(triStaging);
    m_topoStagingAllocs.push_back(triStagingAlloc);

    SharedTopologyHandle handle{};
    handle.topologyVerticesSlot  = vertSlot;
    handle.topologyTrianglesSlot = triSlot;
    handle.vertexCount           = static_cast<u32>(vertexIndices.size());
    handle.triangleCount         = static_cast<u32>(triangleBytes.size());
    // meshletCount is set by the caller (it knows the LOD template meshlet count).
    return handle;
}

void GpuMeshletBuffer::freeMeshletRange(MeshletRange range) {
    if (range.count == 0) return;
    m_freeMeshletRanges.push_back(range);
}

void GpuMeshletBuffer::beginFrame(u32 /*frameIndex*/) {
    // Reset the staging cursor so this frame's first appendIncremental writes
    // to byte 0 of the ring slot. Safe to reset unconditionally each frame —
    // the GPU has already consumed MAX_FRAMES_IN_FLIGHT-old ring data by the
    // time this frame begins recording (frame fence gate upstream).
    m_stagingCursor = 0;
}

u32 GpuMeshletBuffer::appendIncremental(VkCommandBuffer cmd, u32 frameIndex,
                                        const std::vector<Meshlet>& patchMeshlets)
{
    ENIGMA_ASSERT(m_meshlets_buf != VK_NULL_HANDLE &&
                  "appendIncremental called before reserveCapacity");
    ENIGMA_ASSERT(m_reservedCapacity > 0 && "appendIncremental requires reserveCapacity()");

    if (patchMeshlets.empty()) {
        return static_cast<u32>(m_meshlets.size());
    }

    const VkDeviceSize bytes = patchMeshlets.size() * sizeof(Meshlet);
    ENIGMA_ASSERT(m_stagingCursor + bytes <= m_stagingRingSize &&
                  "appendIncremental: staging ring slot full — "
                  "raise kStagingRingMeshletsPerSlot or cap activations/frame");

    // Try to reuse a retired range of identical size before growing the end.
    u32 offset = UINT32_MAX;
    const u32 wanted = static_cast<u32>(patchMeshlets.size());
    for (auto it = m_freeMeshletRanges.begin(); it != m_freeMeshletRanges.end(); ++it) {
        if (it->count == wanted) {
            offset = it->offset;
            m_freeMeshletRanges.erase(it);
            break;
        }
    }

    const bool appendedEnd = (offset == UINT32_MAX);
    if (appendedEnd) {
        ENIGMA_ASSERT(m_meshlets.size() + patchMeshlets.size() <= m_reservedCapacity &&
                      "appendIncremental overflows reserved capacity");
        offset = static_cast<u32>(m_meshlets.size());
    }

    const u32 slot = frameIndex % gfx::MAX_FRAMES_IN_FLIGHT;
    const u32 srcOffset = m_stagingCursor;

    // Upload into this frame's staging slot at the current cursor position.
    // Multiple activations per frame accumulate side-by-side in the slot; the
    // cursor advances after each copy so subsequent calls don't clobber the
    // bytes referenced by previously-recorded vkCmdCopyBuffer(s).
    void* mapped = nullptr;
    ENIGMA_VK_CHECK(vmaMapMemory(m_allocator->handle(), m_stagingRingAlloc[slot], &mapped));
    std::memcpy(static_cast<u8*>(mapped) + srcOffset,
                patchMeshlets.data(), static_cast<size_t>(bytes));
    vmaUnmapMemory(m_allocator->handle(), m_stagingRingAlloc[slot]);
    vmaFlushAllocation(m_allocator->handle(), m_stagingRingAlloc[slot], srcOffset, bytes);

    // Record copy staging[slot][srcOffset..] -> GPU meshlet buffer[offset..].
    VkBufferCopy region{};
    region.srcOffset = srcOffset;
    region.dstOffset = static_cast<VkDeviceSize>(offset) * sizeof(Meshlet);
    region.size      = bytes;
    vkCmdCopyBuffer(cmd, m_stagingRingBuf[slot], m_meshlets_buf, 1, &region);

    m_stagingCursor += static_cast<u32>(bytes);

    // Mirror to the CPU vector. For an end-append, extend; for a reused range,
    // overwrite the existing slots so total_meshlet_count() stays correct.
    if (appendedEnd) {
        m_meshlets.insert(m_meshlets.end(), patchMeshlets.begin(), patchMeshlets.end());
    } else {
        std::memcpy(m_meshlets.data() + offset, patchMeshlets.data(),
                    static_cast<size_t>(bytes));
    }

    return offset;
}

} // namespace enigma
