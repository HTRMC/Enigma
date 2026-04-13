#pragma once

#include "core/Types.h"
#include "renderer/Meshlet.h"

#include <volk.h>

#include <vector>

// Forward declare VMA handles.
struct VmaAllocator_T;
using VmaAllocator = VmaAllocator_T*;
struct VmaAllocation_T;
using VmaAllocation = VmaAllocation_T*;

namespace enigma::gfx {
class Device;
class Allocator;
class DescriptorAllocator;
} // namespace enigma::gfx

namespace enigma {

// GpuMeshletBuffer
// ================
// Holds the global meshlet data arrays for all meshes on the GPU.
// Static after upload — call build() then upload() once when a scene loads.
//
// Three buffers:
//   meshlets        (Meshlet[])  — binding 2 (StructuredBuffer<float4>)
//   meshlet_vertices (u32[])     — binding 2 (StructuredBuffer<float4>)
//   meshlet_triangles (u8 packed) — binding 5 (RWByteAddressBuffer UAV)
//
// Each mesh's MeshletData is appended in order; the GpuInstance stores a
// meshlet_offset (into this global array) and meshlet_count so the
// GPU can locate any mesh's meshlets with a single base+index.
class GpuMeshletBuffer {
public:
    GpuMeshletBuffer(gfx::Device& device, gfx::Allocator& allocator,
                     gfx::DescriptorAllocator& descriptors);
    ~GpuMeshletBuffer();

    GpuMeshletBuffer(const GpuMeshletBuffer&)            = delete;
    GpuMeshletBuffer& operator=(const GpuMeshletBuffer&) = delete;

    // Accumulate meshlet data from one mesh. Returns the meshlet_offset
    // for this mesh (first meshlet index in the global buffer). Call for
    // each mesh before upload().
    u32 append(const MeshletData& data);

    // Upload all accumulated data to GPU via staging buffers.
    // Inserts pipeline barriers and destroys staging after transfer completes
    // (call vkDeviceWaitIdle or submit + wait before assuming data is resident).
    void upload(VkCommandBuffer cmd);

    // Destroy staging buffers after the upload command buffer has been submitted
    // and the GPU has finished. Call after the frame's fence signals.
    void flush_staging();

    // Total number of meshlets across all appended meshes.
    // Remains valid after upload() — CPU vector is not cleared.
    u32 total_meshlet_count() const { return static_cast<u32>(m_meshlets.size()); }

    // CPU-side meshlet array for VS fallback draw command generation.
    // Valid after append(), including after upload().
    const std::vector<Meshlet>& cpu_meshlets() const { return m_meshlets; }

    // Bindless slots for the GPU shader to access.
    u32 meshlets_slot()   const { return m_meshlets_slot;   }
    u32 vertices_slot()   const { return m_vertices_slot;   }
    u32 triangles_slot()  const { return m_triangles_slot;  }

private:
    void create_and_upload(VkCommandBuffer cmd,
                           const void*     data,
                           VkDeviceSize    size,
                           VkBufferUsageFlags extra_usage,
                           VkBuffer&       out_buffer,
                           VmaAllocation&  out_alloc,
                           VkBuffer&       out_staging,
                           VmaAllocation&  out_staging_alloc);

    gfx::Device*              m_device      = nullptr;
    gfx::Allocator*           m_allocator   = nullptr;
    gfx::DescriptorAllocator* m_descriptors = nullptr;

    // CPU-side accumulation (cleared after upload).
    std::vector<Meshlet> m_meshlets;
    std::vector<u32>     m_vertices;   // flat meshlet_vertices from all meshes
    std::vector<u8>      m_triangles;  // flat meshlet_triangles (packed u8)

    // GPU-side buffers.
    VkBuffer      m_meshlets_buf   = VK_NULL_HANDLE;
    VmaAllocation m_meshlets_alloc = nullptr;
    VkBuffer      m_vertices_buf   = VK_NULL_HANDLE;
    VmaAllocation m_vertices_alloc = nullptr;
    VkBuffer      m_triangles_buf  = VK_NULL_HANDLE;
    VmaAllocation m_triangles_alloc = nullptr;

    // Staging buffers (alive until flush_staging() is called post-submit).
    VkBuffer      m_staging_meshlets   = VK_NULL_HANDLE;
    VmaAllocation m_staging_meshlets_alloc = nullptr;
    VkBuffer      m_staging_vertices   = VK_NULL_HANDLE;
    VmaAllocation m_staging_vertices_alloc = nullptr;
    VkBuffer      m_staging_triangles  = VK_NULL_HANDLE;
    VmaAllocation m_staging_triangles_alloc = nullptr;

    u32 m_meshlets_slot  = 0;
    u32 m_vertices_slot  = 0;
    u32 m_triangles_slot = 0;
};

} // namespace enigma
