#pragma once

#include "core/Math.h"
#include "core/Types.h"

#include <volk.h>

#include <vector>

// Forward declare VMA handle — full header included only in the .cpp.
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

// Per-instance GPU data. Must match GpuInstance in gpu_cull.comp.hlsl and
// visibility_buffer.mesh.hlsl.
struct alignas(16) GpuInstance {
    mat4 transform;          // world matrix (column-major, GLM default)
    u32  meshlet_offset;     // index into global meshlet buffer where this mesh starts
    u32  meshlet_count;      // number of meshlets in this mesh
    u32  material_index;     // bindless material index
    u32  vertex_buffer_slot; // bindless vertex SSBO slot
};

// CPU-side builder — filled from ECS query or Scene, then uploaded to GPU.
class GpuSceneBuffer {
public:
    GpuSceneBuffer(gfx::Device& device, gfx::Allocator& allocator,
                   gfx::DescriptorAllocator& descriptors);
    ~GpuSceneBuffer();

    GpuSceneBuffer(const GpuSceneBuffer&)            = delete;
    GpuSceneBuffer& operator=(const GpuSceneBuffer&) = delete;

    // Reset per-frame. Call at the start of each frame before adding instances.
    void begin_frame();

    // Add an instance. Returns the instance index.
    u32 add_instance(const GpuInstance& inst);

    // Upload all added instances to the GPU buffer.
    // Must be called after all add_instance() calls and before draw.
    void upload(VkCommandBuffer cmd);

    // Bindless slot of the uploaded SSBO (registered in DescriptorAllocator).
    u32 slot() const { return m_slot; }

    // Number of instances this frame.
    size_t instance_count() const { return m_instances.size(); }

    // CPU-side instance array for VS fallback draw command generation.
    const std::vector<GpuInstance>& cpu_instances() const { return m_instances; }

private:
    void ensure_capacity(size_t required);

    gfx::Device*              m_device      = nullptr;
    gfx::Allocator*           m_allocator   = nullptr;
    gfx::DescriptorAllocator* m_descriptors = nullptr;

    std::vector<GpuInstance> m_instances;

    // GPU-side SSBO + staging buffer.
    VkBuffer      m_gpu_buffer    = VK_NULL_HANDLE;
    VmaAllocation m_gpu_alloc     = nullptr;
    VkBuffer      m_staging       = VK_NULL_HANDLE;
    VmaAllocation m_staging_alloc = nullptr;
    size_t        m_gpu_capacity  = 0; // in bytes

    u32 m_slot = 0; // bindless SSBO slot
};

} // namespace enigma
