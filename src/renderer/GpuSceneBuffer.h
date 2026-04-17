#pragma once

#include "core/Math.h"
#include "core/Types.h"
#include "gfx/FrameContext.h"

#include <volk.h>

#include <array>
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
//
// Layout (read as 6 float4s by the GPU):
//   transform     [0..3]  mat4 world transform
//   pack0         [4]     {meshlet_offset, meshlet_count, material_index, vertex_buffer_slot}
//   pack1         [5]     {vertex_base_offset, asfloat(patch_quad_size), verts_per_edge, _pad}
//
// The last three uints in pack1 are terrain-only. For regular meshes they are
// zero (set by GpuInstance inst{}), and the terrain_cdlod.mesh.hlsl shader
// never references them for non-terrain instances.
struct alignas(16) GpuInstance {
    mat4 transform;            // 64 B  world matrix (column-major, GLM default)
    u32  meshlet_offset;       //  4 B  index into global meshlet buffer where this mesh starts
    u32  meshlet_count;        //  4 B  number of meshlets in this mesh
    u32  material_index;       //  4 B  bindless material index
    u32  vertex_buffer_slot;   //  4 B  bindless vertex SSBO slot
    u32  vertex_base_offset;   //  4 B  CDLOD terrain: base vertex offset inside vertex SSBO (0 for regular meshes)
    f32  patch_quad_size;      //  4 B  CDLOD terrain: world-space quad size (patch_size / quadsPerPatch)
    u32  verts_per_edge;       //  4 B  CDLOD terrain: vertices per patch edge = quadsPerPatch + 1
    u32  _pad;                 //  4 B  explicit tail padding — keeps 16-byte alignment
};                             // 96 B total

static_assert(sizeof(GpuInstance) == 96,
              "GpuInstance layout mismatch: GPU shader reads 6 float4s (96 bytes) per instance");

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
    // frameIndex selects which per-frame staging buffer to write into,
    // preventing CPU/GPU races when MAX_FRAMES_IN_FLIGHT > 1.
    // meshletLookupCount is the highest global meshlet id that any shader
    // (cull, material-eval) may ask the reverse-lookup about this frame —
    // typically meshlets.total_meshlet_count(). Entries not claimed by an
    // add_instance() range remain 0xFFFFFFFFu (orphaned).
    // Must be called after all add_instance() calls and before draw.
    void upload(VkCommandBuffer cmd, u32 frameIndex, u32 meshletLookupCount);

    // Bindless slot of the uploaded SSBO (registered in DescriptorAllocator).
    u32 slot() const { return m_slot; }

    // Bindless slot of the meshlet→instance reverse lookup buffer.
    // Indexed by globalMeshletId → instanceId (or 0xFFFFFFFFu if orphaned).
    // Replaces the O(n_instances) per-pixel/per-meshlet scan that both
    // material_eval and gpu_cull previously performed.
    u32 meshlet_to_instance_slot() const { return m_lookup_slot; }

    // Number of instances this frame.
    size_t instance_count() const { return m_instances.size(); }

    // CPU-side instance array for VS fallback draw command generation.
    const std::vector<GpuInstance>& cpu_instances() const { return m_instances; }

private:
    void ensure_capacity(size_t required);
    void ensure_lookup_capacity(size_t required_bytes);

    gfx::Device*              m_device      = nullptr;
    gfx::Allocator*           m_allocator   = nullptr;
    gfx::DescriptorAllocator* m_descriptors = nullptr;

    std::vector<GpuInstance> m_instances;

    // Meshlet → instance reverse lookup. Indexed by globalMeshletId; each
    // u32 holds the owning instance index or 0xFFFFFFFFu for "no current
    // instance owns this slot" (orphaned retired terrain range, or a meshlet
    // past max high-water this frame).
    std::vector<u32> m_meshletToInstance;

    // GPU-side SSBO.
    VkBuffer      m_gpu_buffer   = VK_NULL_HANDLE;
    VmaAllocation m_gpu_alloc    = nullptr;
    size_t        m_gpu_capacity = 0; // in bytes

    // Per-frame staging buffers — one per MAX_FRAMES_IN_FLIGHT to avoid
    // CPU/GPU races when the CPU writes frame N+1 while the GPU reads frame N.
    std::array<VkBuffer,      gfx::MAX_FRAMES_IN_FLIGHT> m_staging{};
    std::array<VmaAllocation, gfx::MAX_FRAMES_IN_FLIGHT> m_staging_alloc{};

    u32 m_slot = 0; // bindless SSBO slot

    // GPU-side meshlet→instance SSBO + per-frame staging.
    VkBuffer      m_lookup_gpu_buffer   = VK_NULL_HANDLE;
    VmaAllocation m_lookup_gpu_alloc    = nullptr;
    size_t        m_lookup_gpu_capacity = 0; // in bytes
    std::array<VkBuffer,      gfx::MAX_FRAMES_IN_FLIGHT> m_lookup_staging{};
    std::array<VmaAllocation, gfx::MAX_FRAMES_IN_FLIGHT> m_lookup_staging_alloc{};
    u32 m_lookup_slot = 0; // bindless SSBO slot for the reverse lookup
};

} // namespace enigma
