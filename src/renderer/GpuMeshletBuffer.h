#pragma once

#include "core/Types.h"
#include "gfx/FrameContext.h"
#include "renderer/Meshlet.h"

#include <volk.h>

#include <array>
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

    // Shared topology handle — per-LOD index arrays stored in separate SSBOs.
    // Used by CDLOD terrain: one template topology per LOD level, with
    // per-patch Meshlet descriptors appended to the main meshlet buffer.
    struct SharedTopologyHandle {
        u32 topologyVerticesSlot  = 0; // bindless StructuredBuffer<uint> — packed vertex indices
        u32 topologyTrianglesSlot = 0; // bindless RWByteAddressBuffer — packed triangle bytes
        u32 meshletCount          = 0; // number of meshlet templates for this LOD
        u32 vertexCount           = 0; // total vertices in shared topology (after dedup)
        u32 triangleCount         = 0; // total triangle-byte entries in shared topology
    };

    // Pre-allocate the device-local meshlet buffer to 'totalMeshlets' capacity.
    // Replaces upload() for the CDLOD path — must be called ONCE after all
    // scene-mesh append() calls. Allocates the GPU buffer, copies current
    // m_meshlets data, registers the bindless slot, and pre-allocates the
    // staging ring. The registered slot is stable for program lifetime
    // (no realloc after this).
    void reserveCapacity(VkCommandBuffer cmd, u32 totalMeshlets);

    // Upload a per-LOD shared topology SSBO (vertex indices + packed triangle
    // bytes). Called once per LOD during CdlodTerrain::initialize().
    SharedTopologyHandle uploadSharedTopology(VkCommandBuffer cmd,
                                              const std::vector<u32>& vertexIndices,
                                              const std::vector<u8>&  triangleBytes);

    // Append per-patch meshlet descriptors during the render loop. Returns the
    // meshletOffset for this patch's first descriptor. Uploads to the
    // pre-allocated GPU buffer at [currentEnd, currentEnd+n).
    // DEBUG assert: m_meshlets.size() + patchMeshlets.size() <= m_reservedCapacity.
    u32 appendIncremental(VkCommandBuffer cmd, u32 frameIndex,
                          const std::vector<Meshlet>& patchMeshlets);

    // Reset the per-frame staging ring cursor. MUST be called once per frame
    // before any appendIncremental() calls for that frame — otherwise multiple
    // activations in the same frame would overwrite each other's bytes at
    // staging offset 0 and all GPU copies would read the last-written bytes.
    void beginFrame(u32 frameIndex);

    // A previously-appended contiguous meshlet range (offset/count) returned
    // to the free-list. Reused by the next appendIncremental() call whose
    // requested count matches.
    struct MeshletRange { u32 offset; u32 count; };

    // Called when a terrain patch is deactivated and its fence is retired.
    // Returns the meshlet range to the free-list for later reuse.
    void freeMeshletRange(MeshletRange range);

    // Total number of meshlets across all appended meshes (grows as terrain
    // patches are activated via appendIncremental).
    u32 total_meshlet_count() const { return static_cast<u32>(m_meshlets.size()); }

    // Number of scene (non-terrain) meshlets — frozen at the point
    // reserveCapacity() or upload() is called. Use this in the renderer to
    // determine the scene-pass meshlet range; total_meshlet_count() grows as
    // terrain patches activate and would otherwise expand the scene range
    // into terrain-meshlet indices, corrupting the scene draw pass.
    u32 scene_meshlet_count() const { return m_sceneMeshletCount; }

    // CPU-side meshlet array for VS fallback draw command generation.
    // Valid after append(), including after upload().
    const std::vector<Meshlet>& cpu_meshlets() const { return m_meshlets; }

    // Bindless slots for the GPU shader to access.
    u32 meshlets_slot()   const { return m_meshlets_slot;   }
    u32 vertices_slot()   const { return m_vertices_slot;   }
    u32 triangles_slot()  const { return m_triangles_slot;  }

    // Raw meshlet buffer handle — exposed so CdlodTerrain can issue the
    // TRANSFER -> shader-read barrier after a frame of appendIncremental().
    VkBuffer meshlets_buffer() const { return m_meshlets_buf; }

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

    // Pre-allocated capacity for the CDLOD incremental-append path. Set by
    // reserveCapacity(); 0 means the legacy upload() path is in use.
    u32 m_reservedCapacity   = 0;

    // Frozen scene-meshlet count — set once in upload() / reserveCapacity()
    // so the renderer's scene-pass range stays fixed even as appendIncremental()
    // grows m_meshlets with terrain-patch descriptors.
    u32 m_sceneMeshletCount  = 0;

    // Ring-buffer staging for appendIncremental (MAX_FRAMES_IN_FLIGHT entries).
    // Each slot holds the maximum per-frame activation payload.
    std::array<VkBuffer,      gfx::MAX_FRAMES_IN_FLIGHT> m_stagingRingBuf{};
    std::array<VmaAllocation, gfx::MAX_FRAMES_IN_FLIGHT> m_stagingRingAlloc{};
    u32 m_stagingRingSize = 0; // bytes per staging slot

    // Per-frame byte cursor into the current staging-ring slot. Reset at the
    // start of each frame by beginFrame(); advances on each appendIncremental
    // call by the payload byte count. Prevents multiple activations in the
    // same frame from clobbering each other's staging bytes.
    u32 m_stagingCursor = 0;

    // Per-LOD topology SSBOs (up to 12 LODs). Indexed in parallel — one entry
    // per uploadSharedTopology() call.
    std::vector<VkBuffer>      m_topoVertBufs;
    std::vector<VmaAllocation> m_topoVertAllocs;
    std::vector<VkBuffer>      m_topoTriBufs;
    std::vector<VmaAllocation> m_topoTriAllocs;
    std::vector<u32>           m_topoVertSlots;
    std::vector<u32>           m_topoTriSlots;

    // Staging buffers used by uploadSharedTopology(); destroyed by
    // flush_staging() once the caller's submit has completed.
    std::vector<VkBuffer>      m_topoStagingBufs;
    std::vector<VmaAllocation> m_topoStagingAllocs;

    // Free-list of meshlet ranges previously returned by CdlodTerrain
    // (fence-retired). appendIncremental() reuses these before bumping the end.
    std::vector<MeshletRange>  m_freeMeshletRanges;
};

} // namespace enigma
