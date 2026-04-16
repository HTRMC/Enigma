#pragma once

#include "core/Math.h"
#include "core/Types.h"
#include "renderer/GpuMeshletBuffer.h"

#include <volk.h>

#include <deque>
#include <unordered_map>
#include <vector>

struct VmaAllocation_T;
using VmaAllocation = VmaAllocation_T*;

namespace enigma::gfx {
class Allocator;
class DescriptorAllocator;
class Device;
} // namespace enigma::gfx

namespace enigma {

class HeightmapLoader;
class IndirectDrawBuffer;
class GpuSceneBuffer;
struct GpuInstance;
struct Scene;

// CdlodConfig
// ===========
// Static tuning for the CDLOD terrain system. Populated at startup; the
// `terrainMaterialIdx` field is filled in during CdlodTerrain::initialize()
// once the dedicated terrain material has been appended to scene.materials.
struct CdlodConfig {
    u32 lodLevels          = 12;
    u32 quadsPerPatch      = 32;
    f32 leafPatchSize      = 1.0f;    // world-space size of LOD 0 patch
    f32 lodDistanceBase    = 8.0f;    // LOD switch dist at LOD 0 (≈4× leaf patch size for 4096m world/12 LODs)
    f32 lodDistanceGrow    = 2.0f;    // each level multiplies distance
    u32 terrainMaterialIdx = 0;       // set during initialize()
    u32 poolSlotsPerLod    = 96;      // vertex pool capacity per LOD
    u32 activationBudget   = 16;      // max new patch activations per frame
};

// A single quad-tree node. Leaves have lod == 0 (finest resolution).
struct CdlodNode {
    u32  lod        = 0;
    vec2 worldMin   = {};
    f32  size       = 0.0f;
    u32  childIndex[4] = { UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX };
};

// Per-LOD vertex pool — one large device-local SSBO that stores up to
// poolSlotsPerLod patch vertex payloads, addressed by slot index.
struct LodPool {
    VkBuffer      buffer          = VK_NULL_HANDLE;
    VmaAllocation alloc           = nullptr;
    u32           slot            = UINT32_MAX; // bindless vertex SSBO slot
    u32           slotCount       = 0;
    u32           nextSlot        = 0;          // ring allocator cursor
    u32           verticesPerSlot = 0;          // (quadsPerPatch+1)^2
};

// Runtime state of an active terrain patch.
struct TerrainPatch {
    u32  nodeIndex     = UINT32_MAX;
    u32  lod           = 0;
    vec2 worldMin      = {};
    f32  size          = 0.0f;
    bool active        = false;
    u32  poolSlot      = UINT32_MAX;
    u32  meshletOffset = UINT32_MAX;
    u32  meshletCount  = 0;
};

// CdlodTerrain
// ============
// Continuous Distance-Dependent LOD terrain for mesh-shader pipelines.
//
// Lifecycle:
//   1. Construct with device/allocator/descriptors.
//   2. After GpuMeshletBuffer::append() has been called for every scene mesh,
//      and after HeightmapLoader::load() has completed, call initialize().
//      This builds the static quad-tree, allocates per-LOD vertex pools,
//      uploads shared topology templates, and reserves meshlet-buffer
//      capacity for incremental patch activations.
//   3. Each frame, call update() with the current camera position. It
//      traverses the quad-tree, activates/deactivates patches (bounded by
//      activationBudget), streams vertex data to the pools, appends
//      Meshlet descriptors to the GPU buffer, and emits a GpuInstance
//      per active patch to sceneBuffer.
class CdlodTerrain {
public:
    CdlodTerrain(gfx::Device& device, gfx::Allocator& allocator,
                 gfx::DescriptorAllocator& descriptors);
    ~CdlodTerrain();

    CdlodTerrain(const CdlodTerrain&)            = delete;
    CdlodTerrain& operator=(const CdlodTerrain&) = delete;

    // Must be called after HeightmapLoader::load() and after
    // GpuMeshletBuffer::append() for all scene meshlets (so reserveCapacity
    // can be called with the right ceiling). Returns false and logs a warning
    // if VK_EXT_mesh_shader is unavailable.
    bool initialize(const CdlodConfig&  config,
                    HeightmapLoader&    heightmap,
                    GpuMeshletBuffer&   meshletBuffer,
                    IndirectDrawBuffer& indirectBuffer,
                    Scene&              scene,
                    VkCommandBuffer     uploadCmd);

    bool isEnabled() const { return m_enabled; }

    // Per-frame: traverse quad-tree, activate/deactivate patches, emit
    // GpuInstances. The cmd must be in the recording state; CDLOD records
    // staging -> pool and staging -> meshlet buffer copies on it.
    void update(const vec3&    cameraPos,
                VkCommandBuffer cmd,
                u32             frameIndex,
                GpuSceneBuffer& sceneBuffer);

    // Read-only access to the heightmap (used by Application for physics rebuild).
    const HeightmapLoader* heightmap() const { return m_heightmap; }

    // Shared-topology handle for the given LOD, used by the terrain VB pipeline
    // push constants. LOD 0 is identical to all others (same template is
    // uploaded per LOD); exposed as a single accessor to keep the renderer
    // decoupled from the m_lodTopology vector layout.
    GpuMeshletBuffer::SharedTopologyHandle sharedTopologyHandle() const {
        return m_lodTopology.empty() ? GpuMeshletBuffer::SharedTopologyHandle{}
                                     : m_lodTopology[0];
    }

private:
    void buildQuadTree();
    u32  subdivide(u32 lod, vec2 worldMin, f32 size);
    void buildMeshletTemplates(GpuMeshletBuffer& meshletBuffer, VkCommandBuffer cmd);
    void allocLodPool(u32 lod);
    u32  allocPoolSlot(u32 lod);
    void collectActive(u32 nodeIndex, const vec3& cameraPos,
                       std::vector<u32>& outNodes, u32 maxNodes) const;
    void activatePatch(u32               nodeIndex,
                       GpuMeshletBuffer& meshletBuffer,
                       VkCommandBuffer   cmd,
                       u32               frameIndex,
                       std::vector<VkBufferMemoryBarrier2>& poolBarriers);
    void deactivatePatch(u32 nodeIndex);
    GpuInstance buildInstance(const TerrainPatch& patch) const;

    // Retired pool slot + meshlet range, held until the frame counter has
    // advanced past MAX_FRAMES_IN_FLIGHT+1 since retirement (safe GPU
    // reclamation; the +1 accounts for the retiring frame having emitted the
    // patch as a GpuInstance before deactivation).
    struct RetiredSlot {
        u32 poolSlot;
        u32 meshletOffset;
        u32 meshletCount;
        u64 retireFrame;
    };

    // Per-frame pending staging buffer awaiting post-submit destruction.
    struct PendingStaging {
        VkBuffer      buf;
        VmaAllocation alloc;
        u64           frameCounter;
    };

    gfx::Device*              m_device      = nullptr;
    gfx::Allocator*           m_allocator   = nullptr;
    gfx::DescriptorAllocator* m_descriptors = nullptr;

    CdlodConfig        m_config{};
    HeightmapLoader*   m_heightmap      = nullptr;
    GpuMeshletBuffer*  m_meshletBuffer  = nullptr;
    bool               m_enabled        = false;

    std::vector<CdlodNode> m_nodeArena;
    u32                    m_rootIndex = UINT32_MAX;

    std::vector<LodPool> m_lodPools;
    std::vector<GpuMeshletBuffer::SharedTopologyHandle> m_lodTopology;

    // Per-LOD meshlet templates (CPU-side, used to fan out Meshlet records
    // for each activated patch). Indexed by LOD level.
    std::vector<std::vector<Meshlet>> m_lodTemplateMeshlets;
    // Per-LOD meshlet vertex-index arrays (CPU-side, parallel to
    // m_lodTemplateMeshlets). Used in activatePatch to compute per-meshlet
    // height min/max for bounding sphere Y correction.
    std::vector<std::vector<u32>> m_lodTemplateVertices;

    // Key: nodeIndex -> runtime patch record.
    std::unordered_map<u32, TerrainPatch> m_patches;

    // Per-LOD retired slot queues (fence-gated reclamation).
    std::vector<std::deque<RetiredSlot>> m_retiredSlots;
    // Per-LOD ready-to-reuse pool slots (drained from m_retiredSlots in bulk
    // so a burst of retirements is immediately available without ring fallback).
    std::vector<std::deque<u32>> m_readySlots;

    // Per-frame pending staging buffers awaiting safe destruction.
    std::vector<PendingStaging> m_pendingStaging;

    u32 m_activationsThisFrame = 0;
    u64 m_frameCounter         = 0; // incremented each update()
};

} // namespace enigma
