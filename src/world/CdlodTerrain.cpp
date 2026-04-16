#include "world/CdlodTerrain.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "gfx/Allocator.h"
#include "gfx/DescriptorAllocator.h"
#include "gfx/Device.h"
#include "gfx/FrameContext.h"
#include "renderer/GpuSceneBuffer.h"
#include "renderer/IndirectDrawBuffer.h"
#include "renderer/MeshletBuilder.h"
#include "scene/Scene.h"
#include "world/HeightmapLoader.h"

#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#if defined(_MSC_VER)
    #pragma warning(push)
    #pragma warning(disable: 4100 4127 4189 4324 4505)
#endif
#include <vk_mem_alloc.h>
#if defined(_MSC_VER)
    #pragma warning(pop)
#endif

#include <algorithm>
#include <cmath>
#include <cstring>

namespace enigma {

// Compact key -> index-in-arena (we key m_patches directly by nodeIndex).

CdlodTerrain::CdlodTerrain(gfx::Device& device, gfx::Allocator& allocator,
                           gfx::DescriptorAllocator& descriptors)
    : m_device(&device)
    , m_allocator(&allocator)
    , m_descriptors(&descriptors) {}

CdlodTerrain::~CdlodTerrain() {
    // Destroy any residual per-patch staging buffers (caller is expected to
    // have waited for GPU idle before destruction, so these are safe to free).
    for (const PendingStaging& p : m_pendingStaging) {
        if (p.buf != VK_NULL_HANDLE) {
            vmaDestroyBuffer(m_allocator->handle(), p.buf, p.alloc);
        }
    }
    m_pendingStaging.clear();

    for (LodPool& pool : m_lodPools) {
        if (pool.slot != UINT32_MAX) {
            m_descriptors->releaseStorageBuffer(pool.slot);
            pool.slot = UINT32_MAX;
        }
        if (pool.buffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(m_allocator->handle(), pool.buffer, pool.alloc);
            pool.buffer = VK_NULL_HANDLE;
            pool.alloc  = nullptr;
        }
    }
}

// ---------------------------------------------------------------------------
// initialize: one-shot setup. Caller provides an uploadCmd in recording
// state. Staging -> device copies are recorded on it; the caller is
// expected to submit + wait before using the data.
// ---------------------------------------------------------------------------

bool CdlodTerrain::initialize(const CdlodConfig&  config,
                              HeightmapLoader&    heightmap,
                              GpuMeshletBuffer&   meshletBuffer,
                              IndirectDrawBuffer& indirectBuffer,
                              Scene&              scene,
                              VkCommandBuffer     uploadCmd)
{
    if (!m_device->supportsMeshShaders()) {
        ENIGMA_LOG_WARN("[cdlod] mesh shaders not supported; terrain disabled");
        m_enabled = false;
        return false;
    }

    ENIGMA_ASSERT(config.lodLevels > 0 && config.lodLevels <= 16);
    ENIGMA_ASSERT(config.quadsPerPatch >= 2);
    ENIGMA_ASSERT(config.poolSlotsPerLod > 0);

    m_config        = config;
    m_heightmap     = &heightmap;
    m_meshletBuffer = &meshletBuffer;

    // Build the static quad-tree covering the entire heightmap footprint.
    buildQuadTree();

    // Build and upload the per-LOD shared topology templates.
    m_lodTopology.resize(config.lodLevels);
    m_lodTemplateMeshlets.resize(config.lodLevels);
    m_lodTemplateVertices.resize(config.lodLevels);
    buildMeshletTemplates(meshletBuffer, uploadCmd);

    // Allocate the per-LOD vertex pools.
    m_lodPools.resize(config.lodLevels);
    for (u32 lod = 0; lod < config.lodLevels; ++lod) {
        allocLodPool(lod);
    }

    // Per-LOD retired-slot queues (used by allocPoolSlot for fence-gated
    // reclamation). Sized once here and never resized.
    m_retiredSlots.resize(config.lodLevels);
    m_readySlots.resize(config.lodLevels);

    // Allocate a dedicated terrain material. Index 0 is reserved for the
    // default fallback material used by mesh primitives without one.
    const u32 matIdx = static_cast<u32>(scene.materials.size());
    Material terrainMat{};
    terrainMat.flags = Material::kFlagTerrain;
    scene.materials.push_back(terrainMat);
    ENIGMA_ASSERT(matIdx != 0 && "terrain material must not be index 0");
    m_config.terrainMaterialIdx = matIdx;

    // Reserve meshlet capacity. Total = already-appended scene meshlets +
    // terrain ceiling (poolSlotsPerLod * lodLevels * meshlets-per-patch).
    // The per-patch meshlet count is whichever LOD's template has the most.
    u32 maxPatchMeshlets = 0;
    for (const auto& tmpl : m_lodTemplateMeshlets) {
        maxPatchMeshlets = std::max(maxPatchMeshlets, static_cast<u32>(tmpl.size()));
    }
    // Conservative ceiling per the plan spec: 18 meshlets per patch.
    const u32 perPatchCeiling = std::max(maxPatchMeshlets, 18u);
    // Double the terrain ceiling to account for the retirement transient:
    // when the camera moves, deactivated patches occupy meshlet ranges for
    // MAX_FRAMES_IN_FLIGHT+1 frames before freeMeshletRange() reclaims them.
    // During that window, newly activated patches still append to the end,
    // so the buffer can transiently hold 2× the steady-state patch count.
    const u32 terrainCeiling  = config.poolSlotsPerLod * config.lodLevels * perPatchCeiling * 2u;
    const u32 sceneMeshlets   = meshletBuffer.total_meshlet_count();
    const u32 totalCeiling    = sceneMeshlets + terrainCeiling;

    meshletBuffer.reserveCapacity(uploadCmd, totalCeiling);

    // Ensure the indirect draw buffer can hold the max possible survivor count.
    if (indirectBuffer.capacity() < totalCeiling) {
        indirectBuffer.resize(totalCeiling);
    }

    m_enabled = true;
    ENIGMA_LOG_INFO("[cdlod] initialized: {} LODs, {} nodes, pool slots/LOD={}, "
                    "material idx={}, meshlet capacity={} ({} scene + {} terrain ceiling)",
                    config.lodLevels, m_nodeArena.size(), config.poolSlotsPerLod,
                    matIdx, totalCeiling, sceneMeshlets, terrainCeiling);
    return true;
}

// ---------------------------------------------------------------------------
// Quad-tree construction. Root covers the full heightmap footprint; the
// coordinate origin matches HeightmapLoader (centered around 0).
// ---------------------------------------------------------------------------

u32 CdlodTerrain::subdivide(u32 lod, vec2 worldMin, f32 size) {
    const u32 nodeIdx = static_cast<u32>(m_nodeArena.size());
    m_nodeArena.push_back({});
    CdlodNode& node = m_nodeArena[nodeIdx];
    node.lod      = lod;
    node.worldMin = worldMin;
    node.size     = size;

    if (lod > 0) {
        const u32 childLod = lod - 1;
        const f32 half     = size * 0.5f;
        const vec2 offs[4] = {
            vec2(0.0f, 0.0f),
            vec2(half, 0.0f),
            vec2(0.0f, half),
            vec2(half, half),
        };
        for (u32 i = 0; i < 4; ++i) {
            const u32 childIdx = subdivide(childLod, worldMin + offs[i], half);
            // m_nodeArena may have been reallocated by recursive push_backs;
            // re-fetch the node pointer through index.
            m_nodeArena[nodeIdx].childIndex[i] = childIdx;
        }
    }
    return nodeIdx;
}

void CdlodTerrain::buildQuadTree() {
    const f32 worldSize = m_heightmap ? m_heightmap->worldSize() : 4096.0f;
    // Root corner — centered around 0 to match HeightmapLoader::origin().
    const vec2 rootMin(-0.5f * worldSize, -0.5f * worldSize);

    m_nodeArena.clear();
    m_nodeArena.reserve((1u << (2u * m_config.lodLevels)) / 3u + 4u);

    const u32 rootLod = m_config.lodLevels - 1u;
    m_rootIndex = subdivide(rootLod, rootMin, worldSize);
}

// ---------------------------------------------------------------------------
// Per-LOD meshlet templates: a regular grid of (quadsPerPatch+1)^2 vertices
// with unit-spacing on X/Z (Y=0). The resulting Meshlet vertex/triangle
// offsets are *local to the LOD topology buffer* and remain identical for
// every patch at that LOD.
// ---------------------------------------------------------------------------

void CdlodTerrain::buildMeshletTemplates(GpuMeshletBuffer& meshletBuffer,
                                          VkCommandBuffer   cmd)
{
    const u32 N       = m_config.quadsPerPatch;
    const u32 verts   = (N + 1u) * (N + 1u);

    std::vector<float> positions(verts * 3u, 0.0f);
    for (u32 z = 0; z <= N; ++z) {
        for (u32 x = 0; x <= N; ++x) {
            const usize i = static_cast<usize>(z) * (N + 1u) + x;
            positions[i * 3 + 0] = static_cast<f32>(x);
            positions[i * 3 + 1] = 0.0f;
            positions[i * 3 + 2] = static_cast<f32>(z);
        }
    }

    std::vector<u32> indices;
    indices.reserve(static_cast<usize>(N) * N * 6u);
    for (u32 z = 0; z < N; ++z) {
        for (u32 x = 0; x < N; ++x) {
            const u32 i0 = z * (N + 1u) + x;
            const u32 i1 = z * (N + 1u) + (x + 1u);
            const u32 i2 = (z + 1u) * (N + 1u) + x;
            const u32 i3 = (z + 1u) * (N + 1u) + (x + 1u);
            // Two triangles per quad, CCW winding (+Y up).
            indices.push_back(i0); indices.push_back(i2); indices.push_back(i1);
            indices.push_back(i1); indices.push_back(i2); indices.push_back(i3);
        }
    }

    // Build meshlets once; the same topology template is reused for every
    // LOD (geometry is scale-invariant on a regular grid).
    MeshletData templateData = MeshletBuilder::build(
        positions.data(), verts,
        indices.data(), indices.size());

    for (u32 lod = 0; lod < m_config.lodLevels; ++lod) {
        // Upload topology SSBOs (per-LOD, since each LOD's patches will fetch
        // through a handle slot pair). The data is identical, but separate
        // buffers keep the bindless indirection simple.
        GpuMeshletBuffer::SharedTopologyHandle handle =
            meshletBuffer.uploadSharedTopology(cmd,
                                               templateData.meshlet_vertices,
                                               templateData.meshlet_triangles);
        handle.meshletCount = static_cast<u32>(templateData.meshlets.size());
        m_lodTopology[lod]         = handle;
        m_lodTemplateMeshlets[lod] = templateData.meshlets;
        m_lodTemplateVertices[lod] = templateData.meshlet_vertices;
    }

    ENIGMA_LOG_INFO("[cdlod] built meshlet template: {} meshlets, {} vertex-idx, {} tri-bytes",
                    templateData.meshlets.size(),
                    templateData.meshlet_vertices.size(),
                    templateData.meshlet_triangles.size());
}

// ---------------------------------------------------------------------------
// Per-LOD vertex pool allocation.
// ---------------------------------------------------------------------------

void CdlodTerrain::allocLodPool(u32 lod) {
    LodPool& pool = m_lodPools[lod];
    const u32 N   = m_config.quadsPerPatch;
    pool.verticesPerSlot = (N + 1u) * (N + 1u);
    pool.slotCount       = m_config.poolSlotsPerLod;
    pool.nextSlot        = 0;

    // Each vertex stores only its Y (height) as a single float. The mesh
    // shader reconstructs X and Z from the patch-local vertex index using
    // verts_per_edge and patch_quad_size (passed per-GpuInstance). This keeps
    // stride at 4B/vertex — consistent with the StructuredBuffer<float4>
    // bindless array: the shader reads float4 elements and picks the
    // appropriate .x/.y/.z/.w component via (idx / 4) and (idx % 4).
    //
    // The total buffer size is rounded up to a multiple of 16 so that reading
    // the pool as StructuredBuffer<float4> can always dereference the final
    // float4 containing the last vertex's Y.
    const VkDeviceSize bytesPerSlot =
        static_cast<VkDeviceSize>(pool.verticesPerSlot) * sizeof(f32);
    const VkDeviceSize totalFloats = static_cast<VkDeviceSize>(pool.slotCount)
                                   * pool.verticesPerSlot;
    const VkDeviceSize totalBytes = ((totalFloats * sizeof(f32) + 15u) / 16u) * 16u;

    VkBufferCreateInfo bufCI{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufCI.size        = totalBytes;
    bufCI.usage       = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    ENIGMA_VK_CHECK(vmaCreateBuffer(m_allocator->handle(), &bufCI, &allocCI,
                                    &pool.buffer, &pool.alloc, nullptr));

    pool.slot = m_descriptors->registerStorageBuffer(pool.buffer, totalBytes);

    ENIGMA_LOG_INFO("[cdlod] LOD {} pool: {} slots x {}B ({}KB) slot={}",
                    lod, pool.slotCount, static_cast<u64>(bytesPerSlot),
                    static_cast<u64>(totalBytes / 1024u), pool.slot);
}

u32 CdlodTerrain::allocPoolSlot(u32 lod) {
    LodPool& pool = m_lodPools[lod];

    // First, drain retired entries whose retirement frame is more than
    // MAX_FRAMES_IN_FLIGHT+1 frames in the past — the +1 accounts for the
    // retiring frame having already emitted a GpuInstance referencing the
    // slot to the scene buffer BEFORE the deactivation took effect. Once that
    // window has passed, the pool slot and meshlet range are guaranteed
    // GPU-safe to reclaim. Uses m_frameCounter — the same clock used for
    // staging buffer drain — NOT any timeline semaphore value.
    // Drain ALL eligible retired entries into m_readySlots in one pass so a
    // burst of retirements on the same frame is immediately available without
    // forcing ring-allocator fallback on subsequent activations this frame.
    auto& retired = m_retiredSlots[lod];
    while (!retired.empty()) {
        const RetiredSlot& front = retired.front();
        if (m_frameCounter < front.retireFrame + gfx::MAX_FRAMES_IN_FLIGHT + 1u)
            break;
        if (m_meshletBuffer && front.meshletCount > 0)
            m_meshletBuffer->freeMeshletRange({ front.meshletOffset, front.meshletCount });
        m_readySlots[lod].push_back(front.poolSlot);
        retired.pop_front();
    }

    // Return from the ready queue before touching the ring allocator.
    auto& ready = m_readySlots[lod];
    if (!ready.empty()) {
        const u32 reclaimedSlot = ready.front();
        ready.pop_front();
        return reclaimedSlot;
    }

    // Fallback: ring allocator. Only wraps after poolSlotsPerLod unique
    // activations without any retirement arriving in time.
    const u32 slot = pool.nextSlot;
    pool.nextSlot = (pool.nextSlot + 1u) % pool.slotCount;
    return slot;
}

// ---------------------------------------------------------------------------
// Quad-tree traversal. Collects the "should be active" node set.
// ---------------------------------------------------------------------------

void CdlodTerrain::collectActive(u32 nodeIndex, const vec3& cameraPos,
                                  std::vector<u32>& outNodes, u32 maxNodes) const
{
    if (nodeIndex == UINT32_MAX) return;
    // Hard cap: pool is full, no point collecting more.
    // Children are visited nearest-first (see below) so this drops far nodes,
    // not near ones — near tiles are always collected before the cap hits.
    if (outNodes.size() >= maxNodes) return;

    const CdlodNode& node = m_nodeArena[nodeIndex];

    // Distance from camera to node center (XZ plane only — y ignored).
    const vec2 center = node.worldMin + vec2(node.size * 0.5f);
    const f32  dx     = cameraPos.x - center.x;
    const f32  dz     = cameraPos.z - center.y;
    const f32  dist   = std::sqrt(dx * dx + dz * dz);

    // LOD switch distance for this node.
    const f32 switchDist = m_config.lodDistanceBase
                         * std::pow(m_config.lodDistanceGrow, static_cast<f32>(node.lod));

    const bool hasChildren = node.lod > 0 && node.childIndex[0] != UINT32_MAX;

    if (hasChildren && dist < switchDist) {
        // Sort children nearest-first so when the cap fires the dropped nodes
        // are the farthest ones, not the closest.  4-element insertion sort —
        // no heap allocation.
        u32 order[4] = {0, 1, 2, 3};
        f32 childDist2[4];
        for (u32 i = 0; i < 4; ++i) {
            if (node.childIndex[i] == UINT32_MAX) { childDist2[i] = FLT_MAX; continue; }
            const CdlodNode& c  = m_nodeArena[node.childIndex[i]];
            const vec2       cc = c.worldMin + vec2(c.size * 0.5f);
            childDist2[i] = (cameraPos.x - cc.x) * (cameraPos.x - cc.x)
                          + (cameraPos.z - cc.y) * (cameraPos.z - cc.y);
        }
        for (u32 i = 1; i < 4; ++i)
            for (u32 j = i; j > 0 && childDist2[order[j]] < childDist2[order[j-1]]; --j)
                std::swap(order[j], order[j-1]);

        for (u32 i = 0; i < 4; ++i)
            collectActive(node.childIndex[order[i]], cameraPos, outNodes, maxNodes);
    } else {
        outNodes.push_back(nodeIndex);
    }
}

// ---------------------------------------------------------------------------
// Per-frame update: activate/deactivate patches, emit GpuInstances.
// ---------------------------------------------------------------------------

void CdlodTerrain::update(const vec3&    cameraPos,
                           VkCommandBuffer cmd,
                           u32             frameIndex,
                           GpuSceneBuffer& sceneBuffer)
{
    if (!m_enabled) return;

    ++m_frameCounter;

    // Reset the per-frame staging-ring cursor so each frame writes starting at
    // byte 0 of the assigned ring slot. The previous frame's bytes in that
    // slot are safe to overwrite — the GPU completed its copy before this
    // cmd buffer is recorded (enforced by the frame fence upstream).
    if (m_meshletBuffer != nullptr) {
        m_meshletBuffer->beginFrame(frameIndex);
    }

    // 0. Drain per-patch staging buffers from frames older than
    //    MAX_FRAMES_IN_FLIGHT — the GPU has long since finished the
    //    vkCmdCopyBuffer recorded at the time of allocation.
    m_pendingStaging.erase(
        std::remove_if(m_pendingStaging.begin(), m_pendingStaging.end(),
            [&](const PendingStaging& p) {
                if (m_frameCounter >= p.frameCounter + gfx::MAX_FRAMES_IN_FLIGHT) {
                    vmaDestroyBuffer(m_allocator->handle(), p.buf, p.alloc);
                    return true;
                }
                return false;
            }),
        m_pendingStaging.end());

    m_activationsThisFrame = 0;

    // 1. Collect desired node set.
    // Reserve pool capacity up front — avoids realloc churn during traversal.
    // collectActive caps output at this size so traversal exits early once
    // the pool is full rather than visiting the entire 4M-node leaf layer.
    const u32 maxDesired = m_config.poolSlotsPerLod * m_config.lodLevels;
    std::vector<u32> desired;
    desired.reserve(maxDesired);
    collectActive(m_rootIndex, cameraPos, desired, maxDesired);

    // 2. Compute set differences.
    // Build toActivate BEFORE sorting desired so that collectActive's nearest-first
    // traversal order is preserved.  Sorting desired first was destroying this order
    // and causing the activation budget to be spent on arbitrary (node-index ordered)
    // patches instead of the closest ones, leaving near terrain holes for several frames.
    std::vector<u32> toActivate;
    toActivate.reserve(desired.size());
    for (u32 nodeIdx : desired) {
        if (m_patches.find(nodeIdx) == m_patches.end()) {
            toActivate.push_back(nodeIdx);
        }
    }

    // Sort desired now (only needed for the binary_search in toDeactivate computation
    // and in the deactivation child-readiness check below).
    std::sort(desired.begin(), desired.end());

    std::vector<u32> toDeactivate;
    toDeactivate.reserve(m_patches.size());
    for (const auto& kv : m_patches) {
        if (!std::binary_search(desired.begin(), desired.end(), kv.first)) {
            toDeactivate.push_back(kv.first);
        }
    }

    // 3. Activate new patches, bounded by activationBudget per frame. Collect
    //    pool-buffer barrier ranges in one vector; the terminal barrier below
    //    emits them all in a single vkCmdPipelineBarrier2 call.
    std::vector<VkBufferMemoryBarrier2> poolBarriers;
    poolBarriers.reserve(m_config.activationBudget);
    for (u32 nodeIdx : toActivate) {
        if (m_activationsThisFrame >= m_config.activationBudget) break;
        activatePatch(nodeIdx, *m_meshletBuffer, cmd, frameIndex, poolBarriers);
        ++m_activationsThisFrame;
    }

    // 4. Retire patches no longer needed (frame-counter-gated reclamation).
    // Guard: do NOT deactivate a patch whose direct children are in the desired set
    // but not yet active.  The activation budget (step 3) may have left some children
    // unloaded for this frame.  Deactivating the parent before its children are ready
    // creates terrain holes that persist until the budget catches up — typically several
    // frames at the default activationBudget of 16.  Keeping the parent alive as a
    // coarser fallback eliminates the hole at the cost of one extra frame of the old LOD.
    for (u32 nodeIdx : toDeactivate) {
        const CdlodNode& node = m_nodeArena[nodeIdx];
        bool childrenReady = true;
        if (node.lod > 0) {
            for (u32 ci = 0; ci < 4; ++ci) {
                const u32 childIdx = node.childIndex[ci];
                if (childIdx == UINT32_MAX) continue;
                // Child is wanted by the desired set but has not been activated yet
                // (activation budget was exhausted this frame).
                if (std::binary_search(desired.begin(), desired.end(), childIdx) &&
                    m_patches.find(childIdx) == m_patches.end()) {
                    childrenReady = false;
                    break;
                }
            }
        }
        if (childrenReady) {
            deactivatePatch(nodeIdx);
        }
        // else: keep parent alive for this frame; child activation continues next frame.
    }

    // 5. Emit ONE terminal barrier covering:
    //    - every pool sub-range written this frame (collected above), and
    //    - the shared meshlet buffer if any meshlet descriptors were appended.
    //    Avoids the N barriers the old per-activation code emitted.
    if (m_activationsThisFrame > 0 && m_meshletBuffer != nullptr) {
        VkBufferMemoryBarrier2 meshletBarrier{
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2, nullptr,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
            | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT
            | VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            m_meshletBuffer->meshlets_buffer(), 0, VK_WHOLE_SIZE
        };
        poolBarriers.push_back(meshletBarrier);
    }
    if (!poolBarriers.empty()) {
        VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dep.bufferMemoryBarrierCount = static_cast<u32>(poolBarriers.size());
        dep.pBufferMemoryBarriers    = poolBarriers.data();
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    // 6. Emit a GpuInstance per active patch.
    for (const auto& kv : m_patches) {
        const TerrainPatch& patch = kv.second;
        if (!patch.active) continue;
        sceneBuffer.add_instance(buildInstance(patch));
    }
}

// ---------------------------------------------------------------------------
// Patch activation.
// ---------------------------------------------------------------------------

void CdlodTerrain::activatePatch(u32               nodeIndex,
                                  GpuMeshletBuffer& meshletBuffer,
                                  VkCommandBuffer   cmd,
                                  u32               frameIndex,
                                  std::vector<VkBufferMemoryBarrier2>& poolBarriers)
{
    ENIGMA_ASSERT(nodeIndex < m_nodeArena.size());
    const CdlodNode& node = m_nodeArena[nodeIndex];
    const u32 lod = node.lod;
    ENIGMA_ASSERT(lod < m_lodPools.size());

    TerrainPatch patch{};
    patch.nodeIndex = nodeIndex;
    patch.lod       = lod;
    patch.worldMin  = node.worldMin;
    patch.size      = node.size;

    // 1. Allocate a vertex pool slot and stream Y-only heights into it.
    patch.poolSlot = allocPoolSlot(lod);

    const u32 N          = m_config.quadsPerPatch;
    const u32 vertCount  = (N + 1u) * (N + 1u);
    const f32 quadSize   = node.size / static_cast<f32>(N);

    // Y-only vertex payload — one float (height) per vertex. The mesh shader
    // reconstructs world X/Z from the patch-local vertex index using
    // verts_per_edge and patch_quad_size from the GpuInstance.
    std::vector<f32> heights(vertCount, 0.0f);
    for (u32 z = 0; z <= N; ++z) {
        for (u32 x = 0; x <= N; ++x) {
            const f32 worldX = node.worldMin.x + static_cast<f32>(x) * quadSize;
            const f32 worldZ = node.worldMin.y + static_cast<f32>(z) * quadSize;
            const usize i    = static_cast<usize>(z) * (N + 1u) + x;
            heights[i] = m_heightmap->sampleBilinear(worldX, worldZ);
        }
    }

    // Host-visible staging for this patch's height payload. Destruction is
    // deferred via m_pendingStaging until the frame's GPU copy is known to
    // have completed (see update()'s drain at the top of each frame).
    const VkDeviceSize payloadBytes =
        static_cast<VkDeviceSize>(heights.size()) * sizeof(f32);

    VkBufferCreateInfo stageCI{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    stageCI.size        = payloadBytes;
    stageCI.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo stageAllocCI{};
    stageAllocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    stageAllocCI.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT
                       | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    VkBuffer      stagingBuf   = VK_NULL_HANDLE;
    VmaAllocation stagingAlloc = nullptr;
    VmaAllocationInfo info{};
    ENIGMA_VK_CHECK(vmaCreateBuffer(m_allocator->handle(), &stageCI, &stageAllocCI,
                                    &stagingBuf, &stagingAlloc, &info));
    std::memcpy(info.pMappedData, heights.data(), static_cast<size_t>(payloadBytes));
    vmaFlushAllocation(m_allocator->handle(), stagingAlloc, 0, VK_WHOLE_SIZE);

    // Copy staging -> LodPool[lod].buffer at slot offset. One float per vertex.
    LodPool& pool = m_lodPools[lod];
    const VkDeviceSize dstOff = static_cast<VkDeviceSize>(patch.poolSlot)
                              * pool.verticesPerSlot * sizeof(f32);
    VkBufferCopy region{ 0, dstOff, payloadBytes };
    vkCmdCopyBuffer(cmd, stagingBuf, pool.buffer, 1, &region);

    // Collect (don't emit) a per-range barrier covering this pool write.
    // update() emits one vkCmdPipelineBarrier2 containing all of them after
    // the activation loop — avoids activationBudget sequential barriers.
    poolBarriers.push_back(VkBufferMemoryBarrier2{
        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2, nullptr,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT
        | VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT,
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        pool.buffer, dstOff, payloadBytes
    });

    // Defer staging destruction — the vkCmdCopyBuffer above is pending on
    // this frame's command buffer, so freeing the staging buffer inline is
    // spec-illegal. Track it; update() drains entries older than
    // MAX_FRAMES_IN_FLIGHT frames at the top of each frame.
    m_pendingStaging.push_back({ stagingBuf, stagingAlloc, m_frameCounter });

    // 2. Build per-patch Meshlet descriptors from the LOD template.
    const std::vector<Meshlet>& templateMeshlets = m_lodTemplateMeshlets[lod];
    const std::vector<u32>&     templateVerts     = m_lodTemplateVertices[lod];
    std::vector<Meshlet> patchMeshlets = templateMeshlets;
    const f32 patchScale = node.size / static_cast<f32>(m_config.quadsPerPatch);
    for (Meshlet& m : patchMeshlets) {
        // Scale the template's unit-spacing bounds up to this patch's size.
        // Do NOT add a world-space translation here — inst.transform is a
        // translation matrix by (worldMin.x, 0, worldMin.y), and the cull
        // shader applies mul(inst.transform, float4(center, 1)) to bring the
        // sphere into world space. Adding the translation here would double it.
        m.bounding_sphere_center *= patchScale;
        m.bounding_sphere_radius *= patchScale;

        // Correct the bounding sphere's Y component. The template was built
        // with all vertices at Y=0; actual terrain heights can be large.
        // Without this correction the frustum cull rejects visible patches
        // whose sphere centre sits far below the actual terrain surface.
        f32 minH =  std::numeric_limits<f32>::max();
        f32 maxH = -std::numeric_limits<f32>::max();
        for (u32 vi = 0; vi < m.vertex_count; ++vi) {
            const u32 vtxIdx = templateVerts[m.vertex_offset + vi];
            const f32 h      = heights[vtxIdx];
            if (h < minH) minH = h;
            if (h > maxH) maxH = h;
        }
        if (minH > maxH) { minH = 0.0f; maxH = 0.0f; }
        const f32 heightMid  = 0.5f * (minH + maxH);
        const f32 heightHalf = 0.5f * (maxH - minH);
        m.bounding_sphere_center.y = heightMid;
        // Expand radius to cover vertical extent (Pythagoras: XZ radius ⊥ Y).
        m.bounding_sphere_radius = std::sqrt(
            m.bounding_sphere_radius * m.bounding_sphere_radius +
            heightHalf * heightHalf);
    }

    patch.meshletOffset = meshletBuffer.appendIncremental(cmd, frameIndex, patchMeshlets);
    patch.meshletCount  = static_cast<u32>(patchMeshlets.size());
    patch.active        = true;

    m_patches.emplace(nodeIndex, patch);
}

// ---------------------------------------------------------------------------
// Patch deactivation. The pool slot and meshlet range are handed to the
// per-LOD retired-slot queue; allocPoolSlot() reclaims them once the fence
// value has advanced past MAX_FRAMES_IN_FLIGHT frames.
// ---------------------------------------------------------------------------

void CdlodTerrain::deactivatePatch(u32 nodeIndex) {
    auto it = m_patches.find(nodeIndex);
    if (it == m_patches.end()) return;
    const TerrainPatch& patch = it->second;

    if (patch.lod < m_retiredSlots.size()) {
        // Stamp with m_frameCounter — this is the same clock used for the
        // staging drain. allocPoolSlot() will reclaim once the counter has
        // advanced by at least MAX_FRAMES_IN_FLIGHT+1 past this value.
        m_retiredSlots[patch.lod].push_back({
            patch.poolSlot,
            patch.meshletOffset,
            patch.meshletCount,
            m_frameCounter
        });
    }

    m_patches.erase(it);
}

// ---------------------------------------------------------------------------
// Build the per-patch GpuInstance written to the per-frame scene SSBO.
// ---------------------------------------------------------------------------

GpuInstance CdlodTerrain::buildInstance(const TerrainPatch& patch) const {
    GpuInstance inst{};
    inst.transform = glm::translate(mat4(1.0f),
                                    vec3(patch.worldMin.x, 0.0f, patch.worldMin.y));
    inst.meshlet_offset     = patch.meshletOffset;
    inst.meshlet_count      = patch.meshletCount;
    inst.material_index     = m_config.terrainMaterialIdx;
    inst.vertex_buffer_slot = m_lodPools[patch.lod].slot;
    // vertex_base_offset is the float index (stride 4B) of the start of this
    // patch's Y-value block in the per-LOD pool. See allocLodPool() — the
    // pool is 1 float per vertex (pure height), NOT 3 floats (XYZ).
    inst.vertex_base_offset = patch.poolSlot
                            * (m_config.quadsPerPatch + 1u)
                            * (m_config.quadsPerPatch + 1u);
    // Terrain-only fields consumed by terrain_cdlod.mesh.hlsl to reconstruct
    // world X/Z per-vertex without needing to store them in the pool.
    inst.patch_quad_size    = patch.size / static_cast<f32>(m_config.quadsPerPatch);
    inst.verts_per_edge     = m_config.quadsPerPatch + 1u;
    // inst._pad is zero-initialized by GpuInstance inst{}.
    return inst;
}

} // namespace enigma
