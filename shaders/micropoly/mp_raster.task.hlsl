// mp_raster.task.hlsl
// =====================
// Task (amplification) shader for the Micropoly HW raster pipeline (M3.3).
// One task workgroup per surviving cluster — dispatched indirectly from the
// indirect-draw buffer built by shaders/micropoly/mp_cluster_cull.comp.hlsl.
//
// Responsibilities:
//   1. Read the clusterIdx this workgroup is responsible for from the indirect
//      draw buffer (layout: 16-byte header + per-cluster 16-byte payload).
//   2. Load the cluster's DAG node to get its pageId.
//   3. Look up the PageCache slot the page is resident in (pageToSlotBuffer).
//      If the page is not resident this frame, skip the cluster cleanly —
//      DispatchMesh(0,...) is a legal no-op per EXT_mesh_shader spec. The
//      cluster-cull pass has already emitted a streaming request for this
//      page, so a subsequent frame will find it resident.
//   4. Read the page's PagePayloadHeader + ClusterOnDisk to get the cluster's
//      triangleCount, then dispatch ceil(triangleCount/MP_MESH_GROUP_TRIS)
//      mesh workgroups with the payload needed to fetch + rasterize vertices.
//
// Layout contracts:
//   * Indirect draw buffer (shaders/micropoly/mp_cluster_cull.comp.hlsl
//     emitDrawCmd): 16-byte header + cmd stride 16 bytes:
//       uint groupCountX, groupCountY, groupCountZ, clusterIdx
//     vkCmdDrawMeshTasksIndirectEXT is called with offset=16 stride=16 so
//     this shader sees (1,1,1) as groupCount and needs to re-read clusterIdx
//     from the indirect buffer via the bindless RWByteAddressBuffer[].
//   * MpDagNode: 48 B per node = 3 float4 (see mp_cluster_cull.comp.hlsl).
//   * PageCache layout: one SSBO of `slotBytes` units indexed by slot; the
//     runtime populates pageToSlotBuffer[pageId] = slotIndex or UINT32_MAX.
//   * Page payload (after zstd decompress):
//       PagePayloadHeader (16 B: clusterCount, version, _pad, _pad)
//       ClusterOnDisk[clusterCount] (76 B each)
//       vertex blob (32 B per vertex: vec3 pos, vec3 normal, vec2 uv)
//       triangle blob (3 B per triangle, packed u8 local indices)
//
// DXC -Zpc caveat: transpose(float4x4(...)) when reconstructing matrices
// from SSBO float4 rows. Not used in the task shader (no matrices loaded),
// but the mesh shader observes it.

#include "../common.hlsl"
#include "mp_cluster_layout.hlsl"
#include "mp_vis_pack.hlsl"

// Mesh-group sub-batch size — how many triangles a single mesh workgroup
// emits. Clusters can hold up to 128 triangles; we dispatch 128 mesh-shader
// threads per group so one thread rasterises one triangle. Keeping this
// constant matches MAX_TRIANGLES in the mesh shader.
#define MP_MESH_GROUP_TRIS 128u

// --- Bindless resource arrays (match mp_cluster_cull.comp.hlsl bindings) ----
[[vk::binding(2, 0)]]
StructuredBuffer<float4> g_buffers[] : register(t0, space1);

[[vk::binding(5, 0)]]
RWByteAddressBuffer g_rwBuffers[] : register(u1, space0);

// --- Push constants --------------------------------------------------------
// Shared with mp_raster.mesh.hlsl PSMain — matches
// MicropolyRasterPushBlock in src/renderer/micropoly/MicropolyRasterPass.cpp.
struct PushBlock {
    uint indirectBufferBindlessIndex; // RWByteAddressBuffer — header + cmds (M3.2 writer)
    uint dagBufferBindlessIndex;      // StructuredBuffer<MpDagNode> as float4
    uint pageToSlotBufferBindlessIndex; // StructuredBuffer<uint> (via float4 -> uint4) — pageId -> slotIndex
    uint pageCacheBufferBindlessIndex;  // RWByteAddressBuffer — entire page pool
    uint cameraSlot;                  // StructuredBuffer<float4> — CameraData
    uint visImageBindlessIndex;       // bindless storage-image slot (R64_UINT)
    uint pageSlotBytes;               // PageCache::slotBytes — used to compute slot offsets
    uint pageCount;                   // pageToSlot array capacity bound
    uint dagNodeCount;                // MpAssetHeader::dagNodeCount — DAG buffer bound
    uint rasterClassBufferBindlessIndex; // M4.4: u32 rasterClass per drawSlot (0=HW, 1=SW)
    uint pageFirstDagNodeBufferBindlessIndex; // M4.5: pageId -> firstDagNodeIdx (u32[])
};
[[vk::push_constant]] PushBlock pc;

// --- Task payload to mesh shader ------------------------------------------
// EXT_mesh_shader payload — limited groupshared bytes (<= 16 KiB spec min).
// All mesh groups for one cluster receive the same payload; mesh uses gid.x
// as the "sub-batch index" within the cluster to pick the 128-triangle slice.
//
// M4.5: localClusterIdx added so the mesh shader can address the correct
// ClusterOnDisk entry within a multi-cluster page. The task shader derives
// it via `globalDagNodeIdx - pageFirstDagNodeBuffer[pageId]`.
struct TaskPayload {
    uint clusterIdx;        // for vis-buffer packing / downstream lookup
    uint pageSlotOffsetB;   // page byte offset into the page-cache buffer
    uint clusterOnDiskOffB; // byte offset from page start to this cluster's ClusterOnDisk
    uint localClusterIdx;   // M4.5: page-local cluster index (0..clusterCount-1)
    uint triangleBlobOffB;  // byte offset from page start to the triangle blob;
                            // computed once here because mesh-shader lanes would
                            // otherwise all redundantly loop over ClusterOnDisk
                            // entries to sum vertexCount across the page.
};

groupshared TaskPayload s_payload;

// --- Helpers ---------------------------------------------------------------

// Read the clusterIdx for this task workgroup from the indirect-draw buffer.
// The cull pass laid the buffer out as:
//   offset 0  : count (u32) + 3 pad u32s
//   offset 16 : cmd[0] = (groupCountX, groupCountY, groupCountZ, clusterIdx)
//   offset 32 : cmd[1] ...
// We are dispatched with offset=16 stride=16, so gid.x is the command index
// within the array. clusterIdx is at byte offset 12 within each 16-byte cmd.
uint loadClusterIdx(uint cmdIndex) {
    RWByteAddressBuffer ind = g_rwBuffers[NonUniformResourceIndex(pc.indirectBufferBindlessIndex)];
    const uint base = 16u + cmdIndex * 16u;  // 16-byte header + stride 16
    return ind.Load(base + 12u);
}

// Load the pageId for a cluster. The DAG buffer is a StructuredBuffer<float4>
// in bindless slot pc.dagBufferBindlessIndex, 3 float4 per node. The pageId
// is packed in m2.w low-24 bits — see mp_cluster_cull.comp.hlsl::loadDagNode
// for the full layout.
uint loadPageIdForCluster(uint clusterIdx) {
    if (clusterIdx >= pc.dagNodeCount) return 0xFFFFFFFFu;
    StructuredBuffer<float4> buf = g_buffers[NonUniformResourceIndex(pc.dagBufferBindlessIndex)];
    const uint base = clusterIdx * 3u;
    float4 m2 = buf[base + 2u];
    const uint packed = asuint(m2.w);
    return packed & 0x00FFFFFFu;
}

// M4.5: look up the global DAG index of the page's first cluster so the
// task shader can compute `localClusterIdx = globalDagNodeIdx - firstDagIdx`
// for multi-cluster pages. Matches the packing used by mp_cluster_cull.
uint firstDagNodeIdxForPage(uint pageId) {
    if (pageId >= pc.pageCount) return 0u;
    StructuredBuffer<float4> buf = g_buffers[NonUniformResourceIndex(pc.pageFirstDagNodeBufferBindlessIndex)];
    const uint float4Idx = pageId >> 2u;
    const uint component = pageId & 3u;
    uint4 packed = asuint(buf[float4Idx]);
    if      (component == 0u) return packed.x;
    else if (component == 1u) return packed.y;
    else if (component == 2u) return packed.z;
    else                      return packed.w;
}

// Look up the resident PageCache slot for a pageId. The runtime-maintained
// pageToSlotBuffer is a StructuredBuffer<float4> of ceil(pageCount/4) entries;
// the pageId -> u32 mapping lives in the matching channel of the float4. A
// value of UINT32_MAX means "not resident this frame."
uint slotForPage(uint pageId) {
    if (pageId >= pc.pageCount) return 0xFFFFFFFFu;
    StructuredBuffer<float4> buf = g_buffers[NonUniformResourceIndex(pc.pageToSlotBufferBindlessIndex)];
    const uint float4Idx = pageId >> 2u;          // / 4
    const uint component = pageId & 3u;           // % 4
    uint4 packed = asuint(buf[float4Idx]);
    uint slot;
    if      (component == 0u) slot = packed.x;
    else if (component == 1u) slot = packed.y;
    else if (component == 2u) slot = packed.z;
    else                      slot = packed.w;
    return slot;
}

// --- Main ------------------------------------------------------------------
// Single-thread-per-group: each workgroup processes exactly one surviving
// cluster. We keep the group size at 1 so the amplification fan-out comes
// purely from DispatchMesh. This matches M3.3's "one cluster -> N mesh
// groups" shape without wasting lanes on a parallel fetch that would
// contend on groupshared writes.
// Implementation note on DispatchMesh placement:
//   EXT_mesh_shader requires DispatchMesh to be called EXACTLY ONCE per
//   invocation. Placing it in a conditionally-returning branch causes DXC's
//   SPIR-V legalization pass to fail with "terminator instruction outside
//   basic block" (reproduced on DXC pinned to the engine's Vulkan SDK).
//   The structurally-safe pattern is: compute a `groupCount` of 0 or 1
//   based on whether the cluster is renderable, and call DispatchMesh
//   exactly once at the end of the function. A 0-group dispatch is a
//   legal no-op per the mesh shader spec.
[numthreads(1, 1, 1)]
void ASMain(uint3 gid : SV_GroupID) {
    const uint clusterIdx = loadClusterIdx(gid.x);
    const uint pageId     = loadPageIdForCluster(clusterIdx);
    const uint slotIndex  = slotForPage(pageId);

    // M4.5 multi-cluster page resolution: `clusterIdx` is the GLOBAL DAG
    // node index. The page-local cluster index is
    // `globalDagIdx - pageFirstDagNodeBuffer[pageId]`. Multi-cluster pages
    // now address the correct ClusterOnDisk entry instead of always 0.
    const uint firstDagIdx = firstDagNodeIdxForPage(pageId);
    // M4.5 Phase 4 HIGH fix: corrupt asset (firstDagIdx > clusterIdx) skips
    // the cluster instead of silently rendering cluster 0; corruptSkip below
    // forces triangleCount=0 so DispatchMesh(0, 1, 1) becomes a no-op. The
    // ternary sidestepps the M3.3-style `localClusterIdx = 0u` pattern the
    // M4.5 plan called out — the value is still zero in the corrupt branch,
    // but the assignment is gated by the same expression that sets the skip.
    const bool corruptSkip = (clusterIdx < firstDagIdx);
    const uint localClusterIdx = (!corruptSkip) ? (clusterIdx - firstDagIdx) : 0u;

    const uint pageByteOffset = slotIndex * pc.pageSlotBytes;
    const uint clusterOnDiskOffsetInPage =
        MP_PAGE_PAYLOAD_HEADER_BYTES + localClusterIdx * MP_CLUSTER_ON_DISK_STRIDE;

    // Read triangleCount + compute page-wide triangleBlobOffset only when
    // the page is resident. Otherwise feed zeros so the dispatch below
    // collapses to groupCount=0 and the mesh shader does no work.
    uint triangleCount = 0u;
    uint triangleBlobOff = 0u;
    if (slotIndex != 0xFFFFFFFFu && !corruptSkip) {
        RWByteAddressBuffer pageBuf = g_rwBuffers[NonUniformResourceIndex(pc.pageCacheBufferBindlessIndex)];
        const uint triCountAddr = pageByteOffset + clusterOnDiskOffsetInPage + 4u;
        triangleCount = pageBuf.Load(triCountAddr);
        // Hard-clamp to 128 (plan cap kMpVisTriBits) as a defence against
        // corrupt pages — the mesh shader will never emit more than this.
        if (triangleCount > MP_MESH_GROUP_TRIS) triangleCount = MP_MESH_GROUP_TRIS;

        // Sum vertexCount across every ClusterOnDisk entry in this page to
        // locate where the triangle blob starts. Multi-cluster pages pack
        // vertex blobs back-to-back; the bake guarantees the blob ends at
        // the same address (vertexBlobStart + SUM(vertexCount)*32) the
        // triangle blob begins at. Loop bound clamped to the architectural
        // cap to sidestep corrupt clusterCount values.
        uint clusterCount = pageBuf.Load(pageByteOffset + 0u);
        // Clamp to the reader's kMaxClustersPerPage ceiling
        // (MpAssetReader.cpp). MUST equal mp_raster.mesh.hlsl's
        // pageVertexBlobStart clamp or the mesh shader's vertex-blob
        // offset drifts against the triangleBlobOff we publish here —
        // silent corruption on any page with clusterCount > 64.
        if (clusterCount > 4096u) clusterCount = 4096u;
        uint totalVertexBytes = 0u;
        for (uint ci = 0u; ci < clusterCount; ++ci) {
            const uint cAddr = pageByteOffset + MP_PAGE_PAYLOAD_HEADER_BYTES
                             + ci * MP_CLUSTER_ON_DISK_STRIDE;
            totalVertexBytes += pageBuf.Load(cAddr) * 32u;
        }
        triangleBlobOff = MP_PAGE_PAYLOAD_HEADER_BYTES
                        + clusterCount * MP_CLUSTER_ON_DISK_STRIDE
                        + totalVertexBytes;
    }

    s_payload.clusterIdx        = clusterIdx;
    s_payload.pageSlotOffsetB   = pageByteOffset;
    s_payload.clusterOnDiskOffB = clusterOnDiskOffsetInPage;
    s_payload.localClusterIdx   = localClusterIdx;
    s_payload.triangleBlobOffB  = triangleBlobOff;

    // M4.4: dispatcher classifier gate. The cull pass wrote a per-drawSlot
    // rasterClass tag (0=HW, 1=SW) in parallel with the indirect-draw cmd.
    // Clusters classified as SW are rasterized by the compute SW path and
    // must be skipped here (0-group DispatchMesh — legal no-op). Clusters
    // classified as HW (or any reserved class >= 2 for forward compat)
    // proceed through the mesh pipeline.
    // PERF-DIAG step 5: full gate chain. hwPath is the last suspect. If
    // the screen goes sparse now, the classifier is sending almost every
    // cluster to the SW path (rasterClass buffer) — either the pxArea
    // test is misfiring or the rasterClass write is wrong.
    RWByteAddressBuffer classBuf = g_rwBuffers[NonUniformResourceIndex(pc.rasterClassBufferBindlessIndex)];
    const uint rasterClass = classBuf.Load(gid.x * 4u);
    const bool hwPath = (rasterClass == kMpRasterClassHw);

    const uint groupCount = (slotIndex != 0xFFFFFFFFu && !corruptSkip
                             && triangleCount > 0u && hwPath) ? 1u : 0u;
    DispatchMesh(groupCount, 1u, 1u, s_payload);
}
