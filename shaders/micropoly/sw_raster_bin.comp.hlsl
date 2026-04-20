// sw_raster_bin.comp.hlsl
// ========================
// M4.2 — SW rasteriser binning pass. One compute workgroup per cull-survivor
// cluster (dispatch count comes from the dispatchIndirect buffer populated by
// sw_raster_bin_prep.comp.hlsl, which in turn reads the u32 count header
// written by mp_cluster_cull.comp.hlsl::emitDrawCmd).
//
// Responsibilities (binning ONLY — the fragment/rasterisation pass lands in
// M4.3 and consumes this milestone's outputs without touching these files):
//
//   1. Read the workgroup's clusterIdx from the cull indirect-draw buffer.
//   2. Walk the cluster's triangles (capped at MP_SW_MAX_TRIS). For each
//      triangle, load its 3 vertex positions from the PageCache, project to
//      clip -> NDC -> viewport -> pixel space.
//   3. Reject degenerate (zero-area) and back-facing triangles (negative
//      signed area in screen space — front-face is CCW, matching HW raster
//      under reverse-Z).
//   4. Compute a pixel-space AABB, clip to viewport, iterate the 8x8 tiles it
//      overlaps.
//   5. For each overlapping tile:
//        oldCount = InterlockedAdd(tileBinCount[tileIdx], 1)
//        if (oldCount < MP_SW_TILE_BIN_CAP)
//            write {clusterIdx, triIdx} packed u32 to tileBinEntries[...]
//        else
//            spillSlot = InterlockedAdd(spillCount[0], 1)
//            if (spillSlot < MP_SW_SPILL_CAP) write {tileIdx, triRef}
//            else InterlockedAdd(spillCount[1], 1)  // dropped counter
//
// Layout contracts:
//   * tileBinCount  : RWByteAddressBuffer, numTiles u32s.
//   * tileBinEntries: RWByteAddressBuffer, numTiles * MP_SW_TILE_BIN_CAP u32s.
//   * spillBuffer   : RWByteAddressBuffer, header {u32 count, u32 dropped}
//                     followed by MP_SW_SPILL_CAP * {u32 tileIdx, u32 triRef}.
//
// Packed triRef layout mirrors vis-pack v2 (mp_vis_pack.hlsl): cluster in the
// upper 23 bits, triangle in the lower 7 bits. kMpRasterClassShift is 30 so
// bits 29..7 are the cluster field, bits 6..0 are the triangle field. Using
// the same layout lets M4.3's fragment pass feed the ref straight back into
// PackMpVis64 without an extra shuffle.
//
// DXC -Zpc caveat: float4x4(v0..v3) takes COLUMN vectors under -Zpc -spirv,
// so matrix SSBO loads use transpose(float4x4(...)). Grep the rest of the
// shaders/ tree for the identical pattern.

#include "../common.hlsl"
#include "mp_vis_pack.hlsl"
#include "mp_cluster_layout.hlsl"

// --- Tile + bin sizing constants ------------------------------------------
// Per plan Condition 12 / §2.3: these are #defines, NOT specialization
// constants — thread-group dims and shared-memory array sizes can't be
// spec-const-sized. A Fork-C1 → C2 tile-size swap is a one-line edit +
// recompile.
#define MP_SW_TILE_X       8u
#define MP_SW_TILE_Y       8u
#define MP_SW_TILE_THREADS 64u
// Must mirror sw_raster.comp.hlsl. 1024 now that overflow routes to a
// per-tile spill linked list — entries past the fixed bin are preserved
// rather than dropped. Before the linked list landed, overflow dropped
// to a counter (non-deterministic) so we had to bump the cap to 4096
// to hide drops; that's no longer necessary.
#define MP_SW_TILE_BIN_CAP 1024u
// Must mirror sw_raster.comp.hlsl. Raised to 1M so the per-tile spill
// lists can absorb small-on-screen BMW without dropping at the tile
// boundary (symptom: 8x8 square holes in the mesh that get worse as
// the model shrinks).
// Must mirror sw_raster.comp.hlsl. Raised 1M → 16M so dense BMW tiles
// don't overflow the global spill counter and drop per-frame non-
// deterministically (which showed up as tile-shaped flickering holes).
#define MP_SW_SPILL_CAP    16777216u

// Clusters carry up to 128 triangles on disk; 64 threads cover 2 triangles
// each at worst when iterating the triangle list.
#define MP_SW_MAX_TRIS     128u

// --- Bindless resource arrays ---------------------------------------------
[[vk::binding(2, 0)]]
StructuredBuffer<float4> g_buffers[] : register(t0, space1);

[[vk::binding(5, 0)]]
RWByteAddressBuffer g_rwBuffers[] : register(u1, space0);

// --- Push constants --------------------------------------------------------
// Matches MicropolySwRasterBinPushBlock in MicropolySwRasterPass.cpp. Field
// order mirrors the C++ struct exactly.
struct PushBlock {
    // Inputs.
    uint indirectBufferBindlessIndex;   // cull survivors (header + cmds)
    uint dagBufferBindlessIndex;        // StructuredBuffer<MpDagNode> as float4
    uint pageToSlotBufferBindlessIndex; // pageId -> slotIndex or UINT32_MAX
    uint pageCacheBufferBindlessIndex;  // entire page pool
    uint cameraSlot;                    // StructuredBuffer<float4> CameraData

    // Outputs.
    uint tileBinCountBindlessIndex;     // RWByteAddressBuffer u32 * numTiles
    uint tileBinEntriesBindlessIndex;   // RWByteAddressBuffer u32 * numTiles * BIN_CAP
    uint spillBufferBindlessIndex;      // RWByteAddressBuffer header + spill array
    // M4.6 per-tile spill linked list: one u32 head per tile, reset to
    // UINT32_MAX each frame. On overflow, atomicExchange returns the prior
    // head and we write {prev, triRef} to the spill slot.
    uint spillHeadsBufferBindlessIndex; // RWByteAddressBuffer u32 * numTiles

    // Runtime constants.
    uint viewportWidth;   // pixels
    uint viewportHeight;  // pixels
    uint tilesX;          // ceil(viewportWidth  / MP_SW_TILE_X)
    uint tilesY;          // ceil(viewportHeight / MP_SW_TILE_Y)
    uint pageSlotBytes;   // PageCache slot byte stride
    uint pageCount;       // pageToSlot array capacity bound
    uint dagNodeCount;    // MpAssetHeader::dagNodeCount — DAG buffer bound
    uint rasterClassBufferBindlessIndex; // M4.4: u32 rasterClass per drawSlot
    uint pageFirstDagNodeBufferBindlessIndex; // M4.5: pageId -> firstDagNodeIdx
};
[[vk::push_constant]] PushBlock pc;

// --- Camera load (matches mp_cluster_cull / mp_raster.mesh pattern) --------
CameraData loadCamera(uint slot) {
    StructuredBuffer<float4> buf = g_buffers[NonUniformResourceIndex(slot)];
    CameraData cam;
    cam.view         = transpose(float4x4(buf[0],  buf[1],  buf[2],  buf[3]));
    cam.proj         = transpose(float4x4(buf[4],  buf[5],  buf[6],  buf[7]));
    cam.viewProj     = transpose(float4x4(buf[8],  buf[9],  buf[10], buf[11]));
    cam.prevViewProj = transpose(float4x4(buf[12], buf[13], buf[14], buf[15]));
    cam.invViewProj  = transpose(float4x4(buf[16], buf[17], buf[18], buf[19]));
    cam.worldPos     = buf[20];
    return cam;
}

// --- Indirect-draw read (mirrors mp_raster.task.hlsl::loadClusterIdx) -----
// The cull pass writes 16-byte header + 16-byte cmds:
//   offset 0 : count u32 + 3 pad
//   offset 16: cmd[0] = {groupCountX, groupCountY, groupCountZ, clusterIdx}
// We dispatch one workgroup per command; gid.x is the command index.
uint loadClusterIdx(uint cmdIndex) {
    RWByteAddressBuffer ind = g_rwBuffers[NonUniformResourceIndex(pc.indirectBufferBindlessIndex)];
    const uint base = 16u + cmdIndex * 16u;  // 16-byte header + stride 16
    return ind.Load(base + 12u);
}

// --- Dag node load — pageId is packed in m2.w low 24 bits ------------------
// MpDagNode stride is 5 float4 (M4: widened to carry maxError /
// parentMaxError; M4-fix: widened further to carry parentCenter as the
// group-coherent SSE LOD anchor — see mp_cluster_cull.comp.hlsl::loadDagNode).
// The binner only needs pageId, which still lives in m2.w.
uint loadPageIdForCluster(uint clusterIdx) {
    if (clusterIdx >= pc.dagNodeCount) return 0xFFFFFFFFu;
    StructuredBuffer<float4> buf = g_buffers[NonUniformResourceIndex(pc.dagBufferBindlessIndex)];
    const uint base = clusterIdx * 5u;
    float4 m2 = buf[base + 2u];
    const uint packed = asuint(m2.w);
    return packed & 0x00FFFFFFu;
}

// --- First-DAG-node lookup (M4.5 multi-cluster page support) --------------
// Mirrors mp_raster.task.hlsl::firstDagNodeIdxForPage. Returns the GLOBAL
// DAG index of the page's first cluster; local cluster index is the
// caller's `globalDagIdx - firstDagIdx`.
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

// --- Resident slot lookup (mirrors mp_raster.task.hlsl::slotForPage) ------
uint slotForPage(uint pageId) {
    if (pageId >= pc.pageCount) return 0xFFFFFFFFu;
    StructuredBuffer<float4> buf = g_buffers[NonUniformResourceIndex(pc.pageToSlotBufferBindlessIndex)];
    const uint float4Idx = pageId >> 2u;
    const uint component = pageId & 3u;
    uint4 packed = asuint(buf[float4Idx]);
    uint slot;
    if      (component == 0u) slot = packed.x;
    else if (component == 1u) slot = packed.y;
    else if (component == 2u) slot = packed.z;
    else                      slot = packed.w;
    return slot;
}

// --- ClusterOnDisk reader (mirrors mp_raster.mesh.hlsl) -------------------
struct ClusterFields {
    uint vertexCount;
    uint triangleCount;
    uint vertexOffsetInBlob;
    uint triangleOffsetInBlob;
};

ClusterFields loadClusterFields(uint pageByteOffset, uint clusterOnDiskOff) {
    RWByteAddressBuffer pageBuf = g_rwBuffers[NonUniformResourceIndex(pc.pageCacheBufferBindlessIndex)];
    const uint base = pageByteOffset + clusterOnDiskOff;
    uint4 first4 = pageBuf.Load4(base);
    ClusterFields c;
    c.vertexCount          = first4.x;
    c.triangleCount        = first4.y;
    c.vertexOffsetInBlob   = first4.z;
    c.triangleOffsetInBlob = first4.w;
    return c;
}

uint pageVertexBlobStart(uint pageByteOffset) {
    RWByteAddressBuffer pageBuf = g_rwBuffers[NonUniformResourceIndex(pc.pageCacheBufferBindlessIndex)];
    uint clusterCount = pageBuf.Load(pageByteOffset + 0u);
    if (clusterCount > 4096u) clusterCount = 4096u;
    return pageByteOffset + MP_PAGE_PAYLOAD_HEADER_BYTES
         + clusterCount * MP_CLUSTER_ON_DISK_STRIDE;
}

// Multi-cluster pages concatenate vertex blobs back-to-back; we must sum
// every ClusterOnDisk entry's vertexCount, not just this cluster's. The
// clusterCount clamp matches the reader's kMaxClustersPerPage ceiling
// (MpAssetReader.cpp) and must equal the clamp in pageVertexBlobStart so
// the two helpers agree on where the ClusterOnDisk table ends.
uint pageTriangleBlobStart(uint pageByteOffset) {
    RWByteAddressBuffer pageBuf = g_rwBuffers[NonUniformResourceIndex(pc.pageCacheBufferBindlessIndex)];
    uint clusterCount = pageBuf.Load(pageByteOffset + 0u);
    if (clusterCount > 4096u) clusterCount = 4096u;
    const uint vertexBlobStart = pageByteOffset + MP_PAGE_PAYLOAD_HEADER_BYTES
                               + clusterCount * MP_CLUSTER_ON_DISK_STRIDE;
    uint totalVertexBytes = 0u;
    for (uint ci = 0u; ci < clusterCount; ++ci) {
        const uint cAddr = pageByteOffset + MP_PAGE_PAYLOAD_HEADER_BYTES
                         + ci * MP_CLUSTER_ON_DISK_STRIDE;
        totalVertexBytes += pageBuf.Load(cAddr) * 32u;
    }
    return vertexBlobStart + totalVertexBytes;
}

float3 loadVertexPos(uint vertexBlobStart, uint vertexIndexInBlob) {
    RWByteAddressBuffer pageBuf = g_rwBuffers[NonUniformResourceIndex(pc.pageCacheBufferBindlessIndex)];
    const uint vertexAddr = vertexBlobStart + vertexIndexInBlob * 32u;
    uint3 bits = pageBuf.Load3(vertexAddr);
    const float3 raw = float3(asfloat(bits.x), asfloat(bits.y), asfloat(bits.z));
    // Match mp_raster.mesh.hlsl and sw_raster.comp.hlsl. Binning needs
    // the same rotated positions as raster or tile AABBs will target
    // the wrong tiles on screen.
    return float3(-raw.z, raw.y, raw.x);
}

uint3 loadTriangleIndices(uint triangleBlobStart, uint clusterTriOffsetBytes, uint triIndex) {
    RWByteAddressBuffer pageBuf = g_rwBuffers[NonUniformResourceIndex(pc.pageCacheBufferBindlessIndex)];
    const uint byteOffset  = triangleBlobStart + clusterTriOffsetBytes + triIndex * 3u;
    const uint wordOffset  = byteOffset & ~3u;
    const uint shift       = (byteOffset & 3u) * 8u;
    uint word0 = pageBuf.Load(wordOffset);
    uint word1 = pageBuf.Load(wordOffset + 4u);
    uint bits;
    if (shift == 0u) {
        bits = word0;
    } else {
        bits = (word0 >> shift) | (word1 << (32u - shift));
    }
    return uint3(bits & 0xFFu, (bits >> 8u) & 0xFFu, (bits >> 16u) & 0xFFu);
}

// --- Pack clusterIdx + triIdx into a 32-bit triRef -------------------------
// Matches vis-pack v2 layout (mp_vis_pack.hlsl): cluster in bits 29..7, tri
// in bits 6..0. The rasterClass field (bits 31..30) is left as zero; M4.3's
// fragment pass OR's in kMpRasterClassSw when building the full 64-bit
// packed vis value.
uint packTriRef(uint clusterIdx, uint triIdx) {
    return ((clusterIdx & kMpClusterMask) << kMpClusterShift)
         | (triIdx     & kMpVisTriMask);
}

// --- Bin a triangle into all tiles it overlaps -----------------------------
void binTriangleToTiles(int minX, int minY, int maxX, int maxY,
                        uint clusterIdx, uint triIdx) {
    RWByteAddressBuffer tileCount   = g_rwBuffers[NonUniformResourceIndex(pc.tileBinCountBindlessIndex)];
    RWByteAddressBuffer tileEntries = g_rwBuffers[NonUniformResourceIndex(pc.tileBinEntriesBindlessIndex)];
    RWByteAddressBuffer spill       = g_rwBuffers[NonUniformResourceIndex(pc.spillBufferBindlessIndex)];
    RWByteAddressBuffer spillHeads  = g_rwBuffers[NonUniformResourceIndex(pc.spillHeadsBufferBindlessIndex)];

    const int tileMinX = minX / (int)MP_SW_TILE_X;
    const int tileMinY = minY / (int)MP_SW_TILE_Y;
    const int tileMaxX = maxX / (int)MP_SW_TILE_X;
    const int tileMaxY = maxY / (int)MP_SW_TILE_Y;

    // Clamp to the tile grid. Caller has already clipped pixel AABB to the
    // viewport so these are tight; the clamp is defence-in-depth against
    // a triangle whose clipped AABB spans the edge.
    const int tx0 = max(tileMinX, 0);
    const int ty0 = max(tileMinY, 0);
    const int tx1 = min(tileMaxX, (int)pc.tilesX - 1);
    const int ty1 = min(tileMaxY, (int)pc.tilesY - 1);
    if (tx1 < tx0 || ty1 < ty0) return;

    const uint triRef = packTriRef(clusterIdx, triIdx);

    for (int ty = ty0; ty <= ty1; ++ty) {
        for (int tx = tx0; tx <= tx1; ++tx) {
            const uint tileIdx = (uint)ty * pc.tilesX + (uint)tx;

            uint oldCount;
            tileCount.InterlockedAdd(tileIdx * 4u, 1u, oldCount);

            if (oldCount < MP_SW_TILE_BIN_CAP) {
                // Fits in the fixed per-tile bin.
                const uint entryByteOffset =
                    (tileIdx * MP_SW_TILE_BIN_CAP + oldCount) * 4u;
                tileEntries.Store(entryByteOffset, triRef);
            } else {
                // Overflow routes to a per-tile spill linked list. Each
                // spill slot stores {u32 nextSlot, u32 triRef}. We reserve
                // a slot via InterlockedAdd on the spill counter, then
                // atomicExchange the spill head for this tile to get the
                // prior head and install our slot as the new head. Readers
                // walk the list head -> next -> next -> UINT32_MAX.
                uint slot;
                spill.InterlockedAdd(0u, 1u, slot);
                if (slot < MP_SW_SPILL_CAP) {
                    uint prevHead;
                    spillHeads.InterlockedExchange(tileIdx * 4u, slot, prevHead);
                    // Spill entry layout starts at byte 8 (skip the 8-byte
                    // {count, dropped} header). Each slot is 8 bytes:
                    // {u32 nextSlot, u32 triRef}.
                    const uint slotByteOff = 8u + slot * 8u;
                    spill.Store2(slotByteOff, uint2(prevHead, triRef));
                } else {
                    // Truly out of spill capacity — surface in diagnostics.
                    uint prevDropped;
                    spill.InterlockedAdd(4u, 1u, prevDropped);
                }
            }
        }
    }
}

// --- M4.4 dispatcher gate (groupshared broadcast) -------------------------
// The cull pass tags each drawSlot with a u32 rasterClass (0=HW, 1=SW).
// Thread 0 loads the tag; if the cluster is HW-classified the whole
// workgroup early-returns (no binning work). Using groupshared + a
// GroupMemoryBarrierWithGroupSync avoids 64 wasted LOAD ops per group —
// only lane 0 touches the SSBO.
groupshared uint s_rasterClass;
// M4.5 Phase 4 HIGH fix: thread-0 detects corrupt asset (firstDagIdx >
// globalClusterIdx) and broadcasts a skip sentinel so the whole workgroup
// early-returns instead of silently binning cluster 0.
groupshared uint s_skip;

// --- Main -----------------------------------------------------------------
[numthreads(MP_SW_TILE_THREADS, 1, 1)]
void CSMain(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    // M4.4: gate on rasterClass before any page-cache reads. Thread 0 loads
    // and broadcasts via groupshared; other threads wait at the barrier.
    // M4.5 Phase 4: thread 0 also pre-computes the corrupt-asset skip flag
    // in parallel so the whole workgroup can bail on one barrier rather
    // than re-computing per-lane.
    if (gtid.x == 0u) {
        RWByteAddressBuffer classBuf = g_rwBuffers[NonUniformResourceIndex(pc.rasterClassBufferBindlessIndex)];
        s_rasterClass = classBuf.Load(gid.x * 4u);
        const uint clusterIdx0 = loadClusterIdx(gid.x);
        const uint pageId0     = loadPageIdForCluster(clusterIdx0);
        const uint firstDagIdx0 = firstDagNodeIdxForPage(pageId0);
        s_skip = (clusterIdx0 < firstDagIdx0) ? 1u : 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    if (s_rasterClass != kMpRasterClassSw || s_skip != 0u) return;

    const uint clusterIdx = loadClusterIdx(gid.x);
    const uint pageId     = loadPageIdForCluster(clusterIdx);
    const uint slotIndex  = slotForPage(pageId);

    // Non-resident page — the HW raster path skips via DispatchMesh(0); here
    // we skip via an early return. The cull pass has already emitted a page
    // request for this pageId, so a subsequent frame will find it resident.
    if (slotIndex == 0xFFFFFFFFu) return;

    // M4.5 multi-cluster page resolution: recover page-local cluster index
    // via pageFirstDagNodeBuffer. `clusterIdx` is the global DAG index
    // written by the cull pass into the indirect-draw cmd. Underflow was
    // already detected by thread 0 and broadcast via s_skip.
    const uint firstDagIdx     = firstDagNodeIdxForPage(pageId);
    const uint localClusterIdx = clusterIdx - firstDagIdx;
    const uint pageByteOffset  = slotIndex * pc.pageSlotBytes;
    const uint clusterOnDiskOffsetInPage =
        MP_PAGE_PAYLOAD_HEADER_BYTES + localClusterIdx * MP_CLUSTER_ON_DISK_STRIDE;

    const ClusterFields cluster =
        loadClusterFields(pageByteOffset, clusterOnDiskOffsetInPage);

    uint vertexCount   = cluster.vertexCount;
    uint triangleCount = cluster.triangleCount;
    if (triangleCount > MP_SW_MAX_TRIS) triangleCount = MP_SW_MAX_TRIS;
    if (vertexCount == 0u) triangleCount = 0u;
    if (triangleCount == 0u) return;

    const uint vertexBlobStart   = pageVertexBlobStart(pageByteOffset);
    const uint triangleBlobStart = pageTriangleBlobStart(pageByteOffset);
    const uint vertexBaseInBlob  = cluster.vertexOffsetInBlob / 32u;

    const CameraData cam = loadCamera(pc.cameraSlot);

    const float viewW = (float)pc.viewportWidth;
    const float viewH = (float)pc.viewportHeight;

    // Each thread handles triangles gtid.x, gtid.x + 64, ... up to
    // triangleCount. With MP_SW_MAX_TRIS = 128 and MP_SW_TILE_THREADS = 64,
    // each thread touches at most 2 triangles.
    for (uint triIdx = gtid.x; triIdx < triangleCount; triIdx += MP_SW_TILE_THREADS) {
        const uint3 idx = loadTriangleIndices(triangleBlobStart,
                                              cluster.triangleOffsetInBlob,
                                              triIdx);

        // Clamp indices defensively to the cluster's vertex range.
        const uint i0 = min(idx.x, vertexCount - 1u);
        const uint i1 = min(idx.y, vertexCount - 1u);
        const uint i2 = min(idx.z, vertexCount - 1u);

        const float3 p0 = loadVertexPos(vertexBlobStart, vertexBaseInBlob + i0);
        const float3 p1 = loadVertexPos(vertexBlobStart, vertexBaseInBlob + i1);
        const float3 p2 = loadVertexPos(vertexBlobStart, vertexBaseInBlob + i2);

        // World -> clip. No instance transform (micropoly is static geometry
        // in M3/M4; Principle 5).
        const float4 c0 = mul(cam.viewProj, float4(p0, 1.0f));
        const float4 c1 = mul(cam.viewProj, float4(p1, 1.0f));
        const float4 c2 = mul(cam.viewProj, float4(p2, 1.0f));

        // Reject triangles fully behind the camera (w <= 0). A proper
        // clipper would split straddling triangles; M4.2's binning stage
        // conservatively drops them and leaves near-plane straddlers for
        // M4.3's per-pixel edge test to handle as "off-tile" after
        // viewport clipping. This matches the spirit of the plan's
        // "degenerate rejection" rule without adding a full Sutherland-
        // Hodgman stage that M4.3 doesn't need yet.
        if (c0.w <= 0.0f || c1.w <= 0.0f || c2.w <= 0.0f) continue;

        const float invW0 = 1.0f / c0.w;
        const float invW1 = 1.0f / c1.w;
        const float invW2 = 1.0f / c2.w;

        // Clip -> NDC.
        const float2 ndc0 = float2(c0.x * invW0, c0.y * invW0);
        const float2 ndc1 = float2(c1.x * invW1, c1.y * invW1);
        const float2 ndc2 = float2(c2.x * invW2, c2.y * invW2);

        // NDC -> pixel space. Vulkan NDC has y-down after the viewport
        // flip; we match the engine's proj matrix by computing pixel
        // coords with (ndc.y * 0.5 + 0.5) * height (no inversion).
        const float2 pix0 = float2((ndc0.x * 0.5f + 0.5f) * viewW,
                                   (ndc0.y * 0.5f + 0.5f) * viewH);
        const float2 pix1 = float2((ndc1.x * 0.5f + 0.5f) * viewW,
                                   (ndc1.y * 0.5f + 0.5f) * viewH);
        const float2 pix2 = float2((ndc2.x * 0.5f + 0.5f) * viewW,
                                   (ndc2.y * 0.5f + 0.5f) * viewH);

        // Signed area in pixel space. PERF-DIAG: only skip truly
        // degenerate triangles (area==0). HW raster uses cullMode=NONE,
        // so we don't backface-cull here either — otherwise SW silently
        // drops 50% of triangles vs HW parity. If the sign is negative,
        // swap two vertex indices in shadePixel to keep the edge test
        // working consistently.
        const float edge = (pix1.x - pix0.x) * (pix2.y - pix0.y)
                         - (pix1.y - pix0.y) * (pix2.x - pix0.x);
        if (edge == 0.0f) continue;

        // Pixel-space AABB clipped to viewport. We use floor for min and
        // ceil for max so the sampled-pixel grid (integer pixel centres)
        // is covered conservatively.
        float fMinX = min(pix0.x, min(pix1.x, pix2.x));
        float fMinY = min(pix0.y, min(pix1.y, pix2.y));
        float fMaxX = max(pix0.x, max(pix1.x, pix2.x));
        float fMaxY = max(pix0.y, max(pix1.y, pix2.y));

        int minX = (int)floor(fMinX);
        int minY = (int)floor(fMinY);
        int maxX = (int)ceil(fMaxX)  - 1;
        int maxY = (int)ceil(fMaxY)  - 1;

        minX = max(minX, 0);
        minY = max(minY, 0);
        maxX = min(maxX, (int)pc.viewportWidth  - 1);
        maxY = min(maxY, (int)pc.viewportHeight - 1);
        if (maxX < minX || maxY < minY) continue;

        binTriangleToTiles(minX, minY, maxX, maxY, clusterIdx, triIdx);
    }
}
