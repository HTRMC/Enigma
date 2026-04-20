// sw_raster.comp.hlsl
// =====================
// M4.3 — SW rasteriser fragment pass. One workgroup = one 8x8 screen-space
// tile (matching the tile size M4.2's binning shader produces). One thread
// = one pixel in that tile. Thread iterates the tile's bin + spill entries,
// performs per-pixel point-in-triangle tests, and writes through the
// bindless R64_UINT vis image via InterlockedMax.
//
// Reverse-Z reminder: the engine-wide convention (see mp_vis_pack.hlsl
// banner + M3.4 Phase 4 HW raster fix) is that NEAREST samples pack to
// LARGER uint64, so the correct atomic op on the vis image is
// InterlockedMax. The M4 plan text at line 457 says InterlockedMin; that
// predates the M3.4 fix. We match the engine convention so HW + SW
// samples can co-exist on the same vis image.
//
// Workgroup count: direct dispatch, ceil(W/8) * ceil(H/8) * 1 — extent-
// driven. Groups that step past the viewport early-out per pixel.
//
// Performance note (deferred to M4.6): this implementation is naive. One
// thread per pixel iterates all the tile's triangles sequentially; warp
// divergence is high when a tile has a mix of covered + uncovered
// pixels. The plan's ≤1.5 ms SW-total target is M4.5 acceptance, not
// gated here. Tile-shared-memory triangle caching is a natural M4.6
// optimization, not required for correctness in M4.3.
//
// DXC -Zpc caveat: float4x4(v0..v3) takes COLUMN vectors under -Zpc -spirv,
// so matrix SSBO loads use transpose(float4x4(...)). The bin shader's
// loadCamera mirrors this; we mirror it here too.
//
// TODO(M5): dedupe the loadCamera / loadPageIdForCluster / slotForPage /
// ClusterFields / vertex / triangle readers into a shared include
// (shaders/micropoly/sw_raster_common.hlsl). For M4.3 we copy verbatim so
// spirv_diff keeps the existing binning shader goldens byte-identical
// without a rebaseline.

#include "../common.hlsl"
#include "mp_vis_pack.hlsl"
#include "mp_cluster_layout.hlsl"

// --- Tile + bin sizing constants (mirror sw_raster_bin.comp.hlsl) ---------
#define MP_SW_TILE_X       8u
#define MP_SW_TILE_Y       8u
// 1024 now that overflow routes to a per-tile spill linked list
// (sw_raster_bin's binTriangleToTiles). No more non-deterministic
// drops at the BIN_CAP boundary — all overflowed triangles are kept
// and walked at raster time via the spill head chain. 1024 * 4B *
// 14400 tiles ≈ 59 MiB for a 1280x720 grid.
#define MP_SW_TILE_BIN_CAP 1024u
// Raised from 65536 to 1048576 (1M entries, 8 MB of SSBO) so a
// small-on-screen BMW — where thousands of triangles concentrate into a
// handful of tiles and blow past the per-tile cap — can spill without
// dropping. Drops produced 8x8-pixel "tile-shaped holes" that got worse
// as the car shrank.
// Raised 1M → 16M (128 MB). 1M was still hitting on dense BMW tiles:
// at ~5 tile overlaps per triangle × 16k clusters × ~80 tris/cluster,
// global bin-entry pressure is ~6M, of which the densest 20% spill.
// Once the global counter exceeded 1M, subsequent entries dropped,
// and because atomic-add ordering is non-deterministic per frame,
// different triangles dropped each frame → tile-shaped holes that
// flickered. 16M covers the worst BMW close-up without touching the
// VRAM budget.
#define MP_SW_SPILL_CAP    16777216u

// --- Bindless resource arrays ---------------------------------------------
[[vk::binding(2, 0)]]
StructuredBuffer<float4> g_buffers[] : register(t0, space1);

[[vk::binding(5, 0)]]
RWByteAddressBuffer g_rwBuffers[] : register(u1, space0);

// R64_UINT storage image array — matches mp_raster.mesh.hlsl PSMain so
// both HW + SW raster write to the same bindless slot via InterlockedMax.
[[vk::binding(1, 0)]]
RWTexture2D<uint64_t> g_visImages[] : register(u0, space0);

// --- Push constants --------------------------------------------------------
// Matches MicropolySwRasterPushBlock in MicropolySwRasterPass.cpp. 16 u32s
// = 64 B — identical shape to MicropolySwRasterBinPushBlock but distinct
// since this shader reads the bin SSBOs rather than writing them.
struct PushBlock {
    uint tileBinCountBindless;       // RWByteAddressBuffer u32 * numTiles
    uint tileBinEntriesBindless;     // RWByteAddressBuffer u32 * numTiles * BIN_CAP
    uint spillBufferBindless;        // RWByteAddressBuffer header + spill array
    uint dagBufferBindless;          // StructuredBuffer<MpDagNode> as float4
    uint pageToSlotBindless;         // pageId -> slotIndex or UINT32_MAX
    uint pageCacheBindless;          // entire page pool
    uint indirectBufferBindless;     // for cluster count / workgroup derivation
    uint visImage64Bindless;         // RWTexture2D<uint64_t>
    uint cameraSlot;                 // StructuredBuffer<float4> CameraData
    uint screenWidth;                // pixels
    uint screenHeight;               // pixels
    uint tilesX;                     // ceil(screenWidth / MP_SW_TILE_X)
    uint pageSlotBytes;              // PageCache slot byte stride
    uint dagNodeCount;               // bounds check
    uint pageCount;                  // bounds check
    uint pageFirstDagNodeBufferBindlessIndex; // M4.5: pageId -> firstDagNodeIdx
    uint spillHeadsBufferBindless;   // M4.6: per-tile spill linked-list heads
};
[[vk::push_constant]] PushBlock pc;

// --- Camera load (matches sw_raster_bin / mp_raster.mesh pattern) ---------
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

// --- Dag node load — pageId is packed in m2.w low 24 bits ------------------
uint loadPageIdForCluster(uint clusterIdx) {
    if (clusterIdx >= pc.dagNodeCount) return 0xFFFFFFFFu;
    StructuredBuffer<float4> buf = g_buffers[NonUniformResourceIndex(pc.dagBufferBindless)];
    const uint base = clusterIdx * 3u;
    float4 m2 = buf[base + 2u];
    const uint packed = asuint(m2.w);
    return packed & 0x00FFFFFFu;
}

// --- First-DAG-node lookup (M4.5 multi-cluster page support) --------------
// Mirrors sw_raster_bin::firstDagNodeIdxForPage.
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

// --- Resident slot lookup (mirrors sw_raster_bin::slotForPage) -------------
uint slotForPage(uint pageId) {
    if (pageId >= pc.pageCount) return 0xFFFFFFFFu;
    StructuredBuffer<float4> buf = g_buffers[NonUniformResourceIndex(pc.pageToSlotBindless)];
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

// --- ClusterOnDisk reader (mirrors sw_raster_bin) -------------------------
struct ClusterFields {
    uint vertexCount;
    uint triangleCount;
    uint vertexOffsetInBlob;
    uint triangleOffsetInBlob;
};

ClusterFields loadClusterFields(uint pageByteOffset, uint clusterOnDiskOff) {
    RWByteAddressBuffer pageBuf = g_rwBuffers[NonUniformResourceIndex(pc.pageCacheBindless)];
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
    RWByteAddressBuffer pageBuf = g_rwBuffers[NonUniformResourceIndex(pc.pageCacheBindless)];
    uint clusterCount = pageBuf.Load(pageByteOffset + 0u);
    if (clusterCount > 4096u) clusterCount = 4096u;
    return pageByteOffset + MP_PAGE_PAYLOAD_HEADER_BYTES
         + clusterCount * MP_CLUSTER_ON_DISK_STRIDE;
}

// Multi-cluster pages concatenate vertex blobs back-to-back; sum every
// ClusterOnDisk entry's vertexCount, not just this cluster's. Clamped
// to 4096 — the reader's kMaxClustersPerPage ceiling (MpAssetReader.cpp).
// MUST match pageVertexBlobStart's clamp or the offset math for the two
// helpers drifts on any page whose clusterCount > 64: pageVertexBlobStart
// accepts up to 4096 and this helper used to cap at 64, silently pointing
// the triangle blob tens of kilobytes into the middle of the ClusterOnDisk
// table.
uint pageTriangleBlobStart(uint pageByteOffset) {
    RWByteAddressBuffer pageBuf = g_rwBuffers[NonUniformResourceIndex(pc.pageCacheBindless)];
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
    RWByteAddressBuffer pageBuf = g_rwBuffers[NonUniformResourceIndex(pc.pageCacheBindless)];
    const uint vertexAddr = vertexBlobStart + vertexIndexInBlob * 32u;
    uint3 bits = pageBuf.Load3(vertexAddr);
    const float3 raw = float3(asfloat(bits.x), asfloat(bits.y), asfloat(bits.z));
    // Engine-wide -90° Y correction (glb +X → physics +Z). Matches the
    // per-node rest-transform rotation in Application.cpp and the
    // matching correction in mp_raster.mesh.hlsl. Without it the mpa
    // geometry sits 90° off vs the glTF, losing depth inside the glTF
    // envelope.
    return float3(-raw.z, raw.y, raw.x);
}

uint3 loadTriangleIndices(uint triangleBlobStart, uint clusterTriOffsetBytes, uint triIndex) {
    RWByteAddressBuffer pageBuf = g_rwBuffers[NonUniformResourceIndex(pc.pageCacheBindless)];
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

// --- Project a triangle's 3 vertices to pixel space ------------------------
// Mirrors sw_raster_bin.comp.hlsl's projection exactly so HW/SW raster see
// the same projected triangle. Returns pixel-space positions + per-vertex
// clip-space depth (z/w) packed alongside, plus w (for 1/w interpolation)
// and a boolean 'valid' bit.
struct ProjectedTri {
    float2 pix0;
    float2 pix1;
    float2 pix2;
    // Clip-space z/w for each vertex (reverse-Z: near=1, far=0).
    float  z0;
    float  z1;
    float  z2;
    bool   valid;  // false when any w<=0, degenerate, or back-facing
};

ProjectedTri projectTri(CameraData cam, float3 p0, float3 p1, float3 p2,
                        float viewW, float viewH) {
    ProjectedTri r;
    r.valid = false;

    const float4 c0 = mul(cam.viewProj, float4(p0, 1.0f));
    const float4 c1 = mul(cam.viewProj, float4(p1, 1.0f));
    const float4 c2 = mul(cam.viewProj, float4(p2, 1.0f));

    if (c0.w <= 0.0f || c1.w <= 0.0f || c2.w <= 0.0f) return r;

    const float invW0 = 1.0f / c0.w;
    const float invW1 = 1.0f / c1.w;
    const float invW2 = 1.0f / c2.w;

    const float2 ndc0 = float2(c0.x * invW0, c0.y * invW0);
    const float2 ndc1 = float2(c1.x * invW1, c1.y * invW1);
    const float2 ndc2 = float2(c2.x * invW2, c2.y * invW2);

    r.pix0 = float2((ndc0.x * 0.5f + 0.5f) * viewW,
                    (ndc0.y * 0.5f + 0.5f) * viewH);
    r.pix1 = float2((ndc1.x * 0.5f + 0.5f) * viewW,
                    (ndc1.y * 0.5f + 0.5f) * viewH);
    r.pix2 = float2((ndc2.x * 0.5f + 0.5f) * viewW,
                    (ndc2.y * 0.5f + 0.5f) * viewH);

    r.z0 = c0.z * invW0;
    r.z1 = c1.z * invW1;
    r.z2 = c2.z * invW2;

    // Signed area (2*triangle). Reject degenerate, but swap winding when
    // back-facing (edge < 0) so the downstream edge-function test still
    // treats the triangle as front-facing. HW raster uses cullMode=NONE,
    // so we must not drop back-facing tris here or SW silently under-
    // covers vs HW.
    const float edge = (r.pix1.x - r.pix0.x) * (r.pix2.y - r.pix0.y)
                     - (r.pix1.y - r.pix0.y) * (r.pix2.x - r.pix0.x);
    if (edge == 0.0f) return r;
    if (edge < 0.0f) {
        const float2 tmpPix = r.pix1; r.pix1 = r.pix2; r.pix2 = tmpPix;
        const float  tmpZ   = r.z1;   r.z1   = r.z2;   r.z2   = tmpZ;
    }

    r.valid = true;
    return r;
}

// --- Load cluster + project -> iterate a single triRef -------------------
// Re-does the clusterIdx -> page -> vertex lookups that M4.2's binning
// shader performed (and then projects triangle i from that cluster). The
// cost is O(trisInBin) loads per pixel — acceptable for the M4.3 perf
// budget; M4.6 can hoist the projection into shared memory if needed.
//
// Returns false if the cluster isn't resident (shouldn't happen for
// entries the binning pass emitted, but defensive: if a page was evicted
// between bin + raster the binning slot refs are stale; skip safely).
struct TriProjection {
    ProjectedTri proj;
    bool         ok;
};

TriProjection loadAndProject(uint clusterIdx, uint triIdxInCluster,
                             CameraData cam, float viewW, float viewH) {
    TriProjection result;
    result.ok = false;
    result.proj.valid = false;

    const uint pageId    = loadPageIdForCluster(clusterIdx);
    const uint slotIndex = slotForPage(pageId);
    if (slotIndex == 0xFFFFFFFFu) return result;

    // M4.5 multi-cluster page resolution: recover the page-local cluster
    // index via pageFirstDagNodeBuffer. `clusterIdx` is the global DAG
    // index unpacked from the per-tile-bin triRef — see sw_raster_bin.comp
    // for the packing and mp_cluster_cull for the draw-slot assignment.
    const uint firstDagIdx = firstDagNodeIdxForPage(pageId);
    // M4.5 Phase 4 HIGH fix: on corrupt asset (firstDagIdx > clusterIdx)
    // skip this triangle entry entirely instead of silently returning
    // cluster 0's vertices. Caller sees !tp.ok and `continue`s.
    if (clusterIdx < firstDagIdx) return result;
    const uint localClusterIdx = clusterIdx - firstDagIdx;
    const uint pageByteOffset  = slotIndex * pc.pageSlotBytes;
    const uint clusterOnDiskOffsetInPage =
        MP_PAGE_PAYLOAD_HEADER_BYTES + localClusterIdx * MP_CLUSTER_ON_DISK_STRIDE;

    const ClusterFields cluster =
        loadClusterFields(pageByteOffset, clusterOnDiskOffsetInPage);

    uint vertexCount   = cluster.vertexCount;
    uint triangleCount = cluster.triangleCount;
    if (vertexCount == 0u || triangleCount == 0u) return result;
    if (triIdxInCluster >= triangleCount) return result;

    const uint vertexBlobStart   = pageVertexBlobStart(pageByteOffset);
    const uint triangleBlobStart = pageTriangleBlobStart(pageByteOffset);
    const uint vertexBaseInBlob  = cluster.vertexOffsetInBlob / 32u;

    const uint3 idx = loadTriangleIndices(triangleBlobStart,
                                           cluster.triangleOffsetInBlob,
                                           triIdxInCluster);
    const uint i0 = min(idx.x, vertexCount - 1u);
    const uint i1 = min(idx.y, vertexCount - 1u);
    const uint i2 = min(idx.z, vertexCount - 1u);

    const float3 p0 = loadVertexPos(vertexBlobStart, vertexBaseInBlob + i0);
    const float3 p1 = loadVertexPos(vertexBlobStart, vertexBaseInBlob + i1);
    const float3 p2 = loadVertexPos(vertexBlobStart, vertexBaseInBlob + i2);

    result.proj = projectTri(cam, p0, p1, p2, viewW, viewH);
    result.ok   = result.proj.valid;
    return result;
}

// --- Unpack bin entry: cluster in bits 29..7, tri in bits 6..0 ------------
void unpackTriRef(uint triRef, out uint clusterIdx, out uint triIdx) {
    clusterIdx = (triRef >> kMpClusterShift) & kMpClusterMask;
    triIdx     =  triRef                     & kMpVisTriMask;
}

// --- Per-pixel point-in-triangle + depth interpolation + atomic-max -------
// Edge-function rasterisation. Uses top-left fill rule via edge >= 0.0
// (not > 0.0 — triangles sharing an edge both own the fragment, which the
// atomic-max resolves by depth). Matches HW raster's standard convention.
void shadePixel(uint2 pixelCoord,
                ProjectedTri proj,
                uint clusterIdx,
                uint triIdxInCluster,
                uint visImageBindless) {
    // Pixel centre for sampling (half-pixel offset matches HW raster).
    const float2 p = float2((float)pixelCoord.x + 0.5f,
                            (float)pixelCoord.y + 0.5f);

    // Edge functions — signed twice-area of each sub-triangle. Front-
    // facing triangles have positive total area (filtered in projectTri),
    // so a pixel inside has all three edge functions >= 0.
    const float e0 = (proj.pix1.x - proj.pix0.x) * (p.y - proj.pix0.y)
                   - (proj.pix1.y - proj.pix0.y) * (p.x - proj.pix0.x);
    const float e1 = (proj.pix2.x - proj.pix1.x) * (p.y - proj.pix1.y)
                   - (proj.pix2.y - proj.pix1.y) * (p.x - proj.pix1.x);
    const float e2 = (proj.pix0.x - proj.pix2.x) * (p.y - proj.pix2.y)
                   - (proj.pix0.y - proj.pix2.y) * (p.x - proj.pix2.x);

    if (e0 < 0.0f || e1 < 0.0f || e2 < 0.0f) return;

    // Barycentrics: lambda_i = e_(i+1) / totalArea (cyclic). Derive
    // totalArea from the three edge sum (area of sub-triangles = area).
    const float area = e0 + e1 + e2;
    if (area <= 0.0f) return;  // defensive: co-linear / degenerate
    const float invArea = 1.0f / area;

    // The barycentric weight for vertex i is the edge opposite that
    // vertex. e0 is opposite vertex 2, e1 opposite vertex 0, e2 opposite
    // vertex 1. Interpolate clip-space z/w linearly in screen space (NDC
    // z is linear in pixel coords for a perspective-divided triangle;
    // this is the standard HW raster convention and matches what an
    // SV_Position.z fragment sees).
    const float w0 = e1 * invArea;   // weight for vertex 0
    const float w1 = e2 * invArea;   // weight for vertex 1
    const float w2 = e0 * invArea;   // weight for vertex 2

    const float depthF = proj.z0 * w0 + proj.z1 * w1 + proj.z2 * w2;

    // Reverse-Z: clamp to [0,1] defensively — a triangle straddling the
    // near plane could produce an out-of-range depth after w>0 gating.
    const float depthClamped = clamp(depthF, 0.0f, 1.0f);

    const uint64_t packed = PackMpVis64(asuint(depthClamped),
                                        kMpRasterClassSw,
                                        clusterIdx,
                                        triIdxInCluster);

    // Under reverse-Z (see mp_vis_pack.hlsl banner) LARGEST packed value
    // wins — that is the NEAREST sample. HW raster's PSMain uses the
    // same InterlockedMax so HW + SW samples co-exist on one vis image.
    InterlockedMax(g_visImages[NonUniformResourceIndex(visImageBindless)][pixelCoord],
                   packed);
}

// Groupshared batch cache. Each iteration, 64 threads cooperatively load+
// project 64 bin entries (one per thread), then all 64 threads rasterise
// each cached triangle against their pixel. Cuts the O(clusterCount) loop
// in pageTriangleBlobStart from "every thread, every entry" down to "every
// thread, every 64th entry". Full warp utilisation preserved: SIMD lanes
// do distinct work during the load phase and identical addresses during
// the shade phase (wave-coalesced reads from groupshared).
#define MP_SW_BATCH 64u  // must match threads per tile (MP_SW_TILE_X*MP_SW_TILE_Y)
groupshared ProjectedTri s_batchProj       [MP_SW_BATCH];
groupshared uint         s_batchClusterIdx [MP_SW_BATCH];
groupshared uint         s_batchTriIdx     [MP_SW_BATCH];
groupshared bool         s_batchOk         [MP_SW_BATCH];

// M4.6 spill linked-list walker scratch. Thread 0 publishes a batch of
// slot indices, the batch size, and the head pointer for the next
// iteration; the whole workgroup reads those after a GroupSync.
groupshared uint s_spillSlots      [MP_SW_BATCH];
groupshared uint s_batchSpillCount;
groupshared uint s_batchSpillHead;

// --- Main ------------------------------------------------------------------
[numthreads(MP_SW_TILE_X, MP_SW_TILE_Y, 1)]
void CSMain(uint3 dispatchId : SV_DispatchThreadID,
            uint3 groupId    : SV_GroupID,
            uint  groupIndex : SV_GroupIndex) {
    const uint2 pixelCoord = dispatchId.xy;

    // Step 2: viewport early-out. Groups on the right/bottom edge may
    // have threads past the rendered area; those threads skip the
    // atomic-max entirely.
    const bool inViewport = (pixelCoord.x < pc.screenWidth)
                         && (pixelCoord.y < pc.screenHeight);

    const uint tileIdx = groupId.y * pc.tilesX + groupId.x;

    // Step 3: load this tile's bin count. Clamp to BIN_CAP to match the
    // saturation semantics in sw_raster_bin (overflow goes to spill,
    // not past the fixed bin).
    RWByteAddressBuffer tileCount   = g_rwBuffers[NonUniformResourceIndex(pc.tileBinCountBindless)];
    RWByteAddressBuffer tileEntries = g_rwBuffers[NonUniformResourceIndex(pc.tileBinEntriesBindless)];
    RWByteAddressBuffer spill       = g_rwBuffers[NonUniformResourceIndex(pc.spillBufferBindless)];
    RWByteAddressBuffer spillHeads  = g_rwBuffers[NonUniformResourceIndex(pc.spillHeadsBufferBindless)];

    uint binCount = tileCount.Load(tileIdx * 4u);
    if (binCount > MP_SW_TILE_BIN_CAP) binCount = MP_SW_TILE_BIN_CAP;

    // M4.6 linked-list spill: each tile has its own head index. UINT32_MAX =
    // no overflow for this tile. We walk the chain at the end of the raster
    // loop below.
    const uint spillHeadInit = spillHeads.Load(tileIdx * 4u);

    // Whole-tile early exit — no triangles to shade here.
    if (binCount == 0u && spillHeadInit == 0xFFFFFFFFu) return;

    const CameraData cam = loadCamera(pc.cameraSlot);
    const float viewW = (float)pc.screenWidth;
    const float viewH = (float)pc.screenHeight;

    // Step 4: primary bin, batched. Each iteration cooperatively loads+
    // projects 64 entries into groupshared, then every thread rasterises
    // the cached batch against its pixel.
    for (uint base = 0u; base < binCount; base += MP_SW_BATCH) {
        const uint entryIdx = base + groupIndex;
        if (entryIdx < binCount) {
            const uint entryByteOffset = (tileIdx * MP_SW_TILE_BIN_CAP + entryIdx) * 4u;
            const uint triRef = tileEntries.Load(entryByteOffset);
            uint clusterIdx, triIdxInCluster;
            unpackTriRef(triRef, clusterIdx, triIdxInCluster);
            const TriProjection tp = loadAndProject(clusterIdx, triIdxInCluster,
                                                     cam, viewW, viewH);
            s_batchProj[groupIndex]       = tp.proj;
            s_batchClusterIdx[groupIndex] = clusterIdx;
            s_batchTriIdx[groupIndex]     = triIdxInCluster;
            s_batchOk[groupIndex]         = tp.ok;
        } else {
            s_batchOk[groupIndex] = false;
        }
        GroupMemoryBarrierWithGroupSync();

        if (inViewport) {
            const uint cacheCount = min(MP_SW_BATCH, binCount - base);
            for (uint k = 0u; k < cacheCount; ++k) {
                if (!s_batchOk[k]) continue;
                shadePixel(pixelCoord, s_batchProj[k],
                           s_batchClusterIdx[k], s_batchTriIdx[k],
                           pc.visImage64Bindless);
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // M4.6 spill linked-list walk. Each tile's head index was read at the
    // top of the function (spillHeadInit). The chain is walked in batches
    // of MP_SW_BATCH: thread 0 pulls up to MP_SW_BATCH slot indices from
    // the list and publishes them via groupshared s_spillSlots; all 64
    // threads then cooperatively load+project one triRef each, shade the
    // cache, and repeat until thread 0 reports spillHead==UINT32_MAX.
    //
    // Only thread 0 advances the chain because spill.Load2(slot) returns
    // {nextSlot, triRef} — the traversal is sequential. This is still
    // dramatically cheaper than the pre-M4.6 design (O(tiles * spillCount))
    // because each tile only visits its own spill entries; total spill-
    // traversal cost is O(totalSpillEntries) summed across tiles, not
    // multiplied.
    if (spillHeadInit != 0xFFFFFFFFu) {
        uint spillHead = spillHeadInit;
        while (true) {
            // Thread 0: pull up to MP_SW_BATCH slots off the chain.
            if (groupIndex == 0u) {
                s_batchSpillCount = 0u;
                uint head = spillHead;
                [unroll] for (uint i = 0u; i < MP_SW_BATCH; ++i) {
                    if (head == 0xFFFFFFFFu) break;
                    s_spillSlots[i] = head;
                    s_batchSpillCount += 1u;
                    // Fetch next pointer; triRef is reloaded below by
                    // whichever thread claims this slot. Load only the
                    // first u32 (next pointer) here to keep thread 0's
                    // serial work small.
                    head = spill.Load(8u + head * 8u);
                    if (head >= MP_SW_SPILL_CAP) head = 0xFFFFFFFFu; // defensive
                }
                s_batchSpillHead = head;
            }
            GroupMemoryBarrierWithGroupSync();

            const uint batchCount = s_batchSpillCount;
            if (batchCount == 0u) break;

            // All threads: cooperatively load+project one slot each.
            s_batchOk[groupIndex] = false;
            if (groupIndex < batchCount) {
                const uint slot = s_spillSlots[groupIndex];
                const uint2 pair = spill.Load2(8u + slot * 8u);
                uint clusterIdx, triIdxInCluster;
                unpackTriRef(pair.y, clusterIdx, triIdxInCluster);
                const TriProjection tp = loadAndProject(clusterIdx, triIdxInCluster,
                                                         cam, viewW, viewH);
                s_batchProj[groupIndex]       = tp.proj;
                s_batchClusterIdx[groupIndex] = clusterIdx;
                s_batchTriIdx[groupIndex]     = triIdxInCluster;
                s_batchOk[groupIndex]         = tp.ok;
            }
            GroupMemoryBarrierWithGroupSync();

            if (inViewport) {
                for (uint k = 0u; k < batchCount; ++k) {
                    if (!s_batchOk[k]) continue;
                    shadePixel(pixelCoord, s_batchProj[k],
                               s_batchClusterIdx[k], s_batchTriIdx[k],
                               pc.visImage64Bindless);
                }
            }
            GroupMemoryBarrierWithGroupSync();

            // Advance head for next iteration (set by thread 0 above).
            spillHead = s_batchSpillHead;
            if (batchCount < MP_SW_BATCH) break;  // chain exhausted
        }
    }
}
