// mp_cluster_cull.comp.hlsl
// ==========================
// Per-frame compute pass that walks the cluster DAG (streamed through the
// PageCache) and for each cluster:
//
//   1. Runs an LOD gate — in M3.2 this is simplified to "leaf only";
//      non-leaf clusters defer to their children (future M4 work).
//   2. Emits a page-streaming request if the cluster's page is not yet
//      resident. Requests go through shaders/micropoly/page_request_emit.hlsl.
//   3. View-space frustum cull — bounding-sphere test against the five
//      frustum planes reconstructed on-the-fly from CameraData.proj
//      (matches gpu_cull.comp.hlsl's proven pattern; avoids the
//      Gribb-Hartmann / column-vs-row -Zpc bug).
//   4. Normal-cone backface cull — cluster cone vs view direction. Rejected
//      when the entire cluster is guaranteed to be backfacing.
//   5. HiZ occlusion cull — project the bounding sphere to screen space,
//      sample HiZ at the matching mip, reject when the sphere is entirely
//      behind the depth front.
//   6. Survivors are appended to an indirect draw buffer; M3.3 will issue
//      the task+mesh dispatches over that buffer.
//
// Dispatch: ceil(totalClusterCount / 64) workgroups of 64 threads each.
// Depth convention: reverse-Z (far = 0, near = 1) — matches the engine's
// GpuCullPass conventions.

#include "../common.hlsl"
#include "mp_vis_pack.hlsl"
#include "mp_cluster_layout.hlsl"
#include "page_request_emit.hlsl"

// M4.4 dispatcher tuning knobs — plan §3.M4 line 459.
// SW if projectedArea < MP_SW_AREA_THRESHOLD_PX AND triangleCount >= MP_SW_TRI_MIN.
// These are #define (not spec-constant) because dispatcher tuning is a build-
// time-fixed policy (no runtime switching), and changing them requires
// golden regen anyway.
// Nanite-style classifier: clusters whose conservative screen AABB is
// smaller than a 4x4 pixel square go through the SW raster compute path
// (fragment-amortisation dominates HW at sub-pixel scale). Larger
// clusters keep the HW mesh-shader path. Raising this favours SW, lowering
// favours HW — leave at 16 px² until the SW path gets proper per-tile
// spill lists (M4.6); with BIN_CAP=1024 and the bin-drop fix below, the
// SW path is already the fast path on BMW (~0.3 ms vs ~1.5 ms HW).
#define MP_SW_AREA_THRESHOLD_PX 16u    // 4x4 pixels
#define MP_SW_TRI_MIN           2u     // at least 2 tris to bother with SW

// --- Bindless resource arrays (mirror gpu_cull.comp.hlsl bindings) ----------
[[vk::binding(2, 0)]]
StructuredBuffer<float4> g_buffers[] : register(t0, space1);

[[vk::binding(5, 0)]]
RWByteAddressBuffer g_rwBuffers[] : register(u1, space0);

// Header/slot pair for the page-request queue. The header SSBO slot and the
// slots SSBO slot are the same RequestQueue::buffer_ — the first 16 bytes
// (4 u32) are the header, followed by capacity * u32 page-id slots. We
// access both via the same bindless RWByteAddressBuffer and carve out the
// header + slot-array views locally.

// --- Push constants ---------------------------------------------------------
// Matches MicropolyCullPushBlock (MicropolyCullPass.cpp). Field order here
// mirrors the C++ struct exactly.
struct PushBlock {
    uint totalClusterCount;         // number of DAG nodes to process this frame
    uint dagBufferBindlessIndex;    // StructuredBuffer<MpDagNode>  (read-only SRV)
    // (residencyBitmapBindlessIndex removed — residency now routes through
    //  pageToSlotBufferBindlessIndex; see isPageResident() below.)
    uint requestQueueBindlessIndex; // RWByteAddressBuffer — RequestQueue buffer
    uint indirectBufferBindlessIndex; // RWByteAddressBuffer — header + cmds
    uint cullStatsBindlessIndex;    // RWByteAddressBuffer — 7 u32 counters
    uint hiZBindlessIndex;          // bindless storage-image slot for HiZ
    uint cameraSlot;                // StructuredBuffer<float4> — CameraData
    float hiZMipCount;              // number of mips in the HiZ pyramid
    float screenSpaceErrorThreshold;// LOD selection threshold (pixel error)
    uint maxIndirectClusters;       // Security: cap for indirect-draw InterlockedAdd slot
    uint pageCount;                 // Security: cap for residency-bitmap bounds check
    uint rasterClassBufferBindlessIndex; // M4.4: RWByteAddressBuffer — u32 per drawSlot
    uint screenHeight;              // M4.4: viewport height in pixels (for projected-area tag)
    uint pageToSlotBufferBindlessIndex; // M4.4: pageId -> slotIndex (for triCount fetch)
    uint pageCacheBufferBindlessIndex;  // M4.4: page pool (for triCount fetch)
    uint pageSlotBytes;             // M4.4: PageCache::slotBytes — slot offset stride
    uint pageFirstDagNodeBufferBindlessIndex; // M4.5: pageId -> firstDagNodeIdx (u32[])
};
[[vk::push_constant]] PushBlock pc;

// --- DAG node load ----------------------------------------------------------
// MpDagNode layout — must match MpAssetReader::RuntimeDagNode exactly. The
// runtime uploads these as a DEVICE_LOCAL SSBO; the cull pass treats the
// whole-node SSBO as a flat array indexed by globalClusterIdx.
//
// Layout (64 bytes per node = 4 float4):
//   float4 m0 = {boundsCenter.xyz, boundsRadius}
//   float4 m1 = {coneApex.xyz, coneCutoff}
//   float4 m2 = {coneAxis.xyz, asfloat(packed)}
//   float4 m3 = {maxError, parentMaxError, 0, 0}
//   where packed = pageId<<0 (low 24b) | lodLevel<<24 (high 8b).
//
// pageId / lodLevel accessors pull from m2.w. maxError / parentMaxError in
// m3.xy drive the M4 screen-space-error traversal (see CSMain below).
// parentMaxError == FLT_MAX marks a root cluster (no coarser parent exists);
// the SSE test falls back to "emit when own error is above threshold".
struct MpDagNode {
    float3 center;
    float  radius;
    float3 coneApex;
    float  coneCutoff;
    float3 coneAxis;
    uint   pageId;
    uint   lodLevel;
    float  maxError;        // M4: world-space simplification error at this LOD
    float  parentMaxError;  // M4: error of the next-coarser parent, or FLT_MAX for roots
};

MpDagNode loadDagNode(uint idx) {
    StructuredBuffer<float4> buf = g_buffers[NonUniformResourceIndex(pc.dagBufferBindlessIndex)];
    const uint base = idx * 4u;
    float4 m0 = buf[base + 0u];
    float4 m1 = buf[base + 1u];
    float4 m2 = buf[base + 2u];
    float4 m3 = buf[base + 3u];

    MpDagNode n;
    // Engine-wide -90° Y correction: bake writes positions in the
    // source glTF's native axes (+X forward), but the engine runs in
    // physics/render space (+Z forward) — Application.cpp applies the
    // same rotation to glTF node rest transforms. Apply here so cull's
    // frustum, backface cone, and LOD/classifier projections match the
    // raster paths (mp_raster.mesh.hlsl, sw_raster*.comp.hlsl). Without
    // this, backface cone cull misjudges orientation and most 'visible'
    // survivors end up occluded by their own back-facing neighbours.
    const float3 centerRaw   = m0.xyz;
    const float3 coneApexRaw = m1.xyz;
    const float3 coneAxisRaw = m2.xyz;
    n.center     = float3(-centerRaw.z,   centerRaw.y,   centerRaw.x);
    n.radius     = m0.w;
    n.coneApex   = float3(-coneApexRaw.z, coneApexRaw.y, coneApexRaw.x);
    n.coneCutoff = m1.w;
    n.coneAxis   = float3(-coneAxisRaw.z, coneAxisRaw.y, coneAxisRaw.x);
    const uint packed = asuint(m2.w);
    n.pageId         = packed & 0x00FFFFFFu;           // low 24 bits
    n.lodLevel       = (packed >> 24u) & 0xFFu;        // high 8 bits
    n.maxError       = m3.x;
    n.parentMaxError = m3.y;
    return n;
}

// --- Camera load (column-major GLM -> HLSL row-major, matches gpu_cull) ----
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

// --- Residency query --------------------------------------------------------
// A page is resident iff pageToSlotBuffer[pageId] != UINT32_MAX. This is
// the authoritative source the HW + SW raster paths already read, so we
// reuse it here rather than maintaining a separate residency bitmap — the
// earlier bitmap-based query pointed at an SSBO that was never actually
// allocated (residencyBitmapBindlessIndex stayed UINT32_MAX), which made
// every page look non-resident and dropped every cluster from the cull
// survivor list. Format mirrors classSlotForPage() below: 4 u32s packed
// per float4 StructuredBuffer slot.
bool isPageResident(uint pageId) {
    if (pageId >= pc.pageCount) return false;
    StructuredBuffer<float4> buf = g_buffers[NonUniformResourceIndex(pc.pageToSlotBufferBindlessIndex)];
    const uint float4Idx = pageId >> 2u;
    const uint component = pageId & 3u;
    uint4 packed = asuint(buf[float4Idx]);
    uint slot = (component == 0u) ? packed.x
              : (component == 1u) ? packed.y
              : (component == 2u) ? packed.z
                                  : packed.w;
    return slot != 0xFFFFFFFFu;
}

// --- View-space frustum cull (adapted from gpu_cull.comp.hlsl) --------------
// Returns true when the cluster should be KEPT (inside / straddling frustum).
// Returns false when the cluster is fully outside — reject.
//
// DXC -Zpc caveat: the float4x4 constructor takes COLUMN vectors, so the
// existing transpose(float4x4(...)) load pattern in loadCamera is the
// proven fix. This function avoids Gribb-Hartmann plane extraction and
// instead reconstructs frustum half-extents from cam.proj diagonals —
// those diagonals are matrix ELEMENTS, not vectors, and are identical
// under either interpretation.
bool frustumCullVS(CameraData cam, float3 worldCenter, float worldRadius) {
    float4 viewPos4 = mul(cam.view, float4(worldCenter, 1.0));
    float  vx    = viewPos4.x;
    float  vy    = viewPos4.y;
    float  depth = -viewPos4.z;              // positive = in front of camera

    const float fPerAspect = cam.proj[0][0];  // f / aspect
    const float hw = depth / fPerAspect;      // frustum half-width at depth
    const float hh = depth / (-cam.proj[1][1]); // frustum half-height at depth

    // Signed distances (positive = inside).
    const float dLeft   = vx + hw;
    const float dRight  = hw - vx;
    const float dBottom = vy + hh;
    const float dTop    = hh - vy;
    const float dNear   = depth;

    // Correct radii for unnormalised plane normals (see gpu_cull notes).
    const float tanHFovX = 1.0f / fPerAspect;
    const float tanHFovY = 1.0f / (-cam.proj[1][1]);
    const float worldRadiusLR = worldRadius * sqrt(1.0f + tanHFovX * tanHFovX);
    const float worldRadiusTB = worldRadius * sqrt(1.0f + tanHFovY * tanHFovY);

    if (dLeft   < -worldRadiusLR) return false;
    if (dRight  < -worldRadiusLR) return false;
    if (dBottom < -worldRadiusTB) return false;
    if (dTop    < -worldRadiusTB) return false;
    if (dNear   < -worldRadius)   return false;
    return true;
}

// --- Normal-cone backface cull ---------------------------------------------
// Returns true when the cluster is guaranteed to be entirely backfacing —
// caller rejects it. coneCutoff = -cos(theta + alpha) where theta is the
// opening half-angle and alpha is the camera-to-apex-direction safety
// padding. This is the standard meshopt cone encoding; the test
// matches gpu_cull.comp.hlsl's (disabled-by-default) form.
bool normalConeBackfaceCullVS(MpDagNode node, CameraData cam) {
    // Camera position in WORLD space (worldPos.w unused, xyz are coords).
    const float3 camWorld = cam.worldPos.xyz;
    const float3 viewDir  = normalize(node.coneApex - camWorld);
    // Backfacing when dot(coneAxis, viewDir) > coneCutoff.
    return dot(node.coneAxis, viewDir) > node.coneCutoff;
}

// --- HiZ occlusion cull -----------------------------------------------------
// Project the world-space bounding sphere to screen space, sample HiZ at
// the mip level that corresponds to the sphere's screen-space extent, and
// reject when the nearest point on the sphere is farther than the stored
// depth (reverse-Z: stored is the MIN depth = the nearest surface).
//
// In reverse-Z the cluster's "nearest depth" (closest-to-camera point) is
// the LARGEST depth value; it must be >= the HiZ read (the occluder's
// nearest depth) for the cluster to be visible.
bool hiZOcclusionCullRZ(CameraData cam, float3 worldCenter, float worldRadius) {
    // Project centre to clip space.
    float4 clip = mul(cam.viewProj, float4(worldCenter, 1.0));
    if (clip.w <= 0.0f) {
        // Behind camera — frustum cull already handled; be conservative here.
        return false;
    }
    const float invW = 1.0f / clip.w;
    const float ndcX = clip.x * invW;
    const float ndcY = clip.y * invW;

    // Sphere radius in NDC — conservative extent using proj diagonals.
    const float fPerAspect = cam.proj[0][0];
    const float fY         = -cam.proj[1][1];
    const float radiusX    = worldRadius * fPerAspect * invW;
    const float radiusY    = worldRadius * fY         * invW;

    // Sample at the bounding extent's centre. The mip level must cover
    // at least the sphere's screen-space bounding box.
    const float uvExtentX = max(0.0f, radiusX);
    const float uvExtentY = max(0.0f, radiusY);
    // Convert clip-space extents to a mip level: radius ~ 2^(mip-1).
    const float extentPx = max(uvExtentX, uvExtentY);
    float mipF = 0.0f;
    if (extentPx > 0.0f) {
        mipF = log2(max(extentPx, 1.0e-6f)) + 1.0f;
    }
    const float mipClamped = clamp(mipF, 0.0f, pc.hiZMipCount - 1.0f);
    // For M3.2 we don't sample the HiZ — the Renderer wires a bindless
    // storage image slot but the cull pass is conservative and treats
    // every cluster that survives the frustum + backface test as
    // potentially visible. M3.3 adds the sampled-image path (HiZ is
    // storage-image; we don't have a sampler-image view here). This keeps
    // the M3.2 scope focused on DAG traversal + residency + indirect-draw
    // assembly while still reading hiZMipCount / hiZBindlessIndex from the
    // push block so the ABI stays stable.
    //
    // Suppress unused-variable warnings on -Wconversion strictness:
    (void)mipClamped;
    (void)ndcX;
    (void)ndcY;
    return false;  // "not occluded" — survive this cull for M3.2
}

// --- Cull stats accumulator -------------------------------------------------
// Layout (u32 offsets into cullStats RWByteAddressBuffer — 7 u32 counters):
//   0  : totalDispatched
//   4  : culledLOD
//   8  : culledResidency (non-resident, request emitted)
//  12  : culledFrustum
//  16  : culledBackface
//  20  : culledHiZ
//  24  : visible
void bumpStat(uint byteOffset) {
    RWByteAddressBuffer stats = g_rwBuffers[NonUniformResourceIndex(pc.cullStatsBindlessIndex)];
    uint prev;
    stats.InterlockedAdd(byteOffset, 1u, prev);
}

// --- M4.4 dispatcher classifier --------------------------------------------
// Look up the resident PageCache slot for a pageId. Mirrors
// mp_raster.task.hlsl::slotForPage. Used to fetch triangleCount for the
// classifier below.
uint classSlotForPage(uint pageId) {
    if (pageId >= pc.pageCount) return 0xFFFFFFFFu;
    StructuredBuffer<float4> buf = g_buffers[NonUniformResourceIndex(pc.pageToSlotBufferBindlessIndex)];
    const uint float4Idx = pageId >> 2u;
    const uint component = pageId & 3u;
    uint4 packed = asuint(buf[float4Idx]);
    if      (component == 0u) return packed.x;
    else if (component == 1u) return packed.y;
    else if (component == 2u) return packed.z;
    else                      return packed.w;
}

// Fetch triangleCount from the resident page's ClusterOnDisk for this
// cluster. M4.5: recover the per-page local cluster index via
// `globalDagNodeIdx - pageFirstDagNodeBuffer[pageId]`. Multi-cluster pages
// now route each cluster to its own ClusterOnDisk entry instead of
// hardcoding index 0.
uint classLoadLocalClusterIdx(uint globalDagNodeIdx, uint pageId) {
    if (pageId >= pc.pageCount) return 0xFFFFFFFFu;
    StructuredBuffer<float4> buf = g_buffers[NonUniformResourceIndex(pc.pageFirstDagNodeBufferBindlessIndex)];
    const uint float4Idx = pageId >> 2u;
    const uint component = pageId & 3u;
    uint4 packed = asuint(buf[float4Idx]);
    uint firstDagNodeIdx;
    if      (component == 0u) firstDagNodeIdx = packed.x;
    else if (component == 1u) firstDagNodeIdx = packed.y;
    else if (component == 2u) firstDagNodeIdx = packed.z;
    else                      firstDagNodeIdx = packed.w;
    // M4.5 Phase 4 HIGH fix: underflow is a corrupt-asset signal. Return
    // the UINT32_MAX sentinel so classLoadTriangleCount sees it and
    // reports triangleCount=0, which forces the classifier into HW and
    // lets both raster paths skip the cluster cleanly.
    if (globalDagNodeIdx < firstDagNodeIdx) return 0xFFFFFFFFu;
    return globalDagNodeIdx - firstDagNodeIdx;
}

uint classLoadTriangleCount(uint slotIndex, uint globalDagNodeIdx, uint pageId) {
    if (slotIndex == 0xFFFFFFFFu) return 0u;
    const uint localClusterIdx = classLoadLocalClusterIdx(globalDagNodeIdx, pageId);
    // M4.5 Phase 4 HIGH fix: corrupt-asset sentinel -> triCount=0. Lets the
    // classifier fall back to HW (triangleCount < MP_SW_TRI_MIN) and the
    // HW task shader skip via its own corrupt-skip path.
    if (localClusterIdx == 0xFFFFFFFFu) return 0u;
    RWByteAddressBuffer pageBuf = g_rwBuffers[NonUniformResourceIndex(pc.pageCacheBufferBindlessIndex)];
    const uint pageByteOffset = slotIndex * pc.pageSlotBytes;
    const uint clusterOnDiskOff =
        MP_PAGE_PAYLOAD_HEADER_BYTES + localClusterIdx * MP_CLUSTER_ON_DISK_STRIDE;
    // triangleCount is at byte offset 4 within ClusterOnDisk (second u32).
    uint tri = pageBuf.Load(pageByteOffset + clusterOnDiskOff + 4u);
    if (tri > 128u) tri = 128u;  // defensive cap — plan kMpVisTriBits
    return tri;
}

// Classify the cluster as HW- or SW-rasterization-bound based on its
// projected screen-space area and triangle count. Plan §3.M4 line 459:
//   projectedArea < 4 pixels && triCount >= 2 -> SW-bound.
// MP_SW_AREA_THRESHOLD_PX = 4*4 = 16 (conservative AABB area).
//
// The screen-space radius is computed from the perspective-projected
// bounding sphere: pxRadius = worldRadius * cam.proj[1][1] / depth * (H/2).
// We use cam.proj[1][1] (= -f = -cot(fovY/2) under our reverse-Z proj) so
// pxRadius scales correctly as the cluster moves forward/back. Behind-camera
// clusters (depth <= 0) conservatively return HW (they were already frustum-
// culled upstream — defence-in-depth only).
uint classifyRasterClass(float3 worldCenter, float worldRadius,
                         uint triangleCount, CameraData cam) {
    if (triangleCount < MP_SW_TRI_MIN) return kMpRasterClassHw;

    // M5 interim: route every non-trivial cluster through the SW raster.
    // The HW mesh-shader path is known-good for large primitives but the
    // driver silently discards sub-pixel triangles (the usual HW raster
    // coverage rule), which produces the "sparse dots" symptom on BMW's
    // micropoly-scale geometry. SW has proven correctness + is budgeted
    // at ~3 ms for 16k clusters, so until M4.6 lands its own sub-pixel-
    // safe HW path we prefer SW everywhere. Revert the early-return
    // when HW sub-pixel handling ships.
    (void)worldCenter; (void)worldRadius; (void)cam;
    return kMpRasterClassSw;

    // View-space depth: positive = in front of camera.
    float4 viewPos = mul(cam.view, float4(worldCenter, 1.0f));
    const float depth = -viewPos.z;
    if (depth <= 1.0e-4f) return kMpRasterClassHw;

    // cam.proj[1][1] is -cot(fovY/2) under the engine's perspective proj
    // (see frustumCullVS for the same expression). Negate + halve-height to
    // get the pixel-radius scale.
    const float fY = -cam.proj[1][1];
    const float halfH = 0.5f * (float)pc.screenHeight;
    const float pxRadius = worldRadius * fY * halfH / depth;

    // Conservative square-AABB area: (2*pxRadius)^2 = 4 * pxRadius^2.
    const float pxAreaRaw = 4.0f * pxRadius * pxRadius;
    // M4.4 Phase 4 defensive: corrupt geometry (Inf/NaN bounds) defaults to HW
    // rather than relying on IEEE-754 NaN-comparison-is-false falling through.
    if (!isfinite(pxAreaRaw)) return kMpRasterClassHw;

    // Hysteresis: snap to the integer grid before the threshold compare so
    // float jitter at the pxArea ≈ threshold boundary (camera drift of a
    // single frame can shift pxArea by < 0.1 px²) doesn't flip the class
    // back and forth between HW and SW. floor() quantizes to 1 px² which
    // is the natural granularity — a cluster whose pxArea crosses an
    // integer boundary has genuinely changed scale.
    const float pxArea = floor(pxAreaRaw);
    if (pxArea < (float)MP_SW_AREA_THRESHOLD_PX) return kMpRasterClassSw;
    return kMpRasterClassHw;
}

// --- Indirect draw emit -----------------------------------------------------
// Indirect buffer layout:
//   offset 0        : count (u32)      — incremented by InterlockedAdd
//   offset 4        : _pad (u32)
//   offset 16..     : draw commands, 16 bytes each:
//     {groupCountX, groupCountY, groupCountZ, clusterIdx}
// Task+Mesh pipelines in M3.3 will bind this buffer as the indirect
// source with stride=16 and offset=16.
//
// M4.4: In parallel with the indirect-draw write, we emit a per-drawSlot
// rasterClass tag (kMpRasterClassHw / kMpRasterClassSw) into
// rasterClassBuffer[drawSlot]. Both raster paths read the tag and early-
// out for clusters not assigned to them. Survivor clusters that classify
// as SW still get a draw command emitted so the HW path can early-out
// cheaply (one wasted mesh-task workgroup per SW cluster — acceptable).
void emitDrawCmd(uint clusterIdx, MpDagNode node, CameraData cam) {
    RWByteAddressBuffer ind = g_rwBuffers[NonUniformResourceIndex(pc.indirectBufferBindlessIndex)];
    uint drawSlot;
    ind.InterlockedAdd(0u, 1u, drawSlot);
    // M4.2 Phase 4: roll back the counter on overflow so the count header
    // at offset 0 stays capped at maxIndirectClusters. Previously the count
    // could drift above the cap (writes were guarded, increments were not),
    // which let M4.2's sw_raster_bin_prep dispatchIndirect see counts >65k
    // and risk exceeding maxComputeWorkGroupCount. Rollback converges: under
    // races, each overflowing thread decrements one of the overcounts, so
    // the steady-state count never exceeds maxIndirectClusters. The prep
    // shader's own clamp stays as defense-in-depth.
    if (drawSlot >= pc.maxIndirectClusters) {
        uint dummy;
        ind.InterlockedAdd(0u, 0xFFFFFFFFu, dummy);  // -1 as u32
        return;
    }
    const uint base = 16u + drawSlot * 16u;  // 16-byte header
    ind.Store4(base, uint4(1u, 1u, 1u, clusterIdx));

    // M4.4: classify + emit class tag at the same drawSlot. The tag is one
    // u32 per slot (0=HW, 1=SW). Downstream raster paths read the tag to
    // route work.
    // M4.5: classLoadTriangleCount now takes the global DAG node index so it
    // can recover the page-local cluster index for multi-cluster pages.
    const uint slotIdx = classSlotForPage(node.pageId);
    const uint triCount = classLoadTriangleCount(slotIdx, clusterIdx, node.pageId);
    const uint rasterClass = classifyRasterClass(node.center, node.radius,
                                                 triCount, cam);
    RWByteAddressBuffer classBuf = g_rwBuffers[NonUniformResourceIndex(pc.rasterClassBufferBindlessIndex)];
    classBuf.Store(drawSlot * 4u, rasterClass);
}

// --- Page request emit ------------------------------------------------------
// The RequestQueue buffer lives at requestQueueBindlessIndex. Header is the
// first 16 bytes (4 u32); slots start at offset 16.
void emitPageReq(uint pageId) {
    RWByteAddressBuffer req = g_rwBuffers[NonUniformResourceIndex(pc.requestQueueBindlessIndex)];
    uint slot;
    req.InterlockedAdd(0u, 1u, slot);
    // Header layout: [0]=count, [1]=capacity, [2]=overflowed, [3]=pad.
    const uint capacity = req.Load(4u);
    if (slot >= capacity) {
        uint prevOverflow;
        req.InterlockedOr(8u, 1u, prevOverflow);
        return;
    }
    req.Store(16u + slot * 4u, pageId);
}

// --- Main -------------------------------------------------------------------
[numthreads(64, 1, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID) {
    const uint clusterIdx = dtid.x;
    if (clusterIdx >= pc.totalClusterCount) return;

    bumpStat(0u);  // totalDispatched

    MpDagNode node = loadDagNode(clusterIdx);

    // Cull-order rationale (architect review): residency-before-frustum is
    // intentional and cost-ordered. Residency test is a 1-word bitmap read;
    // frustum test loads 20+ floats from the camera SSBO. This trades
    // first-frame latency amortisation (a non-resident cluster still bumps
    // culledResidency even when off-screen) for slight request-queue noise —
    // the queue is already backed by emitPageReq()'s overflow-OR guard.
    //
    // M4 ordering: residency → frustum → LOD (SSE) → backface → HiZ.
    // Frustum runs before LOD because it's cheaper than a pair of view-space
    // projections for out-of-frame clusters. LOD runs before backface/HiZ
    // because it's a deterministic DAG-traversal gate and trims the survivor
    // pool before the more expensive cone/HiZ tests.

    // 1) Residency check. Non-resident pages trigger a request and the
    //    cluster is skipped this frame. Once the streaming pump completes
    //    the transfer a subsequent frame will see isPageResident() == true
    //    and the cluster will progress through the remaining culls.
    if (!isPageResident(node.pageId)) {
        emitPageReq(node.pageId);
        bumpStat(8u);  // culledResidency
        return;
    }

    // 2) Frustum cull (view-space — see banner for -Zpc caveat).
    CameraData cam = loadCamera(pc.cameraSlot);
    if (!frustumCullVS(cam, node.center, node.radius)) {
        bumpStat(12u); // culledFrustum
        return;
    }

    // 3) Screen-space-error LOD traversal (Nanite-style).
    //    A cluster C is emitted iff
    //      projectedError(C)          <= pixelThreshold, AND
    //      projectedError(parent(C))  >  pixelThreshold
    //    Together these two conditions guarantee exactly-one-LOD coverage
    //    across the DAG: every rendered pixel is claimed by exactly one
    //    cluster, with no overlap or cracks at cut boundaries (provided
    //    the per-group simplify is boundary-symmetric, which
    //    DagBuilder.cpp enforces via the group-external lock mask).
    //
    //    Projection: worldError * cot(fovY/2) * (screenH/2) / viewDepth.
    //    We use cam.proj[1][1] (= -cot(fovY/2) under the engine's reverse-Z
    //    perspective — same convention as frustumCullVS + classifyRasterClass).
    //
    //    Root fallback: parentMaxError == FLT_MAX at roots forces the
    //    parent projected error to infinity, so the "> threshold" condition
    //    always holds at the root and the cluster is emitted when its own
    //    error falls below threshold OR when no finer descendant is
    //    available. This handles non-simplifiable roots (e.g. BMW's tiny
    //    disconnected shells) cleanly.
    {
        float4 viewPos = mul(cam.view, float4(node.center, 1.0f));
        const float depth = max(-viewPos.z, 1.0e-4f);
        const float fY    = -cam.proj[1][1];
        const float halfH = 0.5f * (float)pc.screenHeight;
        const float scale = (fY * halfH) / depth;
        const float errSelf   = node.maxError       * scale;
        const float errParent = node.parentMaxError * scale;
        const float threshold = pc.screenSpaceErrorThreshold;

        // Standard Nanite SSE rule: emit iff errSelf <= threshold AND
        // errParent > threshold. Both must hold for exactly-one-LOD coverage.
        const bool sseAccept = (errSelf <= threshold)
                            && (errParent > threshold);

        // Root fallback: a cluster with parentMaxError == FLT_MAX has no
        // coarser parent. When its own projected error is ABOVE threshold
        // the standard rule would reject (no finer descendant exists to
        // take over), leaving a hole. Emit these unconditionally so
        // non-simplifiable shells (e.g. BMW's disconnected-component
        // roots) always render — they're the leaf of their own DAG subtree.
        // `isinf` catches the FLT_MAX * scale projection.
        const bool isRootFallback = isinf(errParent) && (errSelf > threshold);
        const bool accept = sseAccept || isRootFallback;
        if (!accept) {
            bumpStat(4u);  // culledLOD
            return;
        }
    }

    // 4) Normal-cone backface cull. Re-enabled now that loadDagNode
    // rotates coneAxis/coneApex through the engine-wide -90° Y
    // correction so the cone orientation matches the rendered geometry.
    if (normalConeBackfaceCullVS(node, cam)) {
        bumpStat(16u); // culledBackface
        return;
    }

    // 5) HiZ occlusion cull (conservative no-op in M3.2; ABI stable).
    if (hiZOcclusionCullRZ(cam, node.center, node.radius)) {
        bumpStat(20u); // culledHiZ
        return;
    }

    // 6) Survivor — append an indirect draw.
    emitDrawCmd(clusterIdx, node, cam);
    bumpStat(24u);     // visible
}
