// material_eval.comp.hlsl
// =======================
// Full-screen compute shader for the visibility buffer material evaluation
// pass. Reads the R32_UINT visibility buffer, reconstructs barycentrics from
// the triangle's clip-space positions, evaluates PBR materials, and writes
// to G-buffer storage images.
//
// G-buffer outputs (layout defined in GBufferFormats.h):
//   albedo      rgba8  — rgb=baseColor, a=ambient occlusion
//   normal      rgba16 — rgb=world normal packed to [0,1], a=unused
//   metalRough  rg8    — r=metallic, g=roughness (perceptual)
//   motionVec   rg16f  — rg=NDC-space velocity
//
// Dispatch: ceil(screenWidth / 8) x ceil(screenHeight / 8) workgroups.
// Depth convention: reverse-Z (far = 0, near = 1).

#include "common.hlsl"
#ifdef MP_ENABLE
#include "micropoly/mp_vis_pack.hlsl"
#include "micropoly/mp_cluster_layout.hlsl"
#endif

// --- Bindless resource arrays ---
[[vk::binding(0, 0)]]
Texture2D g_textures[] : register(t0, space0);

[[vk::binding(1, 0)]]
RWTexture2D<float4> g_storageImages[] : register(u0, space0);

// M3.4 — aliased view of the same bindless storage-image slot space as a
// 64-bit R64_UINT array. Only declared when MP_ENABLE is defined so the
// =false DXC compile never sees Int64 types and the SPIR-V is byte-identical
// to the pre-M3.4 golden (Principle 1 acceptance gate).
#ifdef MP_ENABLE
[[vk::binding(1, 0)]]
RWTexture2D<uint64_t> g_storageImages64[] : register(u0, space0);
#endif

[[vk::binding(2, 0)]]
StructuredBuffer<float4> g_buffers[] : register(t0, space1);

[[vk::binding(3, 0)]]
SamplerState g_samplers[] : register(s0, space0);

[[vk::binding(5, 0)]]
RWByteAddressBuffer g_rwBuffers[] : register(u1, space0);

// --- Push constants ---
struct PushBlock {
    uint visBufferSlot;        // Texture2D<uint> — visibility buffer
    uint depthBufferSlot;      // Texture2D<float> — depth
    uint instanceBufferSlot;   // StructuredBuffer<GpuInstance>
    uint meshletBufferSlot;    // StructuredBuffer<Meshlet>
    uint meshletVerticesSlot;  // StructuredBuffer<uint> — vertex index remapping
    uint meshletTrianglesSlot; // ByteAddressBuffer — packed u8 triangle indices
    uint materialBufferSlot;   // StructuredBuffer<float4> — materials
    uint cameraSlot;
    uint albedoStorageSlot;    // RWTexture2D<float4> — G-buffer write targets
    uint normalStorageSlot;
    uint metalRoughStorageSlot;
    uint motionVecStorageSlot;
    uint screenWidth;
    uint screenHeight;
    uint meshletToInstanceSlot; // StructuredBuffer<float4> — u32 instanceId per globalMeshletId
    // M3.4 — Bindless slot of the 64-bit Micropoly vis image. Only
    // declared when MP_ENABLE is defined so the =false shader's push
    // constant layout is byte-identical to the pre-M3.4 golden (Principle 1).
    // The C++ side pushes the full MP-enabled block size for both variants;
    // Vulkan ignores the trailing bytes that the shader's layout doesn't
    // declare.
#ifdef MP_ENABLE
    uint vis64BufferSlot;
    // M5 material resolution: bindless slots needed to decode an mp sample's
    // geometry (DAG node -> page -> on-disk cluster -> vertices) and reproject
    // it. Appended after vis64BufferSlot so the MP-disabled prefix is still
    // byte-identical to the pre-M3.4 golden.
    uint mpDagBufferSlot;            // StructuredBuffer<float4> MpDagNode
    uint mpPageToSlotSlot;           // pageId -> slotIndex
    uint mpPageCacheSlot;            // entire page pool
    uint mpPageFirstDagNodeSlot;     // pageId -> firstDagNodeIdx
    uint mpPageSlotBytes;            // PageCache slot byte stride
    uint mpPageCount;                // pageToSlot bounds check
    uint mpDagNodeCount;             // DAG bounds check
#endif
};

[[vk::push_constant]] PushBlock pc;

// --- Constants ---
static const uint INVALID_VIS           = 0xFFFFFFFFu;
static const uint INVALID_TEX           = 0xFFFFFFFFu;
static const uint MATERIAL_FLAG_TERRAIN = 0x4u; // bit 2 — must match Scene.h kFlagTerrain

// --- Camera load (column-major GLM -> HLSL row-major) ---
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

// --- GPU struct accessors ---
struct GpuInstance {
    float4x4 transform;
    uint     meshlet_offset;
    uint     meshlet_count;
    uint     material_index;
    uint     vertex_buffer_slot;
    uint     vertex_base_offset;
    uint     _pad;
};

GpuInstance loadInstance(uint idx) {
    StructuredBuffer<float4> buf = g_buffers[NonUniformResourceIndex(pc.instanceBufferSlot)];
    const uint base = idx * 6;
    GpuInstance inst;
    inst.transform = transpose(float4x4(buf[base + 0], buf[base + 1], buf[base + 2], buf[base + 3]));
    uint4 pack0    = asuint(buf[base + 4]);
    inst.meshlet_offset     = pack0.x;
    inst.meshlet_count      = pack0.y;
    inst.material_index     = pack0.z;
    inst.vertex_buffer_slot = pack0.w;
    uint4 pack1             = asuint(buf[base + 5]);
    inst.vertex_base_offset = pack1.x;
    inst._pad               = pack1.y;
    return inst;
}

struct Meshlet {
    uint   vertex_offset;
    uint   triangle_offset;
    uint   vertex_count;
    uint   triangle_count;
    float3 center;
    float  radius;
    float3 cone_axis;
    float  cone_cutoff;
};

Meshlet loadMeshlet(uint idx) {
    StructuredBuffer<float4> buf = g_buffers[NonUniformResourceIndex(pc.meshletBufferSlot)];
    const uint base = idx * 3;
    uint4  m0 = asuint(buf[base + 0]);
    float4 m1 = buf[base + 1];
    float4 m2 = buf[base + 2];

    Meshlet m;
    m.vertex_offset   = m0.x;
    m.triangle_offset = m0.y;
    m.vertex_count    = m0.z;
    m.triangle_count  = m0.w;
    m.center          = m1.xyz;
    m.radius          = m1.w;
    m.cone_axis       = m2.xyz;
    m.cone_cutoff     = m2.w;
    return m;
}

// --- Material struct (mirrors GpuMaterial in GBufferFormats.h) ---
struct GpuMaterial {
    float4 baseColorFactor;
    float4 emissiveFactor;    // .w = alphaCutoff
    float  metallicFactor;
    float  roughnessFactor;
    float  normalScale;
    float  occlusionStrength;
    uint   baseColorTexIdx;
    uint   metalRoughTexIdx;
    uint   normalTexIdx;
    uint   emissiveTexIdx;
    uint   occlusionTexIdx;
    uint   flags;
    uint   samplerSlot;
};

GpuMaterial loadMaterial(uint bufSlot, uint idx) {
    StructuredBuffer<float4> buf = g_buffers[NonUniformResourceIndex(bufSlot)];
    const uint base = idx * 5;
    float4 m0 = buf[base + 0];
    float4 m1 = buf[base + 1];
    float4 m2 = buf[base + 2];
    uint4  m3 = asuint(buf[base + 3]);
    uint4  m4 = asuint(buf[base + 4]);

    GpuMaterial m;
    m.baseColorFactor   = m0;
    m.emissiveFactor    = m1;
    m.metallicFactor    = m2.x;
    m.roughnessFactor   = m2.y;
    m.normalScale       = m2.z;
    m.occlusionStrength = m2.w;
    m.baseColorTexIdx   = m3.x;
    m.metalRoughTexIdx  = m3.y;
    m.normalTexIdx      = m3.z;
    m.emissiveTexIdx    = m3.w;
    m.occlusionTexIdx   = m4.x;
    m.flags             = m4.y;
    m.samplerSlot       = m4.z;
    return m;
}

// --- Vertex attribute loading ---
// Reads a single vertex's full attributes from the vertex SSBO.
struct VertexAttribs {
    float3 position;
    float3 normal;
    float2 uv;
    float3 tangent;
    float  bitangentSign;
};

VertexAttribs loadVertex(uint vertexBufferSlot, uint vid) {
    StructuredBuffer<float4> vbuf = g_buffers[NonUniformResourceIndex(vertexBufferSlot)];
    float4 d0 = vbuf[vid * 3 + 0];
    float4 d1 = vbuf[vid * 3 + 1];
    float4 d2 = vbuf[vid * 3 + 2];

    VertexAttribs v;
    v.position     = d0.xyz;
    v.normal       = float3(d0.w, d1.x, d1.y);
    v.uv           = float2(d1.z, d1.w);
    v.tangent      = d2.xyz;
    v.bitangentSign = d2.w;
    return v;
}

// --- Read a uint from meshlet vertex index buffer (packed as float4) ---
uint loadMeshletVertexIndex(uint slot, uint flatIdx) {
    StructuredBuffer<float4> buf = g_buffers[NonUniformResourceIndex(slot)];
    uint float4Idx = flatIdx / 4;
    uint component = flatIdx % 4;
    uint4 packed   = asuint(buf[float4Idx]);
    if      (component == 0) return packed.x;
    else if (component == 1) return packed.y;
    else if (component == 2) return packed.z;
    else                     return packed.w;
}

// --- Read 3 u8 triangle indices from the packed triangle buffer ---
uint3 loadTriangleIndices(uint trianglesSlot, uint triangleOffset, uint triIdx) {
    RWByteAddressBuffer triBuf = g_rwBuffers[NonUniformResourceIndex(trianglesSlot)];
    uint byteOffset = triangleOffset + triIdx * 3;
    uint wordOffset = byteOffset & ~3u;
    uint shift      = (byteOffset & 3u) * 8;

    uint word0 = triBuf.Load(wordOffset);
    uint word1 = triBuf.Load(wordOffset + 4);

    // When shift == 0, (32 - shift) == 32 and HLSL masks shift amounts
    // modulo 32, so word1 << 32 becomes word1 << 0 = word1 (wrong).
    // Guard against this: if shift == 0, word0 already contains the bytes.
    uint bits;
    if (shift == 0) {
        bits = word0;
    } else {
        bits = (word0 >> shift) | (word1 << (32 - shift));
    }
    uint i0 = bits & 0xFF;
    uint i1 = (bits >> 8) & 0xFF;
    uint i2 = (bits >> 16) & 0xFF;

    return uint3(i0, i1, i2);
}

// --- Screen-space barycentric reconstruction ---
// Given 3 NDC positions (after perspective divide) and the pixel NDC,
// compute barycentrics using the Cramer's rule approach.
float3 computeBarycentrics(float2 ndc, float2 ndc0, float2 ndc1, float2 ndc2) {
    float2 v0 = ndc1 - ndc0;
    float2 v1 = ndc2 - ndc0;
    float2 v2 = ndc - ndc0;
    float d00 = dot(v0, v0);
    float d01 = dot(v0, v1);
    float d11 = dot(v1, v1);
    float d20 = dot(v2, v0);
    float d21 = dot(v2, v1);
    float inv_denom = 1.0 / (d00 * d11 - d01 * d01);
    float v = (d11 * d20 - d01 * d21) * inv_denom;
    float w = (d00 * d21 - d01 * d20) * inv_denom;
    return float3(1.0 - v - w, v, w);
}

// --- Perspective-correct barycentric interpolation ---
// Corrects screen-space barycentrics for perspective using clip-space w.
float3 perspectiveCorrectBarycentrics(float3 screenBary, float w0, float w1, float w2) {
    float3 perspBary = float3(screenBary.x / w0, screenBary.y / w1, screenBary.z / w2);
    float sum = perspBary.x + perspBary.y + perspBary.z;
    return perspBary / sum;
}

// --- Main ---
[numthreads(8, 8, 1)]
void CSMain(uint3 dispatchId : SV_DispatchThreadID) {
    uint2 pixelCoord = dispatchId.xy;
    if (pixelCoord.x >= pc.screenWidth || pixelCoord.y >= pc.screenHeight)
        return;

    // Read visibility buffer (R32_UINT).
    Texture2D visTex = g_textures[NonUniformResourceIndex(pc.visBufferSlot)];
    uint visPacked = asuint(visTex.Load(int3(pixelCoord, 0)).r);

    // M3.4 — Micropoly merge. Gated by the MP_ENABLE preprocessor define so
    // the =false DXC compile never sees Int64 types and produces SPIR-V
    // byte-identical to the pre-M3.4 baseline (Principle 1). When compiled
    // with -D MP_ENABLE=1 we peek at the 64-bit vis image populated by
    // mp_raster.{task,mesh}.hlsl, compare the encoded depth to the 32-bit
    // meshlet vis, and if the micropoly sample is closer we stamp a debug
    // magenta (full material resolution is M5 scope).
#ifdef MP_ENABLE
    {
        const uint64_t mpPacked =
            g_storageImages64[NonUniformResourceIndex(pc.vis64BufferSlot)][pixelCoord];
        if (mpPacked != kMpVisEmpty) {
            // Reverse-Z: larger uint depth = nearer, use `>` comparator.
            // The depth lives in the high 32 bits of the packed vis value
            // (see mp_vis_pack.hlsl banner). IEEE-754 bitwise ordering of
            // non-negative float32 matches numeric ordering, so comparing
            // the uint representations is equivalent to comparing the
            // floats directly but cheaper.
            const uint depth32FromMp    = (uint)(mpPacked >> 32);
            const float meshletDepth    = g_textures[NonUniformResourceIndex(pc.depthBufferSlot)]
                                            .Load(int3(pixelCoord, 0)).r;
            const uint depth32FromMesh  = asuint(meshletDepth);
            const bool meshletInvalid   = (visPacked == INVALID_VIS);
            // mp vs glTF depth resolution: mp positions come from the same
            // world coordinates as glTF (bake folds world transform + the
            // engine-wide -90° Y correction is applied in both paths), but
            // meshlet simplification (even at lodLevel==0 the bake can
            // shift positions by float rounding) and independent fragment
            // interpolation produce depths that differ by a few ulps at
            // matching surfaces. A strict `>` compare loses on every near-
            // tie, which silenced ~99% of mp fragments on BMW. Allow mp
            // to be up to a small relative epsilon farther than glTF so
            // co-located surfaces both survive; genuine occluders (foreground
            // glTF walls) still dominate because their depth is strictly
            // larger than mp by well more than epsilon.
            // mp wins when glTF has nothing at this pixel, OR when mp's
            // depth is within 1e-4 of glTF (tolerate simplification ulp
            // drift — strict `>` silenced ~99% of mp fragments on BMW).
            // Genuine foreground glTF occluders still dominate because
            // their depth is larger than mp by well more than epsilon.
            const float mpDepthF   = asfloat(depth32FromMp);
            const float kDepthEps  = 1.0e-4f;
            const bool mpWins = meshletInvalid
                              || (mpDepthF + kDepthEps >= meshletDepth);
            if (mpWins) {
                // M4.1 — vis-pack v2: 4-output unpack (depth32, rasterClass,
                // clusterId, triId). rasterClass is unused here; the M4.6
                // DebugMode::MicropolyRasterClass overlay reads it elsewhere.
                uint mpDepth32; uint mpRasterClass;
                uint mpClusterId; uint mpTriId;
                UnpackMpVis64(mpPacked, mpDepth32, mpRasterClass,
                              mpClusterId, mpTriId);
                (void)mpRasterClass;

                // M5 material resolution. Mirrors mp_cluster_cull::loadDagNode
                // + sw_raster{,_bin}.comp.hlsl's page->cluster->vertex walk.
                // All lookups are defensive — any out-of-range value falls
                // back to writing a neutral G-buffer sample so the engine's
                // lighting pass treats the pixel as "something is here" rather
                // than stamping garbage.
                //
                // Step 1: load the DAG node (4 float4 = 64 B per node — M4
                // widened from 3 float4 to carry maxError/parentMaxError; the
                // material pass only needs the pageId packed in m2.w). Apply
                // the engine-wide -90° Y correction to its positional data so
                // the cluster's bounds centre/cone match the rendered
                // geometry (the HW + SW raster paths already do the same).
                if (mpClusterId >= pc.mpDagNodeCount) return;
                StructuredBuffer<float4> dagBuf =
                    g_buffers[NonUniformResourceIndex(pc.mpDagBufferSlot)];
                const uint dagBase = mpClusterId * 4u;
                float4 dm0 = dagBuf[dagBase + 0u];
                // dm1 + dm2 carry cone data that the material pass doesn't
                // need beyond the pageId packed in dm2.w.
                float4 dm2 = dagBuf[dagBase + 2u];
                const uint pageId = asuint(dm2.w) & 0x00FFFFFFu;
                (void)dm0;
                if (pageId >= pc.mpPageCount) return;

                // Step 2: resolve the resident page slot + local cluster idx.
                StructuredBuffer<float4> p2sBuf =
                    g_buffers[NonUniformResourceIndex(pc.mpPageToSlotSlot)];
                const uint p2sFloat4 = pageId >> 2u;
                const uint p2sComp   = pageId & 3u;
                uint4 p2sPacked = asuint(p2sBuf[p2sFloat4]);
                uint slotIndex;
                if      (p2sComp == 0u) slotIndex = p2sPacked.x;
                else if (p2sComp == 1u) slotIndex = p2sPacked.y;
                else if (p2sComp == 2u) slotIndex = p2sPacked.z;
                else                    slotIndex = p2sPacked.w;
                if (slotIndex == 0xFFFFFFFFu) return;

                StructuredBuffer<float4> fdnBuf =
                    g_buffers[NonUniformResourceIndex(pc.mpPageFirstDagNodeSlot)];
                uint4 fdnPacked = asuint(fdnBuf[p2sFloat4]);
                uint firstDagIdx;
                if      (p2sComp == 0u) firstDagIdx = fdnPacked.x;
                else if (p2sComp == 1u) firstDagIdx = fdnPacked.y;
                else if (p2sComp == 2u) firstDagIdx = fdnPacked.z;
                else                    firstDagIdx = fdnPacked.w;
                if (mpClusterId < firstDagIdx) return;
                const uint localClusterIdx = mpClusterId - firstDagIdx;

                // Step 3: read the ClusterOnDisk header for this cluster. The
                // page pool is one big RWByteAddressBuffer; each slot starts
                // at slotIndex * mpPageSlotBytes and carries:
                //   PagePayloadHeader (16 B: clusterCount, version, pad, pad)
                //   ClusterOnDisk[clusterCount] (76 B each)
                //   vertex blob (32 B/vertex)
                //   triangle blob (3 B/triangle)
                RWByteAddressBuffer pageBuf =
                    g_rwBuffers[NonUniformResourceIndex(pc.mpPageCacheSlot)];
                const uint pageByteOffset = slotIndex * pc.mpPageSlotBytes;
                const uint clusterOnDiskOff =
                    MP_PAGE_PAYLOAD_HEADER_BYTES
                  + localClusterIdx * MP_CLUSTER_ON_DISK_STRIDE;
                uint4 cFields = pageBuf.Load4(pageByteOffset + clusterOnDiskOff);
                const uint vertexCount          = cFields.x;
                const uint triangleCount        = cFields.y;
                const uint vertexOffsetInBlob   = cFields.z;
                const uint triangleOffsetInBlob = cFields.w;
                if (vertexCount == 0u || triangleCount == 0u) return;
                if (mpTriId >= triangleCount) return;

                // Step 4: locate the vertex + triangle blob starts. Mirrors
                // the same loop in sw_raster.comp.hlsl::pageTriangleBlobStart
                // — multi-cluster pages concatenate vertex blobs back-to-back
                // so we sum every ClusterOnDisk.vertexCount in this page.
                uint pageClusterCount = pageBuf.Load(pageByteOffset + 0u);
                if (pageClusterCount > 64u) pageClusterCount = 64u;
                const uint vertexBlobStart = pageByteOffset
                    + MP_PAGE_PAYLOAD_HEADER_BYTES
                    + pageClusterCount * MP_CLUSTER_ON_DISK_STRIDE;
                uint totalVertexBytes = 0u;
                for (uint ci = 0u; ci < pageClusterCount; ++ci) {
                    const uint cAddr = pageByteOffset
                                     + MP_PAGE_PAYLOAD_HEADER_BYTES
                                     + ci * MP_CLUSTER_ON_DISK_STRIDE;
                    totalVertexBytes += pageBuf.Load(cAddr) * 32u;
                }
                const uint triangleBlobStart = vertexBlobStart + totalVertexBytes;

                // Step 5: read 3 packed u8 local vertex indices for this tri.
                const uint triByteOff = triangleBlobStart
                                      + triangleOffsetInBlob + mpTriId * 3u;
                const uint triWordOff = triByteOff & ~3u;
                const uint triShift   = (triByteOff & 3u) * 8u;
                const uint triW0 = pageBuf.Load(triWordOff);
                const uint triW1 = pageBuf.Load(triWordOff + 4u);
                uint triBits;
                if (triShift == 0u) triBits = triW0;
                else                triBits = (triW0 >> triShift)
                                            | (triW1 << (32u - triShift));
                const uint li0 = min(triBits         & 0xFFu, vertexCount - 1u);
                const uint li1 = min((triBits >>  8u) & 0xFFu, vertexCount - 1u);
                const uint li2 = min((triBits >> 16u) & 0xFFu, vertexCount - 1u);

                // Step 6: read 3 vertices (32 B each: vec3 pos + vec3 normal
                // + vec2 uv). Apply the engine-wide -90° Y correction
                // (x,y,z) -> (-z,y,x) to both positions AND normals. For a
                // pure rotation the normal transforms identically, so no
                // inverse-transpose matrix needed.
                const uint vertexBaseInBlob = vertexOffsetInBlob / 32u;
                const uint v0Addr = vertexBlobStart + (vertexBaseInBlob + li0) * 32u;
                const uint v1Addr = vertexBlobStart + (vertexBaseInBlob + li1) * 32u;
                const uint v2Addr = vertexBlobStart + (vertexBaseInBlob + li2) * 32u;
                uint3 v0Pos3 = pageBuf.Load3(v0Addr);
                uint3 v0Nrm3 = pageBuf.Load3(v0Addr + 12u);
                uint2 v0Uv2  = pageBuf.Load2(v0Addr + 24u);
                uint3 v1Pos3 = pageBuf.Load3(v1Addr);
                uint3 v1Nrm3 = pageBuf.Load3(v1Addr + 12u);
                uint2 v1Uv2  = pageBuf.Load2(v1Addr + 24u);
                uint3 v2Pos3 = pageBuf.Load3(v2Addr);
                uint3 v2Nrm3 = pageBuf.Load3(v2Addr + 12u);
                uint2 v2Uv2  = pageBuf.Load2(v2Addr + 24u);
                const float3 p0Raw = float3(asfloat(v0Pos3.x), asfloat(v0Pos3.y), asfloat(v0Pos3.z));
                const float3 p1Raw = float3(asfloat(v1Pos3.x), asfloat(v1Pos3.y), asfloat(v1Pos3.z));
                const float3 p2Raw = float3(asfloat(v2Pos3.x), asfloat(v2Pos3.y), asfloat(v2Pos3.z));
                const float3 n0Raw = float3(asfloat(v0Nrm3.x), asfloat(v0Nrm3.y), asfloat(v0Nrm3.z));
                const float3 n1Raw = float3(asfloat(v1Nrm3.x), asfloat(v1Nrm3.y), asfloat(v1Nrm3.z));
                const float3 n2Raw = float3(asfloat(v2Nrm3.x), asfloat(v2Nrm3.y), asfloat(v2Nrm3.z));
                const float2 uv0 = float2(asfloat(v0Uv2.x), asfloat(v0Uv2.y));
                const float2 uv1 = float2(asfloat(v1Uv2.x), asfloat(v1Uv2.y));
                const float2 uv2 = float2(asfloat(v2Uv2.x), asfloat(v2Uv2.y));
                const float3 p0W = float3(-p0Raw.z, p0Raw.y, p0Raw.x);
                const float3 p1W = float3(-p1Raw.z, p1Raw.y, p1Raw.x);
                const float3 p2W = float3(-p2Raw.z, p2Raw.y, p2Raw.x);
                const float3 n0W = float3(-n0Raw.z, n0Raw.y, n0Raw.x);
                const float3 n1W = float3(-n1Raw.z, n1Raw.y, n1Raw.x);
                const float3 n2W = float3(-n2Raw.z, n2Raw.y, n2Raw.x);

                // Step 7: project + winding-swap to match sw_raster::projectTri.
                CameraData mpCam = loadCamera(pc.cameraSlot);
                const float4 c0 = mul(mpCam.viewProj, float4(p0W, 1.0f));
                const float4 c1 = mul(mpCam.viewProj, float4(p1W, 1.0f));
                const float4 c2 = mul(mpCam.viewProj, float4(p2W, 1.0f));
                if (c0.w <= 0.0f || c1.w <= 0.0f || c2.w <= 0.0f) return;
                const float invW0 = 1.0f / c0.w;
                const float invW1 = 1.0f / c1.w;
                const float invW2 = 1.0f / c2.w;
                const float viewW = (float)pc.screenWidth;
                const float viewH = (float)pc.screenHeight;
                float2 pix0 = float2((c0.x * invW0 * 0.5f + 0.5f) * viewW,
                                     (c0.y * invW0 * 0.5f + 0.5f) * viewH);
                float2 pix1 = float2((c1.x * invW1 * 0.5f + 0.5f) * viewW,
                                     (c1.y * invW1 * 0.5f + 0.5f) * viewH);
                float2 pix2 = float2((c2.x * invW2 * 0.5f + 0.5f) * viewW,
                                     (c2.y * invW2 * 0.5f + 0.5f) * viewH);
                float3 nA = n0W, nB = n1W, nC = n2W;
                float2 uvA = uv0, uvB = uv1, uvC = uv2;
                const float edgeAll = (pix1.x - pix0.x) * (pix2.y - pix0.y)
                                    - (pix1.y - pix0.y) * (pix2.x - pix0.x);
                if (edgeAll == 0.0f) return;
                if (edgeAll < 0.0f) {
                    const float2 tmpPix = pix1; pix1 = pix2; pix2 = tmpPix;
                    const float3 tmpN  = nB;  nB  = nC;  nC  = tmpN;
                    const float2 tmpUv = uvB; uvB = uvC; uvC = tmpUv;
                }

                // Step 8: edge-function barycentrics at the pixel centre.
                const float2 pc2 = float2((float)pixelCoord.x + 0.5f,
                                          (float)pixelCoord.y + 0.5f);
                const float e0 = (pix1.x - pix0.x) * (pc2.y - pix0.y)
                               - (pix1.y - pix0.y) * (pc2.x - pix0.x);
                const float e1 = (pix2.x - pix1.x) * (pc2.y - pix1.y)
                               - (pix2.y - pix1.y) * (pc2.x - pix1.x);
                const float e2 = (pix0.x - pix2.x) * (pc2.y - pix2.y)
                               - (pix0.y - pix2.y) * (pc2.x - pix2.x);
                const float area = e0 + e1 + e2;
                float w0, w1, w2;
                if (area > 0.0f) {
                    const float invA = 1.0f / area;
                    w0 = e1 * invA;  // weight for vertex 0
                    w1 = e2 * invA;  // weight for vertex 1
                    w2 = e0 * invA;  // weight for vertex 2
                } else {
                    // Degenerate/off-edge — centre the weights so the normal
                    // interpolation still produces a valid output.
                    w0 = 1.0f / 3.0f; w1 = 1.0f / 3.0f; w2 = 1.0f / 3.0f;
                }

                // Step 9: interpolate world normal + normalize. A near-zero
                // sum defends against three nearly-opposing normals averaging
                // to zero.
                float3 N = nA * w0 + nB * w1 + nC * w2;
                const float Nlen = length(N);
                if (Nlen > 1.0e-6f) N /= Nlen;
                else                N = float3(0.0f, 1.0f, 0.0f);

                // Step 9b: interpolate UV from the winding-swapped triangle.
                const float2 mpUV = uvA * w0 + uvB * w1 + uvC * w2;

                // Step 9c: resolve the source material for this cluster. The
                // ClusterOnDisk stores materialIndex at byte offset 68 (see
                // asset::ClusterOnDisk in MpAssetFormat.h). loadMaterial()
                // mirrors the non-mp branch below.
                const uint mpMaterialIdx =
                    pageBuf.Load(pageByteOffset + clusterOnDiskOff + 68u);
                GpuMaterial mpMat = loadMaterial(pc.materialBufferSlot, mpMaterialIdx);

                // Analytic UV gradients so SampleGrad picks the correct mip
                // level. Compute shaders have no implicit ddx/ddy, so
                // SampleLevel(..., 0) forced base-mip everywhere — which
                // produced per-pixel texture aliasing that got worse as the
                // BMW shrank on screen (Nyquist failures on high-frequency
                // paint / bodywork textures). Solve the 2x2 screen-to-UV
                // linear system over the projected triangle:
                //   [ (pix1-pix0).x  (pix1-pix0).y ] [ dUdx ]   [ uvB-uvA ]
                //   [ (pix2-pix0).x  (pix2-pix0).y ] [ dUdy ] = [ uvC-uvA ]
                // Degenerate fallback (determinant near zero) uses zero
                // gradients = mip0; the triangle is sub-pixel anyway so
                // aliasing would be mild regardless.
                const float2 dP1 = pix1 - pix0;
                const float2 dP2 = pix2 - pix0;
                const float2 dUV1 = uvB - uvA;
                const float2 dUV2 = uvC - uvA;
                const float det   = dP1.x * dP2.y - dP1.y * dP2.x;
                float2 mpDUVdx = float2(0.0f, 0.0f);
                float2 mpDUVdy = float2(0.0f, 0.0f);
                if (abs(det) > 1.0e-8f) {
                    const float invDet = 1.0f / det;
                    mpDUVdx = (dUV1 * dP2.y - dUV2 * dP1.y) * invDet;
                    mpDUVdy = (dUV2 * dP1.x - dUV1 * dP2.x) * invDet;
                }

                // Sample albedo with analytic gradients.
                float4 mpBaseColor = mpMat.baseColorFactor;
                if (mpMat.baseColorTexIdx != INVALID_TEX) {
                    Texture2D    mpTex  = g_textures[NonUniformResourceIndex(mpMat.baseColorTexIdx)];
                    SamplerState mpSamp = g_samplers[NonUniformResourceIndex(mpMat.samplerSlot)];
                    mpBaseColor *= mpTex.SampleGrad(mpSamp, mpUV, mpDUVdx, mpDUVdy);
                }

                // Sample metallic-roughness (glTF: B=metallic, G=roughness).
                float mpMetal = mpMat.metallicFactor;
                float mpRough = mpMat.roughnessFactor;
                if (mpMat.metalRoughTexIdx != INVALID_TEX) {
                    Texture2D    mrTex  = g_textures[NonUniformResourceIndex(mpMat.metalRoughTexIdx)];
                    SamplerState mrSamp = g_samplers[NonUniformResourceIndex(mpMat.samplerSlot)];
                    const float4 mr = mrTex.SampleGrad(mrSamp, mpUV, mpDUVdx, mpDUVdy);
                    mpMetal *= mr.b;
                    mpRough *= mr.g;
                }

                // Step 10: write G-buffer. Motion vectors zero out — M5 doesn't
                // reconstruct prev-frame positions yet (would need per-cluster
                // instance transforms to differ across frames).
                const float4 mpAlbedo     = float4(mpBaseColor.rgb, 1.0f);
                const float4 mpNormal     = float4(N * 0.5f + 0.5f, 0.0f);
                const float4 mpMetalRough = float4(mpMetal, mpRough, 0.0f, 0.0f);
                const float4 mpMotion     = float4(0.0f, 0.0f, 0.0f, 0.0f);
                g_storageImages[NonUniformResourceIndex(pc.albedoStorageSlot)][pixelCoord]     = mpAlbedo;
                g_storageImages[NonUniformResourceIndex(pc.normalStorageSlot)][pixelCoord]     = mpNormal;
                g_storageImages[NonUniformResourceIndex(pc.metalRoughStorageSlot)][pixelCoord] = mpMetalRough;
                g_storageImages[NonUniformResourceIndex(pc.motionVecStorageSlot)][pixelCoord]  = mpMotion;
                return;
            }
        }
    }
#endif

    // Background pixels are marked with INVALID_VIS.
    if (visPacked == INVALID_VIS) {
        g_storageImages[NonUniformResourceIndex(pc.albedoStorageSlot)][pixelCoord]     = float4(0, 0, 0, 1);
        g_storageImages[NonUniformResourceIndex(pc.normalStorageSlot)][pixelCoord]     = float4(0.5, 0.5, 1.0, 0);
        g_storageImages[NonUniformResourceIndex(pc.metalRoughStorageSlot)][pixelCoord] = float4(0, 0.5, 0, 0);
        g_storageImages[NonUniformResourceIndex(pc.motionVecStorageSlot)][pixelCoord]  = float4(0, 0, 0, 0);
        return;
    }

    // Decode visibility — encoding set by visibility_buffer.mesh.hlsl:
    //   [31:7] globalMeshletId (25 bits, direct index into the meshlet buffer)
    //   [6:0]  localTriId      (7 bits, MAX_TRIANGLES=124 fits)
    uint globalMeshletId = visPacked >> 7u;
    uint localTriId      = visPacked & 0x7Fu;

    // O(1) reverse lookup: globalMeshletId → instanceId. Replaces the former
    // O(n_instances) per-pixel scan — that loop dominated frame time at higher
    // instance counts (e.g. ~10 ms at 1080p × 3000 instances before the fix).
    // The lookup buffer is indexed as StructuredBuffer<float4>, four u32s per
    // entry.  0xFFFFFFFFu means "no current instance owns this meshlet" —
    // should not normally occur for a live vis-buffer write, but we treat it
    // like a background pixel defensively.
    uint instanceId;
    {
        StructuredBuffer<float4> lookupBuf = g_buffers[NonUniformResourceIndex(pc.meshletToInstanceSlot)];
        uint float4Idx = globalMeshletId / 4u;
        uint component = globalMeshletId & 3u;
        uint4 packed   = asuint(lookupBuf[float4Idx]);
        instanceId = (component == 0u) ? packed.x
                   : (component == 1u) ? packed.y
                   : (component == 2u) ? packed.z
                                       : packed.w;
    }
    if (instanceId == 0xFFFFFFFFu) {
        g_storageImages[NonUniformResourceIndex(pc.albedoStorageSlot)][pixelCoord]     = float4(0, 0, 0, 1);
        g_storageImages[NonUniformResourceIndex(pc.normalStorageSlot)][pixelCoord]     = float4(0.5, 0.5, 1.0, 0);
        g_storageImages[NonUniformResourceIndex(pc.metalRoughStorageSlot)][pixelCoord] = float4(0, 0.5, 0, 0);
        g_storageImages[NonUniformResourceIndex(pc.motionVecStorageSlot)][pixelCoord]  = float4(0, 0, 0, 0);
        return;
    }

    GpuInstance inst = loadInstance(instanceId);
    Meshlet meshlet  = loadMeshlet(globalMeshletId);

    // Load 3 local triangle vertex indices and remap to global vertex indices.
    uint3 triLocalIdx = loadTriangleIndices(pc.meshletTrianglesSlot,
                                            meshlet.triangle_offset, localTriId);

    uint globalVert0 = loadMeshletVertexIndex(pc.meshletVerticesSlot,
                                              meshlet.vertex_offset + triLocalIdx.x);
    uint globalVert1 = loadMeshletVertexIndex(pc.meshletVerticesSlot,
                                              meshlet.vertex_offset + triLocalIdx.y);
    uint globalVert2 = loadMeshletVertexIndex(pc.meshletVerticesSlot,
                                              meshlet.vertex_offset + triLocalIdx.z);

    // Load full vertex attributes.
    VertexAttribs v0 = loadVertex(inst.vertex_buffer_slot, globalVert0);
    VertexAttribs v1 = loadVertex(inst.vertex_buffer_slot, globalVert1);
    VertexAttribs v2 = loadVertex(inst.vertex_buffer_slot, globalVert2);

    CameraData cam = loadCamera(pc.cameraSlot);

    // Transform positions to clip space.
    float4 worldPos0 = mul(inst.transform, float4(v0.position, 1.0));
    float4 worldPos1 = mul(inst.transform, float4(v1.position, 1.0));
    float4 worldPos2 = mul(inst.transform, float4(v2.position, 1.0));

    float4 clipPos0 = mul(cam.viewProj, worldPos0);
    float4 clipPos1 = mul(cam.viewProj, worldPos1);
    float4 clipPos2 = mul(cam.viewProj, worldPos2);

    // NDC after perspective divide.
    float2 ndc0 = clipPos0.xy / clipPos0.w;
    float2 ndc1 = clipPos1.xy / clipPos1.w;
    float2 ndc2 = clipPos2.xy / clipPos2.w;

    // Pixel NDC (Vulkan: x in [-1,1], y in [-1,1], pixel center at +0.5).
    float2 pixelNDC = float2(
        (float(pixelCoord.x) + 0.5) / float(pc.screenWidth)  *  2.0 - 1.0,
        (float(pixelCoord.y) + 0.5) / float(pc.screenHeight) *  2.0 - 1.0
    );

    // Screen-space barycentrics then perspective-correct.
    float3 screenBary = computeBarycentrics(pixelNDC, ndc0, ndc1, ndc2);
    float3 bary = perspectiveCorrectBarycentrics(screenBary, clipPos0.w, clipPos1.w, clipPos2.w);

    // Interpolate vertex attributes.
    float2 uv = v0.uv * bary.x + v1.uv * bary.y + v2.uv * bary.z;

    // Compute UV screen-space gradients analytically so SampleGrad can drive
    // mipmap selection + anisotropic filtering. Compute shaders have no
    // implicit ddx/ddy, and SampleLevel(..., 0) forces base-mip sampling
    // regardless of sampler state — that's why texture aliasing persisted
    // until this change. We reconstruct barycentrics at pixel+(1,0) and
    // pixel+(0,1) using the same triangle NDC positions, interpolate the UV
    // at each, and take finite differences.
    float2 pixelNDCx = float2(
        (float(pixelCoord.x) + 1.5) / float(pc.screenWidth)  * 2.0 - 1.0,
        (float(pixelCoord.y) + 0.5) / float(pc.screenHeight) * 2.0 - 1.0);
    float2 pixelNDCy = float2(
        (float(pixelCoord.x) + 0.5) / float(pc.screenWidth)  * 2.0 - 1.0,
        (float(pixelCoord.y) + 1.5) / float(pc.screenHeight) * 2.0 - 1.0);
    float3 screenBaryX = computeBarycentrics(pixelNDCx, ndc0, ndc1, ndc2);
    float3 screenBaryY = computeBarycentrics(pixelNDCy, ndc0, ndc1, ndc2);
    float3 baryX = perspectiveCorrectBarycentrics(screenBaryX, clipPos0.w, clipPos1.w, clipPos2.w);
    float3 baryY = perspectiveCorrectBarycentrics(screenBaryY, clipPos0.w, clipPos1.w, clipPos2.w);
    float2 uvX = v0.uv * baryX.x + v1.uv * baryX.y + v2.uv * baryX.z;
    float2 uvY = v0.uv * baryY.x + v1.uv * baryY.y + v2.uv * baryY.z;
    float2 ddx_uv = uvX - uv;
    float2 ddy_uv = uvY - uv;

    float3x3 normalMat = (float3x3)inst.transform;
    float3 N0 = normalize(mul(normalMat, v0.normal));
    float3 N1 = normalize(mul(normalMat, v1.normal));
    float3 N2 = normalize(mul(normalMat, v2.normal));
    float3 N  = normalize(N0 * bary.x + N1 * bary.y + N2 * bary.z);

    float3 T0 = normalize(mul(normalMat, v0.tangent));
    float3 T1 = normalize(mul(normalMat, v1.tangent));
    float3 T2 = normalize(mul(normalMat, v2.tangent));
    float3 T  = normalize(T0 * bary.x + T1 * bary.y + T2 * bary.z);
    float  bitSign = v0.bitangentSign * bary.x + v1.bitangentSign * bary.y + v2.bitangentSign * bary.z;

    // Interpolate world position.
    float3 worldPos = worldPos0.xyz * bary.x + worldPos1.xyz * bary.y + worldPos2.xyz * bary.z;

    // Load material and evaluate PBR.
    GpuMaterial mat  = loadMaterial(pc.materialBufferSlot, inst.material_index);

    // Declare G-buffer output variables.
    float4 albedo;
    float3 normal;
    float2 metalRough;
    float2 motionVec;

    if (mat.flags & MATERIAL_FLAG_TERRAIN) {
        // TODO Phase 4 complete: reconstruct normal from heightmap gradient
        // For now: flat grey albedo, up-normal, zero metalRough, zero motionVec
        albedo     = float4(0.45, 0.45, 0.45, 1.0);
        normal     = float3(0.0, 1.0, 0.0);
        metalRough = float2(0.0, 0.9);
        motionVec  = float2(0.0, 0.0);
    } else {
        SamplerState samp = g_samplers[NonUniformResourceIndex(mat.samplerSlot)];

        // Base color.
        float4 baseColor = mat.baseColorFactor;
        if (mat.baseColorTexIdx != INVALID_TEX) {
            baseColor *= g_textures[NonUniformResourceIndex(mat.baseColorTexIdx)].SampleGrad(samp, uv, ddx_uv, ddy_uv);
        }

        // Metallic + roughness (glTF: G=roughness, B=metallic).
        float metallic  = mat.metallicFactor;
        float roughness = mat.roughnessFactor;
        if (mat.metalRoughTexIdx != INVALID_TEX) {
            float4 mr = g_textures[NonUniformResourceIndex(mat.metalRoughTexIdx)].SampleGrad(samp, uv, ddx_uv, ddy_uv);
            roughness *= mr.g;
            metallic  *= mr.b;
        }

        // Normal mapping via TBN.
        if (mat.normalTexIdx != INVALID_TEX) {
            float3 B = normalize(cross(N, T) * bitSign);
            float3x3 TBN = float3x3(T, B, N);

            float4 normalSample = g_textures[NonUniformResourceIndex(mat.normalTexIdx)].SampleGrad(samp, uv, ddx_uv, ddy_uv);
            float3 tn = normalSample.xyz * 2.0 - 1.0;
            tn.xy    *= mat.normalScale;
            tn.y      = -tn.y; // glTF +Y flip for DX/Vulkan NDC
            N = normalize(mul(tn, TBN));
        }

        // Ambient occlusion.
        float occlusion = 1.0;
        if (mat.occlusionTexIdx != INVALID_TEX) {
            float4 occSample = g_textures[NonUniformResourceIndex(mat.occlusionTexIdx)].SampleGrad(samp, uv, ddx_uv, ddy_uv);
            occlusion = lerp(1.0, occSample.r, mat.occlusionStrength);
        }

        // Motion vectors: NDC-space velocity (current - previous).
        float4 prevClipPos = mul(cam.prevViewProj, float4(worldPos, 1.0));
        float4 currClipPos = mul(cam.viewProj,     float4(worldPos, 1.0));
        float2 currentNDC  = currClipPos.xy / currClipPos.w;
        float2 prevNDC     = prevClipPos.xy / prevClipPos.w;

        albedo     = float4(baseColor.rgb, occlusion);
        normal     = N;
        metalRough = float2(metallic, roughness);
        motionVec  = currentNDC - prevNDC;
    }

    // Write G-buffer storage images.
    g_storageImages[NonUniformResourceIndex(pc.albedoStorageSlot)][pixelCoord]     = albedo;
    g_storageImages[NonUniformResourceIndex(pc.normalStorageSlot)][pixelCoord]     = float4(normal * 0.5 + 0.5, 0.0);
    g_storageImages[NonUniformResourceIndex(pc.metalRoughStorageSlot)][pixelCoord] = float4(metalRough, 0, 0);
    g_storageImages[NonUniformResourceIndex(pc.motionVecStorageSlot)][pixelCoord]  = float4(motionVec, 0, 0);
}
