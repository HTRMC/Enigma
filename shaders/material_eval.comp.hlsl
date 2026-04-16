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

// --- Bindless resource arrays ---
[[vk::binding(0, 0)]]
Texture2D g_textures[] : register(t0, space0);

[[vk::binding(1, 0)]]
RWTexture2D<float4> g_storageImages[] : register(u0, space0);

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
    uint instanceCount;  // number of GpuInstance entries — for meshlet→instance walk
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

    // Find which instance owns this global meshlet by range-checking meshlet_offset.
    // O(n_instances) but n_instances is typically small; no extra lookup buffer needed.
    uint instanceId = 0;
    for (uint i = 0; i < pc.instanceCount; ++i) {
        GpuInstance testInst = loadInstance(i);
        if (globalMeshletId >= testInst.meshlet_offset &&
            globalMeshletId <  testInst.meshlet_offset + testInst.meshlet_count) {
            instanceId = i;
            break;
        }
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
            baseColor *= g_textures[NonUniformResourceIndex(mat.baseColorTexIdx)].SampleLevel(samp, uv, 0);
        }

        // Metallic + roughness (glTF: G=roughness, B=metallic).
        float metallic  = mat.metallicFactor;
        float roughness = mat.roughnessFactor;
        if (mat.metalRoughTexIdx != INVALID_TEX) {
            float4 mr = g_textures[NonUniformResourceIndex(mat.metalRoughTexIdx)].SampleLevel(samp, uv, 0);
            roughness *= mr.g;
            metallic  *= mr.b;
        }

        // Normal mapping via TBN.
        if (mat.normalTexIdx != INVALID_TEX) {
            float3 B = normalize(cross(N, T) * bitSign);
            float3x3 TBN = float3x3(T, B, N);

            float4 normalSample = g_textures[NonUniformResourceIndex(mat.normalTexIdx)].SampleLevel(samp, uv, 0);
            float3 tn = normalSample.xyz * 2.0 - 1.0;
            tn.xy    *= mat.normalScale;
            tn.y      = -tn.y; // glTF +Y flip for DX/Vulkan NDC
            N = normalize(mul(tn, TBN));
        }

        // Ambient occlusion.
        float occlusion = 1.0;
        if (mat.occlusionTexIdx != INVALID_TEX) {
            float4 occSample = g_textures[NonUniformResourceIndex(mat.occlusionTexIdx)].SampleLevel(samp, uv, 0);
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
