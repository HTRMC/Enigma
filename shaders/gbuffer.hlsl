// gbuffer.hlsl
// ============
// Deferred geometry pass. Identical vertex transform to mesh.hlsl but the
// pixel shader writes to four MRT targets instead of computing lighting:
//
//   SV_Target0  albedo      VK_FORMAT_R8G8B8A8_UNORM
//               rgb = baseColor, a = ambient occlusion
//   SV_Target1  normal      VK_FORMAT_A2B10G10R10_UNORM_PACK32
//               rgb = world-space normal packed to [0,1]
//   SV_Target2  metalRough  VK_FORMAT_R8G8_UNORM
//               r = metallic, g = roughness (perceptual, not squared)
//   SV_Target3  motionVec   VK_FORMAT_R16G16_SFLOAT
//               rg = NDC-space velocity (current − previous clip pos / w)
//
// Depth writes to VK_FORMAT_D32_SFLOAT (reverse-Z, far = 0).

#include "common.hlsl"

// --- Bindless resource arrays ---
[[vk::binding(2, 0)]]
StructuredBuffer<float4> g_buffers[] : register(t0, space1);

[[vk::binding(0, 0)]]
Texture2D g_textures[] : register(t0, space0);

[[vk::binding(3, 0)]]
SamplerState g_samplers[] : register(s0, space0);

// --- Push constants (80 bytes) ---
struct PushBlock {
    float4x4 model;
    uint     vertexSlot;
    uint     cameraSlot;
    uint     materialBufferSlot;
    uint     materialIndex;
};

[[vk::push_constant]] PushBlock pc;

// --- Constants ---
static const uint INVALID_TEX = 0xFFFFFFFFu;
static const uint FLAG_BLEND  = 1u;
static const uint FLAG_MASK   = 2u;

// --- Material struct (mirrors GpuMaterial in mesh.hlsl) ---
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

// --- Vertex shader ---
struct VSOut {
    float4 pos           : SV_Position;
    float3 worldPos      : TEXCOORD0;
    float3 normal        : TEXCOORD1;
    float2 uv            : TEXCOORD2;
    float3 tangent       : TEXCOORD3;
    float  bitangentSign : TEXCOORD4;
    // Current and previous clip-space positions for motion vector computation.
    float4 clipPos       : TEXCOORD5;
    float4 prevClipPos   : TEXCOORD6;
};

VSOut VSMain(uint vid : SV_VertexID) {
    StructuredBuffer<float4> vbuf = g_buffers[NonUniformResourceIndex(pc.vertexSlot)];
    float4 d0 = vbuf[vid * 3 + 0];
    float4 d1 = vbuf[vid * 3 + 1];
    float4 d2 = vbuf[vid * 3 + 2];

    float3 position = d0.xyz;
    float3 normal   = float3(d0.w, d1.x, d1.y);
    float2 uv       = float2(d1.z, d1.w);
    float3 tangent  = d2.xyz;
    float  bitSign  = d2.w;

    CameraData cam  = loadCamera(pc.cameraSlot);
    float4 worldPos = mul(pc.model, float4(position, 1.0));

    float3x3 normalMat = (float3x3)pc.model;

    float4 clipPos     = mul(cam.viewProj,     worldPos);
    float4 prevClipPos = mul(cam.prevViewProj, worldPos);

    VSOut o;
    o.pos           = clipPos;
    o.worldPos      = worldPos.xyz;
    o.normal        = normalize(mul(normalMat, normal));
    o.tangent       = normalize(mul(normalMat, tangent));
    o.bitangentSign = bitSign;
    o.uv            = uv;
    o.clipPos       = clipPos;
    o.prevClipPos   = prevClipPos;
    return o;
}

// --- G-buffer output struct ---
struct GBufferOut {
    float4 albedo     : SV_Target0; // rgb=baseColor, a=occlusion
    float4 normal     : SV_Target1; // rgb=world normal packed to [0,1], a=unused
    float2 metalRough : SV_Target2; // r=metallic, g=roughness (perceptual)
    float2 motionVec  : SV_Target3; // rg=NDC-space velocity
};

// --- Pixel shader ---
GBufferOut PSMain(VSOut vs) {
    GpuMaterial mat  = loadMaterial(pc.materialBufferSlot, pc.materialIndex);
    SamplerState samp = g_samplers[NonUniformResourceIndex(mat.samplerSlot)];

    // Base color
    float4 baseColor = mat.baseColorFactor;
    if (mat.baseColorTexIdx != INVALID_TEX) {
        baseColor *= g_textures[NonUniformResourceIndex(mat.baseColorTexIdx)].Sample(samp, vs.uv);
    }

    // Alpha discard (MASK mode)
    if (mat.flags & FLAG_MASK) {
        if (baseColor.a < mat.emissiveFactor.w) discard;
    }

    // Metallic + roughness (glTF: G=roughness, B=metallic)
    float metallic  = mat.metallicFactor;
    float roughness = mat.roughnessFactor;
    if (mat.metalRoughTexIdx != INVALID_TEX) {
        float4 mr = g_textures[NonUniformResourceIndex(mat.metalRoughTexIdx)].Sample(samp, vs.uv);
        roughness *= mr.g;
        metallic  *= mr.b;
    }

    // World-space normal via TBN
    float3 N = normalize(vs.normal);
    if (mat.normalTexIdx != INVALID_TEX) {
        float3 T = normalize(vs.tangent);
        float3 B = normalize(cross(N, T) * vs.bitangentSign);
        float3x3 TBN = float3x3(T, B, N);

        float4 normalSample = g_textures[NonUniformResourceIndex(mat.normalTexIdx)].Sample(samp, vs.uv);
        float3 tn = normalSample.xyz * 2.0 - 1.0;
        tn.xy    *= mat.normalScale;
        tn.y      = -tn.y; // glTF +Y flip for DX/Vulkan NDC
        N = normalize(mul(tn, TBN));
    }

    // Ambient occlusion (R channel, packed into albedo.a)
    float occlusion = 1.0;
    if (mat.occlusionTexIdx != INVALID_TEX) {
        float4 occSample = g_textures[NonUniformResourceIndex(mat.occlusionTexIdx)].Sample(samp, vs.uv);
        occlusion = lerp(1.0, occSample.r, mat.occlusionStrength);
    }

    // Motion vector: NDC-space velocity (current − previous)
    float2 currentNDC = vs.clipPos.xy    / vs.clipPos.w;
    float2 prevNDC    = vs.prevClipPos.xy / vs.prevClipPos.w;
    float2 motion     = currentNDC - prevNDC;

    GBufferOut o;
    o.albedo     = float4(baseColor.rgb, occlusion);
    o.normal     = float4(N.xyz * 0.5 + 0.5, 0.0); // encode [-1,1] → [0,1]
    o.metalRough = float2(metallic, roughness);
    o.motionVec  = motion;
    return o;
}
