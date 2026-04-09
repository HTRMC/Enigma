// mesh.hlsl
// =========
// Bindless PBR mesh shader for Enigma. Cook-Torrance BRDF with normal mapping.
//
// Vertex data packed as StructuredBuffer<float4>, 3 entries per vertex:
//   [vid*3+0] = (position.x, position.y, position.z, normal.x)
//   [vid*3+1] = (normal.y,   normal.z,   uv.x,       uv.y)
//   [vid*3+2] = (tangent.x,  tangent.y,  tangent.z,  tangent.w)  // w = bitangent sign
//
// Material SSBO: 5 float4s per material (80 bytes, std430):
//   [0]  baseColorFactor (float4)
//   [1]  emissiveFactor.xyz + alphaCutoff.w (float4)
//   [2]  metallicFactor, roughnessFactor, normalScale, occlusionStrength (float4)
//   [3]  baseColorTexIdx, metalRoughTexIdx, normalTexIdx, emissiveTexIdx (uint4)
//   [4]  occlusionTexIdx, flags, samplerSlot, _pad (uint4)

#include "common.hlsl"

// --- Bindless resource arrays ---
[[vk::binding(2, 0)]]
StructuredBuffer<float4> g_buffers[] : register(t0, space1);

[[vk::binding(0, 0)]]
Texture2D g_textures[] : register(t0, space0);

[[vk::binding(3, 0)]]
SamplerState g_samplers[] : register(s0, space0);

// --- Push constants (112 bytes) ---
struct PushBlock {
    float4x4 model;
    uint     vertexSlot;
    uint     cameraSlot;
    uint     materialBufferSlot;
    uint     materialIndex;
    float4   lightDirIntensity;  // xyz = direction (prenormalized by CPU), w = intensity
    float4   lightColor;          // xyz = color, w = unused
};

[[vk::push_constant]] PushBlock pc;

// --- Constants ---
static const float PI          = 3.14159265359;
static const uint  INVALID_TEX = 0xFFFFFFFFu;
static const uint  FLAG_BLEND  = 1u;
static const uint  FLAG_MASK   = 2u;

// --- Material struct (mirrors CPU Material, read from SSBO) ---
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

// --- Camera load (column-major GLM → HLSL row-major) ---
CameraData loadCamera(uint slot) {
    StructuredBuffer<float4> buf = g_buffers[NonUniformResourceIndex(slot)];
    CameraData cam;
    cam.view     = transpose(float4x4(buf[0],  buf[1],  buf[2],  buf[3]));
    cam.proj     = transpose(float4x4(buf[4],  buf[5],  buf[6],  buf[7]));
    cam.viewProj = transpose(float4x4(buf[8],  buf[9],  buf[10], buf[11]));
    cam.worldPos = buf[12];
    return cam;
}

// --- PBR functions ---

// GGX normal distribution function.
float D_GGX(float NdotH, float alpha) {
    float a2 = alpha * alpha;
    float d  = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-7);
}

// Height-correlated Smith GGX visibility (Heitz 2014).
float V_SmithGGXCorrelated(float NdotV, float NdotL, float alpha) {
    float a2   = alpha * alpha;
    float GGXV = NdotL * sqrt(max(NdotV * NdotV * (1.0 - a2) + a2, 1e-7));
    float GGXL = NdotV * sqrt(max(NdotL * NdotL * (1.0 - a2) + a2, 1e-7));
    return 0.5 / max(GGXV + GGXL, 1e-7);
}

// Schlick Fresnel.
float3 F_Schlick(float VdotH, float3 F0) {
    return F0 + (1.0 - F0) * pow(saturate(1.0 - VdotH), 5.0);
}

// ACES tone mapping approximation (Narkowicz 2015).
float3 ACES(float3 x) {
    return saturate((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14));
}

// --- Vertex shader ---
struct VSOut {
    float4 pos           : SV_Position;
    float3 worldPos      : TEXCOORD0;
    float3 normal        : TEXCOORD1;
    float2 uv            : TEXCOORD2;
    float3 tangent       : TEXCOORD3;
    float  bitangentSign : TEXCOORD4;
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

    VSOut o;
    o.pos           = mul(cam.viewProj, worldPos);
    o.worldPos      = worldPos.xyz;
    o.normal        = normalize(mul(normalMat, normal));
    o.tangent       = normalize(mul(normalMat, tangent));
    o.bitangentSign = bitSign;
    o.uv            = uv;
    return o;
}

// --- Fragment shader ---
float4 PSMain(VSOut vs) : SV_Target {
    GpuMaterial mat = loadMaterial(pc.materialBufferSlot, pc.materialIndex);
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

    // Metallic + roughness (G=roughness, B=metallic per glTF spec)
    float metallic  = mat.metallicFactor;
    float roughness = mat.roughnessFactor;
    if (mat.metalRoughTexIdx != INVALID_TEX) {
        float4 mr = g_textures[NonUniformResourceIndex(mat.metalRoughTexIdx)].Sample(samp, vs.uv);
        roughness *= mr.g;
        metallic  *= mr.b;
    }
    // Squaring roughness: perceptual → linear (critical — don't skip this)
    float alpha = roughness * roughness;

    // Normal mapping via TBN
    float3 N = normalize(vs.normal);
    if (mat.normalTexIdx != INVALID_TEX) {
        float3 T = normalize(vs.tangent);
        float3 B = normalize(cross(N, T) * vs.bitangentSign);
        float3x3 TBN = float3x3(T, B, N); // tangent-space → world-space

        float4 normalSample = g_textures[NonUniformResourceIndex(mat.normalTexIdx)].Sample(samp, vs.uv);
        float3 tn = normalSample.xyz * 2.0 - 1.0;
        tn.xy    *= mat.normalScale;
        tn.y      = -tn.y; // glTF uses OpenGL +Y convention; flip for DX/Vulkan NDC
        N = normalize(mul(tn, TBN));
    }

    // Occlusion (R channel)
    float occlusion = 1.0;
    if (mat.occlusionTexIdx != INVALID_TEX) {
        float4 occSample = g_textures[NonUniformResourceIndex(mat.occlusionTexIdx)].Sample(samp, vs.uv);
        occlusion = lerp(1.0, occSample.r, mat.occlusionStrength);
    }

    // Directional sun light from push constants.
    float3 lightDir   = normalize(pc.lightDirIntensity.xyz);
    float3 lightColor = pc.lightColor.xyz * pc.lightDirIntensity.w;

    float3 V = normalize(loadCamera(pc.cameraSlot).worldPos.xyz - vs.worldPos);
    float3 L = lightDir;
    float3 H = normalize(V + L);

    float NdotL = saturate(dot(N, L));
    float NdotV = saturate(dot(N, V)) + 1e-5;
    float NdotH = saturate(dot(N, H));
    float VdotH = saturate(dot(V, H));

    // Cook-Torrance specular
    float3 F0      = lerp(float3(0.04, 0.04, 0.04), baseColor.rgb, metallic);
    float  D       = D_GGX(NdotH, alpha);
    float  Vis     = V_SmithGGXCorrelated(NdotV, NdotL, alpha);
    float3 F       = F_Schlick(VdotH, F0);
    float3 specular = D * Vis * F;

    // Lambertian diffuse (energy conserving: metals have no diffuse)
    float3 kD      = (1.0 - F) * (1.0 - metallic);
    float3 diffuse = kD * baseColor.rgb / PI;

    float3 Lo = (diffuse + specular) * lightColor * NdotL;

    // Ambient (IBL placeholder — simple constant)
    float3 ambient = 0.03 * baseColor.rgb * occlusion;

    // Emissive
    float3 emissive = mat.emissiveFactor.rgb;
    if (mat.emissiveTexIdx != INVALID_TEX) {
        emissive *= g_textures[NonUniformResourceIndex(mat.emissiveTexIdx)].Sample(samp, vs.uv).rgb;
    }

    float3 color = ambient + Lo + emissive;

    // ACES tone mapping
    color = ACES(color);

    return float4(color, baseColor.a);
}
