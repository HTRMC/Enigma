// clustered_forward.hlsl
// ======================
// Forward-shading vertex + pixel shader for transparent geometry.
// Renders alpha-blended primitives onto the HDR colour buffer.
//
// Lighting: single directional sun (no light grid). PBR Cook-Torrance
// with the same material struct and bindless layout as GBufferFormats.h.
// Alpha is taken from baseColorFactor.a and the base colour texture.
//
// Vertex data layout: 3 float4 per vertex:
//   [vid*3+0] = (position.x, position.y, position.z, normal.x)
//   [vid*3+1] = (normal.y,   normal.z,   uv.x,       uv.y)
//   [vid*3+2] = (tangent.x,  tangent.y,  tangent.z,  tangent.w)

#include "common.hlsl"

// --- Bindless resource arrays ---
[[vk::binding(0, 0)]]
Texture2D g_textures[] : register(t0, space0);

[[vk::binding(2, 0)]]
StructuredBuffer<float4> g_buffers[] : register(t0, space1);

[[vk::binding(3, 0)]]
SamplerState g_samplers[] : register(s0, space0);

// --- Push constants (112 bytes) ---
struct PushBlock {
    float4x4 model;
    float4   sunDirIntensity;   // xyz=sun direction (unit), w=intensity
    float4   sunColor;          // xyz=linear RGB
    uint     vertexSlot;
    uint     cameraSlot;
    uint     materialBufferSlot;
    uint     materialIndex;
};

[[vk::push_constant]] PushBlock pc;

// --- Constants ---
static const uint INVALID_TEX           = 0xFFFFFFFFu;
static const uint FLAG_BLEND            = 0x1u; // bit 0 — matches Scene.h kFlagBlend
static const uint MATERIAL_FLAG_TERRAIN = 0x4u; // bit 2 — matches Scene.h kFlagTerrain
static const float PI                   = 3.14159265358979323846f;

// --- Material struct (std430, mirrors Scene.h Material) ---
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
    uint   _pad;
};

GpuMaterial loadMaterial() {
    StructuredBuffer<float4> buf = g_buffers[NonUniformResourceIndex(pc.materialBufferSlot)];
    const uint base = pc.materialIndex * 5; // 80 bytes / 16 bytes per float4 = 5 float4s
    GpuMaterial m;
    m.baseColorFactor = buf[base + 0];
    m.emissiveFactor  = buf[base + 1];
    float4 f2 = buf[base + 2];
    m.metallicFactor     = f2.x;
    m.roughnessFactor    = f2.y;
    m.normalScale        = f2.z;
    m.occlusionStrength  = f2.w;
    uint4 u3 = asuint(buf[base + 3]);
    m.baseColorTexIdx    = u3.x;
    m.metalRoughTexIdx   = u3.y;
    m.normalTexIdx       = u3.z;
    m.emissiveTexIdx     = u3.w;
    uint4 u4 = asuint(buf[base + 4]);
    m.occlusionTexIdx    = u4.x;
    m.flags              = u4.y;
    m.samplerSlot        = u4.z;
    m._pad               = u4.w;
    return m;
}

CameraData loadCamera() {
    StructuredBuffer<float4> buf = g_buffers[NonUniformResourceIndex(pc.cameraSlot)];
    CameraData cam;
    cam.view         = transpose(float4x4(buf[0],  buf[1],  buf[2],  buf[3]));
    cam.proj         = transpose(float4x4(buf[4],  buf[5],  buf[6],  buf[7]));
    cam.viewProj     = transpose(float4x4(buf[8],  buf[9],  buf[10], buf[11]));
    cam.prevViewProj = transpose(float4x4(buf[12], buf[13], buf[14], buf[15]));
    cam.invViewProj  = transpose(float4x4(buf[16], buf[17], buf[18], buf[19]));
    cam.worldPos     = buf[20];
    return cam;
}

// --- PBR helpers (Cook-Torrance GGX) ---
float distributionGGX(float NdotH, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float d  = NdotH * NdotH * (a2 - 1.0f) + 1.0f;
    return a2 / (PI * d * d + 1e-7f);
}

float geometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0f;
    float k = (r * r) / 8.0f;
    return NdotV / (NdotV * (1.0f - k) + k + 1e-7f);
}

float3 fresnelSchlick(float cosTheta, float3 F0) {
    return F0 + (1.0f - F0) * pow(saturate(1.0f - cosTheta), 5.0f);
}

// --- Vertex output ---
struct VSOutput {
    float4 sv_pos    : SV_Position;
    float3 worldPos  : TEXCOORD0;
    float3 normal    : TEXCOORD1;
    float2 uv        : TEXCOORD2;
    float4 tangent   : TEXCOORD3;
};

// --- Vertex shader ---
VSOutput VSMain(uint vertexIndex : SV_VertexID) {
    StructuredBuffer<float4> vbuf = g_buffers[NonUniformResourceIndex(pc.vertexSlot)];
    float4 d0 = vbuf[vertexIndex * 3 + 0]; // (px, py, pz, nx)
    float4 d1 = vbuf[vertexIndex * 3 + 1]; // (ny, nz, u, v)
    float4 d2 = vbuf[vertexIndex * 3 + 2]; // tangent xyzw

    float3 localPos = d0.xyz;
    float3 normal   = float3(d0.w, d1.xy);
    float2 uv       = d1.zw;
    float4 tangent  = d2;

    float4 worldPos = mul(pc.model, float4(localPos, 1.0f));

    CameraData cam = loadCamera();
    float3x3 normalMat = (float3x3)pc.model; // no non-uniform scale assumed

    VSOutput o;
    o.sv_pos   = mul(cam.viewProj, worldPos);
    o.worldPos = worldPos.xyz;
    o.normal   = normalize(mul(normalMat, normal));
    o.uv       = uv;
    o.tangent  = float4(normalize(mul(normalMat, tangent.xyz)), tangent.w);
    return o;
}

// --- Pixel shader ---
float4 PSMain(VSOutput input) : SV_Target0 {
    GpuMaterial mat = loadMaterial();
    SamplerState smp = g_samplers[NonUniformResourceIndex(mat.samplerSlot)];

    // Base colour + alpha.
    float4 baseColor = mat.baseColorFactor;
    if (mat.baseColorTexIdx != INVALID_TEX) {
        baseColor *= g_textures[NonUniformResourceIndex(mat.baseColorTexIdx)].Sample(smp, input.uv);
    }
    float alpha = baseColor.a;

    // Metallic / roughness.
    float metallic  = mat.metallicFactor;
    float roughness = mat.roughnessFactor;
    if (mat.metalRoughTexIdx != INVALID_TEX) {
        float4 mr = g_textures[NonUniformResourceIndex(mat.metalRoughTexIdx)].Sample(smp, input.uv);
        roughness *= mr.g;
        metallic  *= mr.b;
    }

    // Normal mapping.
    float3 N = normalize(input.normal);
    if (mat.normalTexIdx != INVALID_TEX) {
        float3 T   = normalize(input.tangent.xyz);
        float3 B   = normalize(cross(N, T) * input.tangent.w);
        float3 tN  = g_textures[NonUniformResourceIndex(mat.normalTexIdx)].Sample(smp, input.uv).xyz;
        tN         = tN * 2.0f - 1.0f;
        tN.xy     *= mat.normalScale;
        N          = normalize(mul(tN, float3x3(T, B, N)));
    }

    // Emissive.
    float3 emissive = mat.emissiveFactor.xyz;
    if (mat.emissiveTexIdx != INVALID_TEX) {
        emissive *= g_textures[NonUniformResourceIndex(mat.emissiveTexIdx)].Sample(smp, input.uv).rgb;
    }

    // PBR Cook-Torrance with single directional sun.
    float3 V   = normalize(pc.sunDirIntensity.xyz - input.worldPos); // view direction (approx)
    CameraData cam = loadCamera();
    V = normalize(cam.worldPos.xyz - input.worldPos);
    float3 L   = normalize(pc.sunDirIntensity.xyz);
    float3 H   = normalize(V + L);
    float3 F0  = lerp(float3(0.04f, 0.04f, 0.04f), baseColor.rgb, metallic);

    float NdotL = saturate(dot(N, L));
    float NdotV = saturate(dot(N, V));
    float NdotH = saturate(dot(N, H));
    float VdotH = saturate(dot(V, H));

    float  D   = distributionGGX(NdotH, roughness);
    float  G   = geometrySchlickGGX(NdotL, roughness) * geometrySchlickGGX(NdotV, roughness);
    float3 F   = fresnelSchlick(VdotH, F0);

    float3 kD  = (1.0f - F) * (1.0f - metallic);
    float3 spec = (D * G * F) / max(4.0f * NdotV * NdotL + 1e-7f, 1e-7f);
    float3 diffuse = kD * baseColor.rgb / PI;

    float3 sunLight    = pc.sunColor.xyz * pc.sunDirIntensity.w;
    float3 directLight = (diffuse + spec) * sunLight * NdotL;

    // Minimal ambient so shadowed transparents aren't completely black.
    float3 ambient = 0.03f * baseColor.rgb;

    float3 hdrColor = directLight + ambient + emissive;
    return float4(hdrColor, alpha);
}
