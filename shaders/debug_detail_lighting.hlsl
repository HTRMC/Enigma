// debug_detail_lighting.hlsl — Detail Lighting: full PBR on white material

#include "common.hlsl"

[[vk::binding(0, 0)]] Texture2D g_textures[] : register(t0, space0);
[[vk::binding(2, 0)]] StructuredBuffer<float4> g_buffers[] : register(t0, space1);
[[vk::binding(3, 0)]] SamplerState g_samplers[] : register(s0, space0);

struct PushBlock {
    uint  albedoSlot;       // unused (we override material)
    uint  normalSlot;
    uint  metalRoughSlot;   // unused (we override)
    uint  depthSlot;
    uint  cameraSlot;
    uint  samplerSlot;
    uint  _pad0;
    uint  _pad1;
    float4 lightDirIntensity;
    float4 lightColor;
};
[[vk::push_constant]] PushBlock pc;

static const float PI = 3.14159265359;

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

float D_GGX(float NdotH, float alpha) {
    float a2 = alpha * alpha;
    float d  = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-7);
}

float V_SmithGGXCorrelated(float NdotV, float NdotL, float alpha) {
    float a2   = alpha * alpha;
    float GGXV = NdotL * sqrt(max(NdotV * NdotV * (1.0 - a2) + a2, 1e-7));
    float GGXL = NdotV * sqrt(max(NdotL * NdotL * (1.0 - a2) + a2, 1e-7));
    return 0.5 / max(GGXV + GGXL, 1e-7);
}

float3 F_Schlick(float VdotH, float3 F0) {
    return F0 + (1.0 - F0) * pow(saturate(1.0 - VdotH), 5.0);
}

struct VSOut {
    float4 pos : SV_Position;
    float2 uv  : TEXCOORD0;
};

VSOut VSMain(uint vid : SV_VertexID) {
    float2 uv = float2((vid << 1) & 2, vid & 2);
    VSOut o;
    o.pos = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    o.uv  = uv;
    return o;
}

float4 PSMain(VSOut vs) : SV_Target {
    float2 uv = vs.uv;
    SamplerState samp = g_samplers[NonUniformResourceIndex(pc.samplerSlot)];

    float4 normalSample = g_textures[NonUniformResourceIndex(pc.normalSlot)].Sample(samp, uv);
    float  depth        = g_textures[NonUniformResourceIndex(pc.depthSlot)].Sample(samp, uv).r;

    if (depth == 0.0)
        return float4(0.02, 0.02, 0.05, 1.0);

    // Reconstruct world position.
    float2 ndc    = uv * 2.0 - 1.0;
    float4 ndcPos = float4(ndc.x, ndc.y, depth, 1.0);
    CameraData cam = loadCamera(pc.cameraSlot);
    float4 worldPos4 = mul(cam.invViewProj, ndcPos);
    float3 worldPos  = worldPos4.xyz / worldPos4.w;

    float3 N = normalize(normalSample.rgb * 2.0 - 1.0);

    // Override material: white dielectric at mid roughness.
    float3 baseColor = float3(1.0, 1.0, 1.0);
    float  metallic  = 0.0;
    float  roughness = 0.5;
    float  alpha     = roughness * roughness;

    float3 lightDir   = normalize(pc.lightDirIntensity.xyz);
    float3 lightColor = pc.lightColor.xyz * pc.lightDirIntensity.w;

    float3 V = normalize(cam.worldPos.xyz - worldPos);
    float3 L = lightDir;
    float3 H = normalize(V + L);

    float NdotL = saturate(dot(N, L));
    float NdotV = saturate(dot(N, V)) + 1e-5;
    float NdotH = saturate(dot(N, H));
    float VdotH = saturate(dot(V, H));

    float3 F0      = lerp(float3(0.04, 0.04, 0.04), baseColor, metallic);
    float  D       = D_GGX(NdotH, alpha);
    float  Vis     = V_SmithGGXCorrelated(NdotV, NdotL, alpha);
    float3 F       = F_Schlick(VdotH, F0);
    float3 specular = D * Vis * F;

    float3 kD      = (1.0 - F) * (1.0 - metallic);
    float3 diffuse = kD * baseColor / PI;

    float3 Lo = (diffuse + specular) * lightColor * NdotL;
    float3 ambient = 0.03 * baseColor;

    return float4(ambient + Lo, 1.0);
}
