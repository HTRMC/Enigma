// debug_unlit.hlsl — Unlit debug view: outputs raw G-buffer albedo

#include "common.hlsl"

[[vk::binding(0, 0)]] Texture2D g_textures[] : register(t0, space0);
[[vk::binding(3, 0)]] SamplerState g_samplers[] : register(s0, space0);

struct PushBlock {
    uint albedoSlot;
    uint normalSlot;      // unused
    uint metalRoughSlot;  // unused
    uint depthSlot;
    uint cameraSlot;      // unused
    uint samplerSlot;
    uint _pad0;
    uint _pad1;
    float4 _lightDirIntensity; // unused
    float4 _lightColor;        // unused
};
[[vk::push_constant]] PushBlock pc;

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
    SamplerState samp = g_samplers[NonUniformResourceIndex(pc.samplerSlot)];
    float2 uv = vs.uv;
    float depth = g_textures[NonUniformResourceIndex(pc.depthSlot)].Sample(samp, uv).r;
    if (depth == 0.0)
        return float4(0.02, 0.02, 0.05, 1.0);
    float3 albedo = g_textures[NonUniformResourceIndex(pc.albedoSlot)].Sample(samp, uv).rgb;
    return float4(albedo, 1.0);
}
