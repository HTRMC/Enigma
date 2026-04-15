// debug_blit.hlsl — Blit HDR intermediate to swapchain with simple Reinhard tonemap

#include "common.hlsl"

[[vk::binding(0, 0)]] Texture2D g_textures[] : register(t0, space0);
[[vk::binding(3, 0)]] SamplerState g_samplers[] : register(s0, space0);

struct PushBlock {
    uint _unused0;
    uint _unused1;
    uint _unused2;
    uint _unused3;
    uint _unused4;
    uint samplerSlot;
    uint hdrSlot;
    uint _pad;
    float4 _a;
    float4 _b;
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
    float3 hdr = g_textures[NonUniformResourceIndex(pc.hdrSlot)].Sample(samp, vs.uv).rgb;
    float3 tonemapped = hdr / (1.0 + hdr); // simple Reinhard
    return float4(tonemapped, 1.0);
}
