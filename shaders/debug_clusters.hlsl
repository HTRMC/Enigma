// debug_clusters.hlsl — Clusters debug view: colors each meshlet uniquely

#include "common.hlsl"

[[vk::binding(0, 0)]] Texture2D g_textures[] : register(t0, space0);
[[vk::binding(3, 0)]] SamplerState g_samplers[] : register(s0, space0);

struct PushBlock {
    uint albedoSlot;    // unused
    uint normalSlot;    // unused
    uint metalRoughSlot; // unused
    uint depthSlot;     // unused
    uint cameraSlot;    // unused
    uint samplerSlot;
    uint visBufferSlot;
    uint _pad1;
    float4 _lightDirIntensity;
    float4 _lightColor;
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

float3 MeshletColor(uint meshletId) {
    uint h = meshletId * 2654435761u;
    return float3(
        (h & 0xFFu) / 255.0,
        ((h >> 8u) & 0xFFu) / 255.0,
        ((h >> 16u) & 0xFFu) / 255.0);
}

float4 PSMain(VSOut vs) : SV_Target {
    int2 pixelCoord = int2(vs.pos.xy);
    Texture2D visTex = g_textures[NonUniformResourceIndex(pc.visBufferSlot)];
    uint vis = asuint(visTex.Load(int3(pixelCoord, 0)).r);
    if (vis == 0xFFFFFFFFu)
        return float4(0.02, 0.02, 0.05, 1.0);
    uint globalMeshletId = vis >> 7u;
    return float4(MeshletColor(globalMeshletId), 1.0);
}
