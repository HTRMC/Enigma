// csm.hlsl
// =======
// Cascaded Shadow Map (CSM) fallback for Min tier GPUs.
// 4 cascades, PCF filtering.
//
// CSM raster generation deferred to Phase 3; this stub returns no occlusion.
// The pass always outputs 1.0 (fully lit) as a placeholder until the actual
// shadow map atlas and cascade matrices are wired in Phase 3.

#include "common.hlsl"

[[vk::binding(0, 0)]]
Texture2D g_textures[] : register(t0, space0);

[[vk::binding(1, 0)]]
RWTexture2D<float4> g_storageImages[] : register(u0, space0);

[[vk::binding(2, 0)]]
StructuredBuffer<float4> g_buffers[] : register(t0, space1);

[[vk::binding(3, 0)]]
SamplerState g_samplers[] : register(s0, space0);

struct PushBlock {
    uint depthSlot;
    uint normalSlot;
    uint cameraSlot;
    uint outputSlot;
    uint screenWidth;
    uint screenHeight;
    uint _pad0;
    uint _pad1;
};

[[vk::push_constant]] PushBlock pc;

// --- Vertex shader: fullscreen triangle ---
struct VSOut {
    float4 pos      : SV_Position;
    float2 texCoord : TEXCOORD0;
};

VSOut VSMain(uint vid : SV_VertexID) {
    float2 uv = float2((vid << 1) & 2, vid & 2);
    VSOut o;
    o.pos      = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    o.texCoord = uv;
    return o;
}

// --- Pixel shader: CSM stub (returns fully lit) ---
float4 PSMain(VSOut vs) : SV_Target {
    // CSM raster generation deferred to Phase 3; this stub returns no occlusion.
    return float4(1.0, 1.0, 1.0, 1.0);
}
