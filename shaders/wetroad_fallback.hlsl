// wetroad_fallback.hlsl
// =====================
// Screen-space planar reflection fallback for wet road on Min tier.
// Simple screen-space lookup: reflect view ray around Y-axis, reproject
// into UV space. Blend with scene colour using wetnessFactor.

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
    uint  normalSlot;
    uint  depthSlot;
    uint  cameraSlot;
    uint  outputSlot;
    float wetnessFactor;
    uint  screenWidth;
    uint  screenHeight;
    uint  _pad0;
};

[[vk::push_constant]] PushBlock pc;

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

// --- Pixel shader: screen-space planar reflection ---
float4 PSMain(VSOut vs) : SV_Target {
    float2 uv   = vs.texCoord;
    int2   coord = int2(uv * float2(pc.screenWidth, pc.screenHeight));

    float4 normalSample = g_textures[NonUniformResourceIndex(pc.normalSlot)].Load(int3(coord, 0));
    float  depth        = g_textures[NonUniformResourceIndex(pc.depthSlot)].Load(int3(coord, 0)).r;

    float3 N = normalize(normalSample.rgb * 2.0 - 1.0);

    // Skip non-road and background pixels.
    if (depth == 0.0 || N.y < 0.8 || pc.wetnessFactor < 0.001) return float4(0, 0, 0, 0);

    // Reconstruct world position.
    CameraData cam = loadCamera(pc.cameraSlot);
    float2 ndc     = uv * 2.0 - 1.0;
    float4 wp4     = mul(cam.invViewProj, float4(ndc, depth, 1.0));
    float3 worldPos = wp4.xyz / wp4.w;

    // Reflect view ray around Y-axis (planar assumption).
    float3 V          = normalize(cam.worldPos.xyz - worldPos);
    float3 reflectedV = float3(V.x, -V.y, V.z);

    // Reproject reflected point into screen space.
    float3 reflectedPos = worldPos + reflectedV * 2.0;
    float4 proj = mul(cam.viewProj, float4(reflectedPos, 1.0));
    float2 reflUV = (proj.xy / proj.w) * 0.5 + 0.5;

    // Out-of-bounds check.
    if (reflUV.x < 0.0 || reflUV.x > 1.0 || reflUV.y < 0.0 || reflUV.y > 1.0)
        return float4(0, 0, 0, 0);

    // Simple sky-tinted colour as placeholder reflection.
    float t = saturate(reflectedV.y * 0.5 + 0.5);
    float3 sky = lerp(float3(0.1, 0.1, 0.2), float3(0.4, 0.6, 1.0), t);
    return float4(sky * pc.wetnessFactor * 0.5, pc.wetnessFactor);
}
