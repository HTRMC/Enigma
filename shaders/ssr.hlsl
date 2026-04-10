// ssr.hlsl
// =======
// Screen-space reflections (raster fallback for Min tier / GTX 1650).
// Simple linear ray march against the depth buffer.
// Input: G-buffer normal + depth + camera SSBO. Output: reflection colour.
// Quality is intentionally lower than RT; used when hardware RT is unavailable.

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
    uint normalSlot;
    uint depthSlot;
    uint cameraSlot;
    uint samplerSlot;
    uint outputSlot;
    uint screenWidth;
    uint screenHeight;
    uint _pad0;
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

// --- Pixel shader: linear SSR ray march ---
float4 PSMain(VSOut vs) : SV_Target {
    float2 uv = vs.texCoord;
    SamplerState samp = g_samplers[NonUniformResourceIndex(pc.samplerSlot)];

    float  depth       = g_textures[NonUniformResourceIndex(pc.depthSlot)].Sample(samp, uv).r;
    float4 normalSamp  = g_textures[NonUniformResourceIndex(pc.normalSlot)].Sample(samp, uv);

    // Background: no reflection.
    if (depth == 0.0) return float4(0, 0, 0, 0);

    float3 N = normalize(normalSamp.rgb * 2.0 - 1.0);

    // Reconstruct world position.
    float2 ndc      = uv * 2.0 - 1.0;
    CameraData cam  = loadCamera(pc.cameraSlot);
    float4 worldPos4 = mul(cam.invViewProj, float4(ndc, depth, 1.0));
    float3 worldPos  = worldPos4.xyz / worldPos4.w;

    float3 V          = normalize(cam.worldPos.xyz - worldPos);
    float3 reflectDir = reflect(-V, N);

    // Project reflection direction to screen space and march.
    static const int   kMaxSteps   = 32;
    static const float kStepSize   = 0.05;
    static const float kThickness  = 0.1;

    float3 rayPos = worldPos + N * 0.01;
    for (int i = 0; i < kMaxSteps; ++i) {
        rayPos += reflectDir * kStepSize;

        float4 projPos = mul(cam.viewProj, float4(rayPos, 1.0));
        float3 ndcRay  = projPos.xyz / projPos.w;
        float2 ssUV    = ndcRay.xy * 0.5 + 0.5;

        // Out of screen bounds.
        if (ssUV.x < 0.0 || ssUV.x > 1.0 || ssUV.y < 0.0 || ssUV.y > 1.0) break;

        float sceneDepth = g_textures[NonUniformResourceIndex(pc.depthSlot)].SampleLevel(samp, ssUV, 0).r;
        float diff = ndcRay.z - sceneDepth;

        if (diff > 0.0 && diff < kThickness) {
            // Hit: return a sky-tinted color as placeholder.
            float t = saturate(reflectDir.y * 0.5 + 0.5);
            float3 sky = lerp(float3(0.1, 0.1, 0.2), float3(0.4, 0.6, 1.0), t);
            float fade = 1.0 - float(i) / float(kMaxSteps);
            return float4(sky * fade * 0.5, fade);
        }
    }

    return float4(0, 0, 0, 0);
}
