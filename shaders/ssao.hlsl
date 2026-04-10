// ssao.hlsl
// ========
// SSAO fallback (HBAO+ style, 16-tap half-res) for Min tier GPUs.
// Hemisphere occlusion from depth buffer. Output: single-channel
// occlusion + scaled constant ambient as RGBA16F.

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

// --- Pixel shader: HBAO+ style SSAO (16 taps) ---
static const int   NUM_TAPS   = 16;
static const float AO_RADIUS  = 0.5;
static const float AO_BIAS    = 0.025;
static const float AMBIENT    = 0.15;

float4 PSMain(VSOut vs) : SV_Target {
    float2 uv   = vs.texCoord;
    int2   coord = int2(uv * float2(pc.screenWidth, pc.screenHeight));

    float depth = g_textures[NonUniformResourceIndex(pc.depthSlot)].Load(int3(coord, 0)).r;

    // Background: full ambient.
    if (depth == 0.0) return float4(AMBIENT, AMBIENT, AMBIENT, 1.0);

    float3 N = normalize(g_textures[NonUniformResourceIndex(pc.normalSlot)].Load(int3(coord, 0)).rgb * 2.0 - 1.0);

    CameraData cam = loadCamera(pc.cameraSlot);
    float2 ndc     = uv * 2.0 - 1.0;
    float4 wp4     = mul(cam.invViewProj, float4(ndc, depth, 1.0));
    float3 worldPos = wp4.xyz / wp4.w;

    // 16-tap hemisphere sampling.
    float occlusion = 0.0;
    for (int i = 0; i < NUM_TAPS; ++i) {
        uint seed = uint(coord.x) * 1973u + uint(coord.y) * 9277u + uint(i) * 26699u;
        float u1 = frac(float(seed) / 4096.0);
        seed = seed * 747796405u + 2891336453u;
        float u2 = frac(float(seed) / 4096.0);
        seed = seed * 747796405u + 2891336453u;
        float u3 = frac(float(seed) / 4096.0);

        // Random point in hemisphere.
        float3 sampleDir = float3(u1 * 2.0 - 1.0, u2 * 2.0 - 1.0, u3);
        sampleDir = normalize(sampleDir);
        if (dot(sampleDir, N) < 0.0) sampleDir = -sampleDir;

        float3 samplePos = worldPos + sampleDir * AO_RADIUS;

        // Project back to screen.
        float4 proj = mul(cam.viewProj, float4(samplePos, 1.0));
        float3 ndcSample = proj.xyz / proj.w;
        float2 sampleUV  = ndcSample.xy * 0.5 + 0.5;

        if (sampleUV.x < 0.0 || sampleUV.x > 1.0 || sampleUV.y < 0.0 || sampleUV.y > 1.0)
            continue;

        int2 sCoord = int2(sampleUV * float2(pc.screenWidth, pc.screenHeight));
        float sceneDepth = g_textures[NonUniformResourceIndex(pc.depthSlot)].Load(int3(sCoord, 0)).r;

        // Reverse-Z: closer = larger depth value.
        float rangeCheck = smoothstep(0.0, 1.0, AO_RADIUS / abs(ndcSample.z - sceneDepth + AO_BIAS));
        occlusion += (sceneDepth >= ndcSample.z + AO_BIAS ? 1.0 : 0.0) * rangeCheck;
    }

    occlusion = 1.0 - (occlusion / float(NUM_TAPS));
    float3 ambient = float3(AMBIENT, AMBIENT, AMBIENT) * occlusion;
    return float4(ambient, 1.0);
}
