// denoise_temporal.hlsl
// =====================
// Temporal accumulation with neighborhood clamping (compute shader).
// Blend factor ~0.1 per frame = ~10 frame history. Uses motion vectors
// for reprojection. Clamps history sample to current neighborhood AABB
// to prevent ghosting.

[[vk::binding(0, 0)]]
Texture2D g_textures[] : register(t0, space0);

[[vk::binding(1, 0)]]
RWTexture2D<float4> g_storageImages[] : register(u0, space0);

[[vk::binding(2, 0)]]
StructuredBuffer<float4> g_buffers[] : register(t0, space1);

struct PushBlock {
    uint inputSlot;
    uint historySlot;
    uint motionVecSlot;
    uint outputSlot;
    uint screenWidth;
    uint screenHeight;
    uint _pad0;
    uint _pad1;
};

[[vk::push_constant]] PushBlock pc;

static const float BLEND_FACTOR = 0.1; // ~10 frame history

[numthreads(8, 8, 1)]
void CSMain(uint3 dispatchID : SV_DispatchThreadID) {
    if (dispatchID.x >= pc.screenWidth || dispatchID.y >= pc.screenHeight)
        return;

    int2 coord = int2(dispatchID.xy);
    float2 uv  = (float2(coord) + 0.5) / float2(pc.screenWidth, pc.screenHeight);

    float4 current = g_storageImages[NonUniformResourceIndex(pc.inputSlot)][coord];

    // Read motion vector for temporal reprojection.
    float2 motionVec = g_textures[NonUniformResourceIndex(pc.motionVecSlot)].Load(int3(coord, 0)).rg;
    float2 historyUV = uv - motionVec;

    // Compute neighborhood AABB for clamping (3x3 neighborhood).
    float4 nMin = current;
    float4 nMax = current;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int2 nc = coord + int2(dx, dy);
            nc = clamp(nc, int2(0, 0), int2(pc.screenWidth - 1, pc.screenHeight - 1));
            float4 neighborVal = g_storageImages[NonUniformResourceIndex(pc.inputSlot)][nc];
            nMin = min(nMin, neighborVal);
            nMax = max(nMax, neighborVal);
        }
    }

    // Sample history with bounds check.
    float4 history = current; // fallback if out of bounds
    if (historyUV.x >= 0.0 && historyUV.x <= 1.0 &&
        historyUV.y >= 0.0 && historyUV.y <= 1.0) {
        int2 histCoord = int2(historyUV * float2(pc.screenWidth, pc.screenHeight));
        histCoord = clamp(histCoord, int2(0, 0), int2(pc.screenWidth - 1, pc.screenHeight - 1));
        history = g_storageImages[NonUniformResourceIndex(pc.historySlot)][histCoord];
    }

    // Clamp history to neighborhood AABB to prevent ghosting.
    history = clamp(history, nMin, nMax);

    // Temporal blend.
    float4 result = lerp(history, current, BLEND_FACTOR);
    g_storageImages[NonUniformResourceIndex(pc.outputSlot)][coord] = result;
}
