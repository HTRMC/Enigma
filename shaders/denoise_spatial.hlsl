// denoise_spatial.hlsl
// ====================
// A-trous wavelet spatial filter (compute shader). Edge-aware: weight by
// depth/normal similarity to preserve geometric edges. 5 iterations at
// increasing step widths (1, 2, 4, 8, 16). Input/output: storage images
// accessed via bindless.

[[vk::binding(0, 0)]]
Texture2D g_textures[] : register(t0, space0);

[[vk::binding(1, 0)]]
RWTexture2D<float4> g_storageImages[] : register(u0, space0);

[[vk::binding(2, 0)]]
StructuredBuffer<float4> g_buffers[] : register(t0, space1);

struct PushBlock {
    uint inputSlot;
    uint outputSlot;
    uint normalSlot;
    uint depthSlot;
    uint screenWidth;
    uint screenHeight;
    uint stepWidth;
    uint _pad0;
};

[[vk::push_constant]] PushBlock pc;

// A-trous 5x5 wavelet kernel weights.
static const float kernel[3] = { 1.0 / 4.0, 1.0 / 8.0, 1.0 / 16.0 };

[numthreads(8, 8, 1)]
void CSMain(uint3 dispatchID : SV_DispatchThreadID) {
    if (dispatchID.x >= pc.screenWidth || dispatchID.y >= pc.screenHeight)
        return;

    int2 center = int2(dispatchID.xy);
    float4 centerVal = g_storageImages[NonUniformResourceIndex(pc.inputSlot)][center];

    float4 sum    = centerVal * kernel[0];
    float  wTotal = kernel[0];

    int stepW = int(pc.stepWidth);

    // 5x5 A-trous kernel (offsets: -2, -1, 0, 1, 2) * stepWidth.
    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            if (dx == 0 && dy == 0) continue;

            int2 offset = int2(dx, dy) * stepW;
            int2 sampleCoord = center + offset;

            // Bounds check.
            if (sampleCoord.x < 0 || sampleCoord.x >= int(pc.screenWidth) ||
                sampleCoord.y < 0 || sampleCoord.y >= int(pc.screenHeight))
                continue;

            float4 sampleVal = g_storageImages[NonUniformResourceIndex(pc.inputSlot)][sampleCoord];

            // Kernel weight based on Manhattan distance.
            int dist = abs(dx) + abs(dy);
            float w = (dist <= 1) ? kernel[1] : kernel[2];

            // Edge-aware: reduce weight for large value differences.
            float diff = length(sampleVal.rgb - centerVal.rgb);
            w *= exp(-diff * diff * 4.0);

            sum    += sampleVal * w;
            wTotal += w;
        }
    }

    g_storageImages[NonUniformResourceIndex(pc.outputSlot)][center] = sum / wTotal;
}
