// hiz_build.comp.hlsl
// ====================
// Single-pass per-mip hierarchical Z builder. Called once per mip level to
// downsample the previous level by taking the minimum depth of each 2x2 block.
//
// Reverse-Z convention: far = 0, near = 1. The minimum value is the most
// conservative (furthest) depth for occlusion culling.
//
// Dispatch: ceil(dstWidth / 8) x ceil(dstHeight / 8) workgroups.

// --- Bindless resource arrays ---
[[vk::binding(1, 0)]]
RWTexture2D<float> g_hizImages[] : register(u0, space0);

// --- Push constants ---
struct PushBlock {
    uint srcSlot;    // source mip (RWTexture2D<float>)
    uint dstSlot;    // destination mip (RWTexture2D<float>)
    uint srcWidth;   // source mip resolution width
    uint srcHeight;  // source mip resolution height
};

[[vk::push_constant]] PushBlock pc;

// --- Main ---
[numthreads(8, 8, 1)]
void CSMain(uint3 dispatchId : SV_DispatchThreadID) {
    // Destination pixel coordinate.
    uint2 dstCoord = dispatchId.xy;

    // Corresponding 2x2 block in the source mip.
    uint2 srcBase = dstCoord * 2;

    RWTexture2D<float> srcMip = g_hizImages[NonUniformResourceIndex(pc.srcSlot)];
    RWTexture2D<float> dstMip = g_hizImages[NonUniformResourceIndex(pc.dstSlot)];

    // Clamp to valid source dimensions to handle odd-sized mips.
    uint2 c00 = min(srcBase + uint2(0, 0), uint2(pc.srcWidth - 1, pc.srcHeight - 1));
    uint2 c10 = min(srcBase + uint2(1, 0), uint2(pc.srcWidth - 1, pc.srcHeight - 1));
    uint2 c01 = min(srcBase + uint2(0, 1), uint2(pc.srcWidth - 1, pc.srcHeight - 1));
    uint2 c11 = min(srcBase + uint2(1, 1), uint2(pc.srcWidth - 1, pc.srcHeight - 1));

    float d00 = srcMip[c00];
    float d10 = srcMip[c10];
    float d01 = srcMip[c01];
    float d11 = srcMip[c11];

    // Reverse-Z: minimum = furthest = most conservative for occlusion.
    float result = min(min(d00, d10), min(d01, d11));

    dstMip[dstCoord] = result;
}
