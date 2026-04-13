// smaa_blend.hlsl
// ===============
// SMAA Pass 2: blending weight calculation.
//
// For each edge pixel, searches along the edge in both directions (up/down for
// horizontal boundaries, left/right for vertical boundaries) to find the extent
// of the edge segment. A triangle blending weight is computed from the position
// within that segment: weight peaks at 0.5 (50% blend) at the midpoint and
// falls to 0 at the endpoints.
//
// This is a simplified MLAA-style weight calculation (no area-texture lookup).
// It produces good visual results on straight and gently curved edges.
//
// Search radius: SMAA_MAX_STEPS = 8 pixels in each direction.
//
// Output format: R8G8B8A8_UNORM
//   r = blend weight for vertical-axis smoothing   (edge.g boundary)
//   g = blend weight for horizontal-axis smoothing (edge.r boundary)
//   ba = reserved (0)

[[vk::binding(0, 0)]]
Texture2D g_textures[] : register(t0, space0);

[[vk::binding(3, 0)]]
SamplerState g_samplers[] : register(s0, space0);

struct PushBlock {
    uint edgeSlot;     // bindless sampled slot for the R8G8 edge texture
    uint samplerSlot;  // nearest-clamp sampler
    uint width;
    uint height;
};

[[vk::push_constant]] PushBlock pc;

static const int SMAA_MAX_STEPS = 8;

struct VSOutput {
    float4 sv_pos : SV_Position;
    float2 uv     : TEXCOORD0;
};

VSOutput VSMain(uint id : SV_VertexID) {
    float2 uv  = float2((id == 1) ? 2.0f : 0.0f,
                        (id == 2) ? 2.0f : 0.0f);
    float4 pos = float4(uv.x * 2.0f - 1.0f,
                        uv.y * 2.0f - 1.0f,
                        0.0f, 1.0f);
    VSOutput o;
    o.sv_pos = pos;
    o.uv     = uv;
    return o;
}

// Triangle weight: 0 at endpoints (t=0 or t=1), 0.5 at midpoint (t=0.5).
float triangleWeight(int negLen, int posLen) {
    const float total = float(negLen + posLen) + 1.0f;
    const float t     = float(negLen) / total;             // position in [0,1]
    return max(0.0f, 0.5f - abs(t - 0.5f));               // tent in [0, 0.5]
}

float4 PSMain(VSOutput input) : SV_Target0 {
    Texture2D    edge = g_textures[NonUniformResourceIndex(pc.edgeSlot)];
    SamplerState smp  = g_samplers[NonUniformResourceIndex(pc.samplerSlot)];

    const float2 texel = float2(1.0f / float(pc.width),
                                1.0f / float(pc.height));
    const float2 uv    = input.uv;

    const float2 edges = edge.Sample(smp, uv).rg;

    float4 weights = float4(0.0f, 0.0f, 0.0f, 0.0f);

    // ---- edge.g: horizontal boundary (colours differ top-to-bottom) ----
    // The edge line runs HORIZONTALLY. Smooth it by blending vertically.
    // Search left and right along the edge row.
    if (edges.g > 0.5f) {
        int leftLen = 0;
        [unroll(8)]
        for (int dx = 1; dx <= SMAA_MAX_STEPS; ++dx) {
            if (edge.Sample(smp, uv - float2(float(dx) * texel.x, 0.0f)).g < 0.5f)
                break;
            ++leftLen;
        }
        int rightLen = 0;
        [unroll(8)]
        for (int dx2 = 1; dx2 <= SMAA_MAX_STEPS; ++dx2) {
            if (edge.Sample(smp, uv + float2(float(dx2) * texel.x, 0.0f)).g < 0.5f)
                break;
            ++rightLen;
        }
        weights.r = triangleWeight(leftLen, rightLen);
    }

    // ---- edge.r: vertical boundary (colours differ left-to-right) ----
    // The edge line runs VERTICALLY. Smooth it by blending horizontally.
    // Search up and down along the edge column.
    if (edges.r > 0.5f) {
        int upLen = 0;
        [unroll(8)]
        for (int dy = 1; dy <= SMAA_MAX_STEPS; ++dy) {
            if (edge.Sample(smp, uv - float2(0.0f, float(dy) * texel.y)).r < 0.5f)
                break;
            ++upLen;
        }
        int downLen = 0;
        [unroll(8)]
        for (int dy2 = 1; dy2 <= SMAA_MAX_STEPS; ++dy2) {
            if (edge.Sample(smp, uv + float2(0.0f, float(dy2) * texel.y)).r < 0.5f)
                break;
            ++downLen;
        }
        weights.g = triangleWeight(upLen, downLen);
    }

    return weights;
}
