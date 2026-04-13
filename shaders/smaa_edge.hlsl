// smaa_edge.hlsl
// ==============
// SMAA Pass 1: luma-based edge detection.
//
// For each pixel P, computes the luminance contrast with its left and top
// neighbours. Marks a pixel as a horizontal-boundary edge when the left
// contrast exceeds the threshold (output.r), and as a vertical-boundary edge
// when the top contrast exceeds it (output.g).
//
// Output format: R8G8_UNORM
//   r = horizontal boundary (strong left-right contrast → vertical edge line)
//   g = vertical   boundary (strong top-bottom contrast → horizontal edge line)
//
// Threshold is hard-coded to 0.1 (SMAA "Low/Medium" quality setting).
// Raising it gives fewer AA samples and may expose aliasing on fine geometry.

[[vk::binding(0, 0)]]
Texture2D g_textures[] : register(t0, space0);

[[vk::binding(3, 0)]]
SamplerState g_samplers[] : register(s0, space0);

struct PushBlock {
    uint colorSlot;    // bindless sampled slot for the tonemapped LDR buffer
    uint samplerSlot;  // nearest-clamp sampler
    uint width;        // framebuffer width  (pixels)
    uint height;       // framebuffer height (pixels)
};

[[vk::push_constant]] PushBlock pc;

static const float SMAA_THRESHOLD = 0.1f;

// ITU-R BT.601 luma — matches human perception better than equal weights.
float luma(float3 c) {
    return dot(c, float3(0.299f, 0.587f, 0.114f));
}

struct VSOutput {
    float4 sv_pos : SV_Position;
    float2 uv     : TEXCOORD0;
};

VSOutput VSMain(uint id : SV_VertexID) {
    // Fullscreen triangle: 3 vertices cover [-1,1]^2 without clipping.
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

float2 PSMain(VSOutput input) : SV_Target0 {
    Texture2D   col = g_textures[NonUniformResourceIndex(pc.colorSlot)];
    SamplerState smp = g_samplers[NonUniformResourceIndex(pc.samplerSlot)];

    const float2 texel = float2(1.0f / float(pc.width),
                                1.0f / float(pc.height));
    const float2 uv    = input.uv;

    const float lP    = luma(col.Sample(smp, uv).rgb);
    const float lLeft = luma(col.Sample(smp, uv - float2(texel.x, 0.0f)).rgb);
    const float lTop  = luma(col.Sample(smp, uv - float2(0.0f, texel.y)).rgb);

    float edgeH = step(SMAA_THRESHOLD, abs(lP - lLeft));
    float edgeV = step(SMAA_THRESHOLD, abs(lP - lTop));

    return float2(edgeH, edgeV);
}
