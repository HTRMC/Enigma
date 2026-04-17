// smaa_edge.hlsl — SMAA 1x luma edge detection with local-contrast
// adaptation.  Ported from Jorge Jimenez's SMAA.hlsl reference
// (MIT-licensed — see https://github.com/iryoku/smaa).
//
// Output: R8G8_UNORM
//   r = horizontal contrast (left-to-center)  → vertical edge line
//   g = vertical contrast   (top-to-center)   → horizontal edge line
//
// Local-contrast adaptation: a candidate edge is kept only if its contrast
// is at least 1/SMAA_LOCAL_CONTRAST_ADAPTATION_FACTOR of the maximum
// 4-neighbour contrast. This prunes spurious edges inside textured regions
// without missing silhouettes.

[[vk::binding(0, 0)]]
Texture2D g_textures[] : register(t0, space0);

[[vk::binding(3, 0)]]
SamplerState g_samplers[] : register(s0, space0);

struct PushBlock {
    uint colorSlot;     // bindless sampled slot for the tonemapped LDR buffer
    uint samplerSlot;   // linear-clamp sampler
    uint width;
    uint height;
};

[[vk::push_constant]] PushBlock pc;

// SMAA 1x Ultra preset threshold — applied to *perceptual* (gamma-≈2.0) luma.
// The LDR buffer is an sRGB image so hardware-decoded samples arrive linear;
// the sqrt() in luma() lifts them back into the paper's working space.
static const float SMAA_THRESHOLD                         = 0.05f;
static const float SMAA_LOCAL_CONTRAST_ADAPTATION_FACTOR  = 2.0f;

float luma(float3 c) {
    const float3 perceptual = sqrt(max(c, 0.0f));
    return dot(perceptual, float3(0.299f, 0.587f, 0.114f));
}

struct VSOutput {
    float4 sv_pos  : SV_Position;
    float2 uv      : TEXCOORD0;
    float4 offset0 : TEXCOORD1;   // (-1,0,0,-1) * texel — left  / top
    float4 offset1 : TEXCOORD2;   // ( 1,0,0, 1) * texel — right / bottom
    float4 offset2 : TEXCOORD3;   // (-2,0,0,-2) * texel — leftleft / toptop
};

VSOutput VSMain(uint id : SV_VertexID) {
    float2 uv  = float2((id == 1) ? 2.0f : 0.0f,
                        (id == 2) ? 2.0f : 0.0f);
    float4 pos = float4(uv.x * 2.0f - 1.0f,
                        uv.y * 2.0f - 1.0f,
                        0.0f, 1.0f);

    const float2 texel = float2(1.0f / float(pc.width),
                                1.0f / float(pc.height));
    const float4 tex4  = float4(texel, texel);

    VSOutput o;
    o.sv_pos  = pos;
    o.uv      = uv;
    o.offset0 = uv.xyxy + tex4 * float4(-1.0f, 0.0f, 0.0f, -1.0f);
    o.offset1 = uv.xyxy + tex4 * float4( 1.0f, 0.0f, 0.0f,  1.0f);
    o.offset2 = uv.xyxy + tex4 * float4(-2.0f, 0.0f, 0.0f, -2.0f);
    return o;
}

float2 PSMain(VSOutput input) : SV_Target0 {
    Texture2D    col = g_textures[NonUniformResourceIndex(pc.colorSlot)];
    SamplerState smp = g_samplers[NonUniformResourceIndex(pc.samplerSlot)];

    const float2 threshold = float2(SMAA_THRESHOLD, SMAA_THRESHOLD);

    const float L     = luma(col.SampleLevel(smp, input.uv,         0.0f).rgb);
    const float Lleft = luma(col.SampleLevel(smp, input.offset0.xy, 0.0f).rgb);
    const float Ltop  = luma(col.SampleLevel(smp, input.offset0.zw, 0.0f).rgb);

    float4 delta;
    delta.xy = abs(L - float2(Lleft, Ltop));
    float2 edges = step(threshold, delta.xy);

    // Early-out the majority of pixels that have no edge in either axis —
    // saves the 4 following sample fetches.
    if (dot(edges, float2(1.0f, 1.0f)) == 0.0f) {
        discard;
    }

    const float Lright  = luma(col.SampleLevel(smp, input.offset1.xy, 0.0f).rgb);
    const float Lbottom = luma(col.SampleLevel(smp, input.offset1.zw, 0.0f).rgb);
    delta.zw = abs(L - float2(Lright, Lbottom));

    float2 maxDelta = max(delta.xy, delta.zw);

    const float Lleftleft = luma(col.SampleLevel(smp, input.offset2.xy, 0.0f).rgb);
    const float Ltoptop   = luma(col.SampleLevel(smp, input.offset2.zw, 0.0f).rgb);
    delta.zw = abs(float2(Lleft, Ltop) - float2(Lleftleft, Ltoptop));

    maxDelta = max(maxDelta.xy, delta.zw);
    const float finalDelta = max(maxDelta.x, maxDelta.y);

    edges.xy *= step(finalDelta, SMAA_LOCAL_CONTRAST_ADAPTATION_FACTOR * delta.xy);

    return edges;
}
