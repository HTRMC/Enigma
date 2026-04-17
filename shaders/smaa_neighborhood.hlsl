// smaa_neighborhood.hlsl — SMAA 1x neighbourhood blending (3rd pass).
// Ported from Jorge Jimenez's SMAA.hlsl reference (MIT-licensed — see
// https://github.com/iryoku/smaa).
//
// Samples the 4-direction blend weights produced by smaa_blend.hlsl from
// this pixel *and its +X/+Y neighbours*, picks the axis with greater total
// weight, and blends the current pixel with the correct cross-edge
// neighbour using the weight as the lerp factor. The 4-direction lookup
// (rather than only-self) is what ensures both sides of every edge receive
// a blend contribution and eliminates the "halo on one side only" symptom
// of naive MLAA.

[[vk::binding(0, 0)]]
Texture2D g_textures[] : register(t0, space0);

[[vk::binding(3, 0)]]
SamplerState g_samplers[] : register(s0, space0);

struct PushBlock {
    uint colorSlot;     // tonemapped LDR colour (sRGB view → linear sample)
    uint weightSlot;    // RGBA8 blend weight texture from pass 2
    uint samplerSlot;   // linear-clamp sampler
    uint width;
    uint height;
    uint _pad;
};

[[vk::push_constant]] PushBlock pc;

struct VSOutput {
    float4 sv_pos : SV_Position;
    float2 uv     : TEXCOORD0;
    float4 offset : TEXCOORD1;   // xy: uv+(texel.x,0)   zw: uv+(0,texel.y)
};

VSOutput VSMain(uint id : SV_VertexID) {
    float2 uv  = float2((id == 1) ? 2.0f : 0.0f,
                        (id == 2) ? 2.0f : 0.0f);
    float4 pos = float4(uv.x * 2.0f - 1.0f,
                        uv.y * 2.0f - 1.0f,
                        0.0f, 1.0f);

    const float2 texel = float2(1.0f / float(pc.width),
                                1.0f / float(pc.height));

    VSOutput o;
    o.sv_pos = pos;
    o.uv     = uv;
    o.offset = uv.xyxy + float4(texel, texel) * float4(1.0f, 0.0f, 0.0f, 1.0f);
    return o;
}

float4 PSMain(VSOutput input) : SV_Target0 {
    Texture2D    colTex = g_textures[NonUniformResourceIndex(pc.colorSlot)];
    Texture2D    wgtTex = g_textures[NonUniformResourceIndex(pc.weightSlot)];
    SamplerState smp    = g_samplers[NonUniformResourceIndex(pc.samplerSlot)];

    const float4 rtm = float4(1.0f / float(pc.width), 1.0f / float(pc.height),
                              float(pc.width),        float(pc.height));

    // 4-direction blend weights:
    //   a.x = right   (neighbour at +X contributes .a = bottom-of-that-edge)
    //   a.y = top     (neighbour at +Y contributes .g = top-of-that-edge)
    //   a.z = left    (this pixel's .x = right-of-the-edge-on-our-left)
    //   a.w = bottom  (this pixel's .z = bottom-of-the-edge-on-our-top)
    float4 a;
    a.x  = wgtTex.SampleLevel(smp, input.offset.xy, 0.0f).a;
    a.y  = wgtTex.SampleLevel(smp, input.offset.zw, 0.0f).g;
    a.wz = wgtTex.SampleLevel(smp, input.uv,        0.0f).xz;

    [branch]
    if (dot(a, float4(1.0f, 1.0f, 1.0f, 1.0f)) < 1e-5f) {
        return colTex.SampleLevel(smp, input.uv, 0.0f);
    }

    // Pick the axis (horizontal or vertical) with the larger total weight.
    const bool h = max(a.x, a.z) > max(a.y, a.w);

    float4 blendingOffset = float4(0.0f, a.y, 0.0f, a.w);
    float2 blendingWeight = a.yw;
    if (h) {
        blendingOffset = float4(a.x, 0.0f, a.z, 0.0f);
        blendingWeight = a.xz;
    }
    blendingWeight /= dot(blendingWeight, float2(1.0f, 1.0f));

    // Two sample positions — the edge midpoints on the chosen axis.  Exploits
    // bilinear filtering to cross-interpolate this pixel with the neighbour
    // at each side in a single fetch.
    const float4 blendingCoord = mad(blendingOffset,
                                      float4(rtm.xy, -rtm.xy),
                                      input.uv.xyxy);

    float4 color;
    color  = blendingWeight.x * colTex.SampleLevel(smp, blendingCoord.xy, 0.0f);
    color += blendingWeight.y * colTex.SampleLevel(smp, blendingCoord.zw, 0.0f);
    return color;
}
