// smaa_neighborhood.hlsl
// ======================
// SMAA Pass 3: neighbourhood blending.
//
// Reads the blend weights computed in Pass 2 and produces the final
// anti-aliased colour by lerping the current pixel with the average of its
// opposite neighbours, perpendicular to the detected edge direction:
//
//   - weights.r (horizontal boundary): blend current pixel with the average of
//     its top and bottom neighbours (±0.5 texel vertically).
//   - weights.g (vertical boundary):   blend current pixel with the average of
//     its left and right neighbours (±0.5 texel horizontally).
//
// The 0.5-texel offset means we sample exactly between the current pixel and
// its neighbour, relying on the GPU's bilinear filter to interpolate. This
// gives sub-pixel accuracy without extra samples.
//
// Uses a linear sampler for colour (bilinear interpolation) and a nearest
// sampler for the blend weight texture (no filtering wanted on weights).

[[vk::binding(0, 0)]]
Texture2D g_textures[] : register(t0, space0);

[[vk::binding(3, 0)]]
SamplerState g_samplers[] : register(s0, space0);

struct PushBlock {
    uint colorSlot;    // tonemapped LDR colour (linear sampler)
    uint weightSlot;   // SMAA blend weight texture (nearest sampler)
    uint samplerSlot;  // nearest-clamp sampler (used for both — colour via bilinear)
    uint width;
    uint height;
    uint _pad;
};

[[vk::push_constant]] PushBlock pc;

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

float4 PSMain(VSOutput input) : SV_Target0 {
    Texture2D    colTex = g_textures[NonUniformResourceIndex(pc.colorSlot)];
    Texture2D    wgtTex = g_textures[NonUniformResourceIndex(pc.weightSlot)];
    SamplerState smp    = g_samplers[NonUniformResourceIndex(pc.samplerSlot)];

    const float2 texel = float2(1.0f / float(pc.width),
                                1.0f / float(pc.height));
    const float2 uv    = input.uv;

    float4 weights = wgtTex.Sample(smp, uv);
    float4 color   = colTex.Sample(smp, uv);

    // weights.r: horizontal edge — blend vertically.
    // Sample ±0.5 texels above and below; use GPU bilinear to interpolate.
    if (weights.r > 0.0f) {
        float4 above = colTex.Sample(smp, uv - float2(0.0f, 0.5f * texel.y));
        float4 below = colTex.Sample(smp, uv + float2(0.0f, 0.5f * texel.y));
        color = lerp(color, (above + below) * 0.5f, weights.r * 2.0f);
    }

    // weights.g: vertical edge — blend horizontally.
    if (weights.g > 0.0f) {
        float4 left  = colTex.Sample(smp, uv - float2(0.5f * texel.x, 0.0f));
        float4 right = colTex.Sample(smp, uv + float2(0.5f * texel.x, 0.0f));
        color = lerp(color, (left + right) * 0.5f, weights.g * 2.0f);
    }

    return color;
}
