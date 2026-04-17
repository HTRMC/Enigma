// smaa_blend.hlsl — SMAA 1x blending-weight calculation (2nd pass).
// Ported from Jorge Jimenez's SMAA.hlsl reference (MIT-licensed — see
// https://github.com/iryoku/smaa).
//
// Reads the R8G8 edge mask produced by smaa_edge.hlsl, looks up the
// precomputed AreaTex and SearchTex lookup textures to classify L / U / Z /
// diagonal patterns, and writes RGBA8 directional blend weights:
//   r = left   (vertical edge line)
//   g = top    (horizontal edge line)
//   b = right  (vertical edge line)
//   a = bottom (horizontal edge line)
// Diagonal and corner detection are enabled at Ultra quality.

[[vk::binding(0, 0)]]
Texture2D g_textures[] : register(t0, space0);

[[vk::binding(3, 0)]]
SamplerState g_samplers[] : register(s0, space0);

struct PushBlock {
    uint edgeSlot;         // R8G8   edge texture from pass 1
    uint areaTexSlot;      // RG8    AreaTex   (160×560)
    uint searchTexSlot;    // R8     SearchTex (64×16)
    uint samplerSlot;      // linear-clamp sampler
    uint width;
    uint height;
};

[[vk::push_constant]] PushBlock pc;

// ---- Preset (Ultra) ----
static const int   SMAA_MAX_SEARCH_STEPS      = 32;
static const int   SMAA_MAX_SEARCH_STEPS_DIAG = 16;
static const int   SMAA_CORNER_ROUNDING       = 25;

// ---- AreaTex / SearchTex constants (must match Jimenez AreaTex.h / SearchTex.h) ----
static const float  SMAA_AREATEX_MAX_DISTANCE      = 16.0f;
static const float  SMAA_AREATEX_MAX_DISTANCE_DIAG = 20.0f;
static const float2 SMAA_AREATEX_PIXEL_SIZE        = 1.0f / float2(160.0f, 560.0f);
static const float  SMAA_AREATEX_SUBTEX_SIZE       = 1.0f / 7.0f;
static const float2 SMAA_SEARCHTEX_SIZE            = float2(66.0f, 33.0f);
static const float2 SMAA_SEARCHTEX_PACKED_SIZE     = float2(64.0f, 16.0f);
static const float  SMAA_CORNER_ROUNDING_NORM      = float(SMAA_CORNER_ROUNDING) / 100.0f;

// ---- Bindless texture/sampler helpers ----
float4 rtmetrics() {
    const float w = float(pc.width);
    const float h = float(pc.height);
    return float4(1.0f / w, 1.0f / h, w, h);
}

float4 edgesSample(float2 uv) {
    return g_textures[NonUniformResourceIndex(pc.edgeSlot)].SampleLevel(
        g_samplers[NonUniformResourceIndex(pc.samplerSlot)], uv, 0.0f);
}

float4 edgesSampleOffset(float2 uv, int2 offs) {
    return g_textures[NonUniformResourceIndex(pc.edgeSlot)].SampleLevel(
        g_samplers[NonUniformResourceIndex(pc.samplerSlot)], uv, 0.0f, offs);
}

float4 areaSample(float2 uv) {
    return g_textures[NonUniformResourceIndex(pc.areaTexSlot)].SampleLevel(
        g_samplers[NonUniformResourceIndex(pc.samplerSlot)], uv, 0.0f);
}

float4 searchSample(float2 uv) {
    return g_textures[NonUniformResourceIndex(pc.searchTexSlot)].SampleLevel(
        g_samplers[NonUniformResourceIndex(pc.samplerSlot)], uv, 0.0f);
}

void SMAAMovc2(bool2 cond, inout float2 variable, float2 value) {
    if (cond.x) variable.x = value.x;
    if (cond.y) variable.y = value.y;
}

// ---- Diagonal search helpers ----
float2 SMAADecodeDiagBilinearAccess(float2 e) {
    e.r = e.r * abs(5.0f * e.r - 5.0f * 0.75f);
    return round(e);
}

float4 SMAADecodeDiagBilinearAccess4(float4 e) {
    e.rb = e.rb * abs(5.0f * e.rb - 5.0f * 0.75f);
    return round(e);
}

float2 SMAASearchDiag1(float2 texcoord, float2 dir, out float2 e) {
    const float4 rtm   = rtmetrics();
    float4 coord       = float4(texcoord, -1.0f, 1.0f);
    const float3 t     = float3(rtm.xy, 1.0f);
    [loop]
    while (coord.z < float(SMAA_MAX_SEARCH_STEPS_DIAG - 1) && coord.w > 0.9f) {
        coord.xyz = mad(t, float3(dir, 1.0f), coord.xyz);
        e         = edgesSample(coord.xy).rg;
        coord.w   = dot(e, float2(0.5f, 0.5f));
    }
    return coord.zw;
}

float2 SMAASearchDiag2(float2 texcoord, float2 dir, out float2 e) {
    const float4 rtm = rtmetrics();
    float4 coord     = float4(texcoord, -1.0f, 1.0f);
    coord.x         += 0.25f * rtm.x;
    const float3 t   = float3(rtm.xy, 1.0f);
    [loop]
    while (coord.z < float(SMAA_MAX_SEARCH_STEPS_DIAG - 1) && coord.w > 0.9f) {
        coord.xyz = mad(t, float3(dir, 1.0f), coord.xyz);
        e         = edgesSample(coord.xy).rg;
        e         = SMAADecodeDiagBilinearAccess(e);
        coord.w   = dot(e, float2(0.5f, 0.5f));
    }
    return coord.zw;
}

float2 SMAAAreaDiag(float2 dist, float2 e, float offset) {
    float2 texcoord = mad(float2(SMAA_AREATEX_MAX_DISTANCE_DIAG,
                                  SMAA_AREATEX_MAX_DISTANCE_DIAG),
                          e, dist);
    texcoord = mad(SMAA_AREATEX_PIXEL_SIZE, texcoord, 0.5f * SMAA_AREATEX_PIXEL_SIZE);
    texcoord.x += 0.5f;
    texcoord.y += SMAA_AREATEX_SUBTEX_SIZE * offset;
    return areaSample(texcoord).rg;
}

float2 SMAACalculateDiagWeights(float2 texcoord, float2 e, float4 subsampleIndices) {
    const float4 rtm = rtmetrics();
    float2 weights   = float2(0.0f, 0.0f);

    float4 d;
    float2 end;
    if (e.r > 0.0f) {
        d.xz = SMAASearchDiag1(texcoord, float2(-1.0f,  1.0f), end);
        d.x += float(end.y > 0.9f);
    } else {
        d.xz = float2(0.0f, 0.0f);
    }
    d.yw = SMAASearchDiag1(texcoord, float2(1.0f, -1.0f), end);

    [branch]
    if (d.x + d.y > 2.0f) {
        const float4 coords = mad(float4(-d.x + 0.25f, d.x, d.y, -d.y - 0.25f),
                                  rtm.xyxy, texcoord.xyxy);
        float4 c;
        c.xy = edgesSampleOffset(coords.xy, int2(-1,  0)).rg;
        c.zw = edgesSampleOffset(coords.zw, int2( 1,  0)).rg;
        c.yxwz = SMAADecodeDiagBilinearAccess4(c.xyzw);

        float2 cc = mad(float2(2.0f, 2.0f), c.xz, c.yw);
        SMAAMovc2(bool2(step(0.9f, d.zw)), cc, float2(0.0f, 0.0f));
        weights += SMAAAreaDiag(d.xy, cc, subsampleIndices.z);
    }

    d.xz = SMAASearchDiag2(texcoord, float2(-1.0f, -1.0f), end);
    if (edgesSampleOffset(texcoord, int2(1, 0)).r > 0.0f) {
        d.yw = SMAASearchDiag2(texcoord, float2(1.0f, 1.0f), end);
        d.y += float(end.y > 0.9f);
    } else {
        d.yw = float2(0.0f, 0.0f);
    }

    [branch]
    if (d.x + d.y > 2.0f) {
        const float4 coords = mad(float4(-d.x, -d.x, d.y, d.y), rtm.xyxy, texcoord.xyxy);
        float4 c;
        c.x  = edgesSampleOffset(coords.xy, int2(-1,  0)).g;
        c.y  = edgesSampleOffset(coords.xy, int2( 0, -1)).r;
        c.zw = edgesSampleOffset(coords.zw, int2( 1,  0)).gr;
        float2 cc = mad(float2(2.0f, 2.0f), c.xz, c.yw);
        SMAAMovc2(bool2(step(0.9f, d.zw)), cc, float2(0.0f, 0.0f));
        weights += SMAAAreaDiag(d.xy, cc, subsampleIndices.w).gr;
    }

    return weights;
}

// ---- Orthogonal search helpers ----
float SMAASearchLength(float2 e, float offset) {
    float2 scale = SMAA_SEARCHTEX_SIZE * float2(0.5f, -1.0f);
    float2 bias  = SMAA_SEARCHTEX_SIZE * float2(offset, 1.0f);
    scale += float2(-1.0f,  1.0f);
    bias  += float2( 0.5f, -0.5f);
    scale *= 1.0f / SMAA_SEARCHTEX_PACKED_SIZE;
    bias  *= 1.0f / SMAA_SEARCHTEX_PACKED_SIZE;
    return searchSample(mad(scale, e, bias)).r;
}

float SMAASearchXLeft(float2 texcoord, float end) {
    const float4 rtm = rtmetrics();
    float2 e = float2(0.0f, 1.0f);
    [loop]
    while (texcoord.x > end && e.g > 0.8281f && e.r == 0.0f) {
        e        = edgesSample(texcoord).rg;
        texcoord = mad(-float2(2.0f, 0.0f), rtm.xy, texcoord);
    }
    const float offset = mad(-(255.0f / 127.0f), SMAASearchLength(e, 0.0f), 3.25f);
    return mad(rtm.x, offset, texcoord.x);
}

float SMAASearchXRight(float2 texcoord, float end) {
    const float4 rtm = rtmetrics();
    float2 e = float2(0.0f, 1.0f);
    [loop]
    while (texcoord.x < end && e.g > 0.8281f && e.r == 0.0f) {
        e        = edgesSample(texcoord).rg;
        texcoord = mad(float2(2.0f, 0.0f), rtm.xy, texcoord);
    }
    const float offset = mad(-(255.0f / 127.0f), SMAASearchLength(e, 0.5f), 3.25f);
    return mad(-rtm.x, offset, texcoord.x);
}

float SMAASearchYUp(float2 texcoord, float end) {
    const float4 rtm = rtmetrics();
    float2 e = float2(1.0f, 0.0f);
    [loop]
    while (texcoord.y > end && e.r > 0.8281f && e.g == 0.0f) {
        e        = edgesSample(texcoord).rg;
        texcoord = mad(-float2(0.0f, 2.0f), rtm.xy, texcoord);
    }
    const float offset = mad(-(255.0f / 127.0f), SMAASearchLength(e.gr, 0.0f), 3.25f);
    return mad(rtm.y, offset, texcoord.y);
}

float SMAASearchYDown(float2 texcoord, float end) {
    const float4 rtm = rtmetrics();
    float2 e = float2(1.0f, 0.0f);
    [loop]
    while (texcoord.y < end && e.r > 0.8281f && e.g == 0.0f) {
        e        = edgesSample(texcoord).rg;
        texcoord = mad(float2(0.0f, 2.0f), rtm.xy, texcoord);
    }
    const float offset = mad(-(255.0f / 127.0f), SMAASearchLength(e.gr, 0.5f), 3.25f);
    return mad(-rtm.y, offset, texcoord.y);
}

float2 SMAAArea(float2 dist, float e1, float e2, float offset) {
    float2 texcoord = mad(float2(SMAA_AREATEX_MAX_DISTANCE,
                                  SMAA_AREATEX_MAX_DISTANCE),
                          round(4.0f * float2(e1, e2)), dist);
    texcoord = mad(SMAA_AREATEX_PIXEL_SIZE, texcoord, 0.5f * SMAA_AREATEX_PIXEL_SIZE);
    texcoord.y = mad(SMAA_AREATEX_SUBTEX_SIZE, offset, texcoord.y);
    return areaSample(texcoord).rg;
}

// ---- Corner detection ----
void SMAADetectHorizontalCornerPattern(inout float2 weights, float4 texcoord, float2 d) {
    float2 leftRight = step(d.xy, d.yx);
    float2 rounding  = (1.0f - SMAA_CORNER_ROUNDING_NORM) * leftRight;
    rounding /= (leftRight.x + leftRight.y);

    float2 factor = float2(1.0f, 1.0f);
    factor.x -= rounding.x * edgesSampleOffset(texcoord.xy, int2(0,  1)).r;
    factor.x -= rounding.y * edgesSampleOffset(texcoord.zw, int2(1,  1)).r;
    factor.y -= rounding.x * edgesSampleOffset(texcoord.xy, int2(0, -2)).r;
    factor.y -= rounding.y * edgesSampleOffset(texcoord.zw, int2(1, -2)).r;

    weights *= saturate(factor);
}

void SMAADetectVerticalCornerPattern(inout float2 weights, float4 texcoord, float2 d) {
    float2 leftRight = step(d.xy, d.yx);
    float2 rounding  = (1.0f - SMAA_CORNER_ROUNDING_NORM) * leftRight;
    rounding /= (leftRight.x + leftRight.y);

    float2 factor = float2(1.0f, 1.0f);
    factor.x -= rounding.x * edgesSampleOffset(texcoord.xy, int2( 1, 0)).g;
    factor.x -= rounding.y * edgesSampleOffset(texcoord.zw, int2( 1, 1)).g;
    factor.y -= rounding.x * edgesSampleOffset(texcoord.xy, int2(-2, 0)).g;
    factor.y -= rounding.y * edgesSampleOffset(texcoord.zw, int2(-2, 1)).g;

    weights *= saturate(factor);
}

// ---- VS + PS ----
struct VSOutput {
    float4 sv_pos   : SV_Position;
    float2 uv       : TEXCOORD0;
    float2 pixcoord : TEXCOORD1;
    float4 offset0  : TEXCOORD2;
    float4 offset1  : TEXCOORD3;
    float4 offset2  : TEXCOORD4;
};

VSOutput VSMain(uint id : SV_VertexID) {
    float2 uv  = float2((id == 1) ? 2.0f : 0.0f,
                        (id == 2) ? 2.0f : 0.0f);
    float4 pos = float4(uv.x * 2.0f - 1.0f,
                        uv.y * 2.0f - 1.0f,
                        0.0f, 1.0f);

    const float4 rtm = float4(1.0f / float(pc.width), 1.0f / float(pc.height),
                              float(pc.width),        float(pc.height));

    VSOutput o;
    o.sv_pos   = pos;
    o.uv       = uv;
    o.pixcoord = uv * rtm.zw;
    o.offset0  = uv.xyxy + rtm.xyxy * float4(-0.25f, -0.125f,  1.25f, -0.125f);
    o.offset1  = uv.xyxy + rtm.xyxy * float4(-0.125f, -0.25f, -0.125f,  1.25f);
    o.offset2  = float4(o.offset0.xz, o.offset1.yw) +
                 rtm.xxyy * float4(-2.0f, 2.0f, -2.0f, 2.0f) * float(SMAA_MAX_SEARCH_STEPS);
    return o;
}

float4 PSMain(VSOutput input) : SV_Target0 {
    const float4 rtm            = rtmetrics();
    const float4 subsampleIdx   = float4(0.0f, 0.0f, 0.0f, 0.0f); // SMAA 1x
    float4 weights              = float4(0.0f, 0.0f, 0.0f, 0.0f);

    const float2 e = edgesSample(input.uv).rg;

    [branch]
    if (e.g > 0.0f) {  // Edge at north (horizontal edge above)
        weights.rg = SMAACalculateDiagWeights(input.uv, e, subsampleIdx);

        [branch]
        // No diagonal found → run the orthogonal pattern for a horizontal edge.
        if (weights.r == -weights.g) {
            float2 d;
            float3 coords;

            coords.x = SMAASearchXLeft(input.offset0.xy, input.offset2.x);
            coords.y = input.offset1.y;
            d.x      = coords.x;

            const float e1 = edgesSample(coords.xy).r;

            coords.z = SMAASearchXRight(input.offset0.zw, input.offset2.y);
            d.y      = coords.z;

            d = abs(round(mad(rtm.zz, d, -input.pixcoord.xx)));
            const float2 sqrt_d = sqrt(d);

            const float e2 = edgesSampleOffset(coords.zy, int2(1, 0)).r;

            weights.rg = SMAAArea(sqrt_d, e1, e2, subsampleIdx.y);

            coords.y = input.uv.y;
            SMAADetectHorizontalCornerPattern(weights.rg, coords.xyzy, d);
        }
    }

    [branch]
    if (e.r > 0.0f) {  // Edge at west (vertical edge to the left)
        float2 d;
        float3 coords;

        coords.y = SMAASearchYUp(input.offset1.xy, input.offset2.z);
        coords.x = input.offset0.x;
        d.x      = coords.y;

        const float e1 = edgesSample(coords.xy).g;

        coords.z = SMAASearchYDown(input.offset1.zw, input.offset2.w);
        d.y      = coords.z;

        d = abs(round(mad(rtm.ww, d, -input.pixcoord.yy)));
        const float2 sqrt_d = sqrt(d);

        const float e2 = edgesSampleOffset(coords.xz, int2(0, 1)).g;

        weights.ba = SMAAArea(sqrt_d, e1, e2, subsampleIdx.x);

        coords.x = input.uv.x;
        SMAADetectVerticalCornerPattern(weights.ba, coords.xyxz, d);
    }

    return weights;
}
