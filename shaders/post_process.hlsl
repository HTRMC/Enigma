// post_process.hlsl
// =================
// Final screen-space pass that produces the display-ready image from the
// HDR linear intermediate (R16G16B16A16_SFLOAT). Performs, in order:
//
//   1. Aerial Perspective apply  — applies the AP volume to geometry pixels.
//      Sky pixels (depth==0) already have correct scattering from SkyView LUT.
//   2. Exposure                  — pow(2, exposureEV) * scene colour.
//   3. Bloom                     — threshold + 13-tap star-pattern single-pass
//                                  bloom (fast, no extra render targets; quality
//                                  can be upgraded to multi-pass Kawase later).
//   4. Tone mapping              — AgX (mode 0, default) or ACES (mode 1).
//
// Inputs:
//   Set 0: global bindless descriptors (textures, buffers, samplers)
//   Set 1: AP read set — Texture3D<float4> g_apVolume at binding 0
//
// The AP volume stores (in-scatter.rgb, transmittance.mono) baked each frame
// by AtmospherePass::updatePerFrame(). Froxel depth uses a log distribution
// matching atmosphere_aerial_perspective.hlsl exactly (near=0.01 km, far=50 km).

#include "common.hlsl"
#include "atmosphere_common.hlsl"

[[vk::binding(2, 0)]] StructuredBuffer<float4> g_buffers[]  : register(t0, space1);
[[vk::binding(0, 0)]] Texture2D               g_textures[] : register(t0, space0);
[[vk::binding(3, 0)]] SamplerState            g_samplers[] : register(s0, space0);

// Set 1: AP volume (Texture3D can't live in the bindless Texture2D[] array in HLSL)
[[vk::binding(0, 1)]] Texture3D<float4>        g_apVolume   : register(t0, space2);

struct PushBlock {
    uint   hdrColorSlot;
    uint   depthSlot;
    uint   cameraSlot;
    uint   samplerSlot;
    float4 cameraWorldPosKm;  // xyz = km from planet centre, w = unused
    float  exposureEV;
    float  bloomThreshold;
    float  bloomIntensity;
    uint   tonemapMode;       // 0 = AgX, 1 = ACES
    uint   bloomEnabled;
    uint   apEnabled;
    uint   _pad0;
    uint   _pad1;
};
[[vk::push_constant]] PushBlock pc;

// AP froxel depth constants come from atmosphere_common.hlsl (AP_NEAR, AP_FAR).

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

// ---- Tone mapping ----

// AgX: converts linear scene colour to display-encoded sRGB.
// Based on Troy Sobotka's AgX reference (Blender 4.x implementation).
float3 agxDefaultContrastApprox(float3 x) {
    float3 x2 = x * x;
    float3 x4 = x2 * x2;
    return  15.5f    * x4 * x2
           - 40.14f  * x4 * x
           + 31.96f  * x4
           - 6.868f  * x2 * x
           + 0.4298f * x2
           + 0.1191f * x
           - 0.00232f;
}

float3 AgX(float3 val) {
    // Input: linear BT.709 sRGB, any positive value.
    static const float3x3 agxInset = float3x3(
        0.842479062253094f, 0.042328422610123f, 0.042375654905705f,
        0.078433599999999f, 0.878468636469772f, 0.078433600000000f,
        0.079223745147764f, 0.079166127460543f, 0.879142973793104f
    );
    static const float3x3 agxOutset = float3x3(
        1.196879005120170f, -0.052896851757456f, -0.052971635514444f,
       -0.098020881140137f,  1.151903129904170f, -0.098043450117124f,
       -0.099029744079721f, -0.098961176844843f,  1.151073672641160f
    );

    val = mul(agxInset, max(val, float3(1e-10f, 1e-10f, 1e-10f)));
    val = clamp(log2(val), -12.47393f, 4.026069f);
    val = (val - (-12.47393f)) / (4.026069f - (-12.47393f)); // normalise [0,1]
    val = agxDefaultContrastApprox(val);
    return max(float3(0.0f, 0.0f, 0.0f), mul(agxOutset, val));
}

float3 ACES(float3 x) {
    return saturate((x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f));
}

float3 tonemap(float3 color, uint mode) {
    return (mode == 1u) ? ACES(color) : AgX(color);
}

// ---- Vertex shader: fullscreen triangle ----
struct VSOut {
    float4 pos      : SV_Position;
    float2 texCoord : TEXCOORD0;
};

VSOut VSMain(uint vid : SV_VertexID) {
    float2 uv = float2((vid << 1) & 2, vid & 2);
    VSOut o;
    o.pos      = float4(uv * 2.0f - 1.0f, 0.0f, 1.0f);
    o.texCoord = uv;
    return o;
}

// ---- Pixel shader ----
float4 PSMain(VSOut vs) : SV_Target {
    float2 uv   = vs.texCoord;
    SamplerState samp = g_samplers[NonUniformResourceIndex(pc.samplerSlot)];

    Texture2D hdrTex  = g_textures[NonUniformResourceIndex(pc.hdrColorSlot)];
    float3 color      = hdrTex.SampleLevel(samp, uv, 0).rgb;

    // ---- 1. Aerial Perspective apply (geometry pixels only) ----
    if (pc.apEnabled != 0u) {
        float depth = g_textures[NonUniformResourceIndex(pc.depthSlot)].Sample(samp, uv).r;
        if (depth != 0.0f) {
            CameraData cam  = loadCamera(pc.cameraSlot);
            float2 ndc      = uv * 2.0f - 1.0f;
            float4 worldH   = mul(cam.invViewProj, float4(ndc, depth, 1.0f));
            float3 worldPos = worldH.xyz / worldH.w;
            float3 worldKm  = worldPos * 0.001f; // metres → km

            float depthKm = length(worldKm - pc.cameraWorldPosKm.xyz);
            depthKm = max(depthKm, AP_NEAR);

            // Log-distributed slice index matching atmosphere_aerial_perspective.hlsl
            float w = saturate(log(depthKm / AP_NEAR) / log(AP_FAR / AP_NEAR));
            float4 ap = g_apVolume.SampleLevel(samp, float3(uv, w), 0);

            // ap.rgb = in-scatter, ap.a = average transmittance
            color = color * ap.a + ap.rgb;
        }
    }

    // ---- 2. Exposure ----
    const float expMul = pow(2.0f, pc.exposureEV);
    color *= expMul;

    // ---- 3. Bloom ----
    if (pc.bloomEnabled != 0u) {
        uint w, h;
        hdrTex.GetDimensions(w, h);
        float2 texelSize = 1.0f / float2(w, h);

        // 13-tap star-pattern bloom (no extra render targets).
        // Each tap samples at a fixed multiple of the texel size; only
        // light above the threshold contributes.  Multiplying the offset
        // by a bloom-radius scale gives a soft halo at screen resolution.
        static const float2 kOffsets[13] = {
            float2(-2.0f,  0.0f), float2( 2.0f,  0.0f),
            float2( 0.0f, -2.0f), float2( 0.0f,  2.0f),
            float2(-1.0f, -1.0f), float2( 1.0f, -1.0f),
            float2(-1.0f,  1.0f), float2( 1.0f,  1.0f),
            float2(-3.0f,  0.0f), float2( 3.0f,  0.0f),
            float2( 0.0f, -3.0f), float2( 0.0f,  3.0f),
            float2( 0.0f,  0.0f),
        };
        const float kBloomScale = 4.0f; // texel radius multiplier

        // Threshold on luminance, not per-channel, so saturated highlights
        // bloom with correct color instead of desaturating to grey halos.
        float3 bloomSum = float3(0.0f, 0.0f, 0.0f);
        [unroll]
        for (int i = 0; i < 13; ++i) {
            float3 c = hdrTex.SampleLevel(samp,
                uv + kOffsets[i] * texelSize * kBloomScale, 0).rgb * expMul;
            float luma = dot(c, float3(0.2126f, 0.7152f, 0.0722f));
            float knee = max(0.0f, luma - pc.bloomThreshold);
            // Colour-preserving weight: ratio in [0,1] so contribution stays
            // proportional to the original colour, not quadratically amplified.
            bloomSum += c * (knee / max(luma, 1e-6f));
        }
        color += bloomSum * (pc.bloomIntensity / 13.0f);
    }

    // ---- 4. Tone mapping ----
    color = tonemap(color, pc.tonemapMode);

    return float4(color, 1.0f);
}
