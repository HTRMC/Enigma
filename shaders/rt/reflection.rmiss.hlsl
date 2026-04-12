// reflection.rmiss.hlsl
// =====================
// Miss shader for RT reflections. Samples the Hillaire 2020 SkyView LUT
// for reflection rays that escape the scene, plus a physically-based sun disk.
// Falls back to a sky gradient if the LUT slot is not yet initialised (slot 0).

#include "../atmosphere_common.hlsl"

[[vk::binding(0, 0)]] Texture2D    g_textures[] : register(t0, space0);
[[vk::binding(3, 0)]] SamplerState g_samplers[] : register(s0, space0);

// Must exactly match reflection.rgen.hlsl PushBlock.
struct PushBlock {
    uint   normalSlot;
    uint   depthSlot;
    uint   cameraSlot;
    uint   samplerSlot;
    uint   tlasSlot;
    uint   outputSlot;
    uint   skyViewLutSlot;
    uint   transmittanceLutSlot;
    float4 sunWorldDirIntensity;
    float4 cameraWorldPosKm;
};
[[vk::push_constant]] PushBlock pc;

struct ReflectionPayload {
    float3 color;
    float  hitT;
};

// Convert a world-space direction to SkyView LUT UV.
// Matches the inverse of atmosphere_skyview.hlsl's skyViewUVToDir().
float2 dirToSkyViewUV(float3 viewDir, float3 camPosKm) {
    float3 up       = normalize(camPosKm);
    // Check cross-product length BEFORE normalising to handle near-pole cameras.
    float3 rightRaw = cross(up, float3(0.0f, 0.0f, 1.0f));
    if (length(rightRaw) < 0.01f)
        rightRaw = cross(up, float3(1.0f, 0.0f, 0.0f));
    float3 right   = normalize(rightRaw);
    float3 forward = cross(right, up);

    float localX = dot(viewDir, right);
    float localY = dot(viewDir, up);
    float localZ = dot(viewDir, forward);

    float az = atan2(localZ, localX);
    float u  = frac(az / (2.0f * PI) + 1.0f);
    // sign(0) is undefined on some drivers → use explicit ternary for the horizon row.
    float signY = (localY >= 0.0f) ? 1.0f : -1.0f;
    float v  = 0.5f + 0.5f * signY * sqrt(abs(localY));
    return float2(u, v);
}

[shader("miss")]
void MissMain(inout ReflectionPayload payload) {
    float3 rayDir = normalize(WorldRayDirection());

    float3 camPosKm = pc.cameraWorldPosKm.xyz;
    if (dot(camPosKm, camPosKm) < 1.0f)
        camPosKm = float3(0.0f, R_EARTH + 0.001f, 0.0f);

    SamplerState samp = g_samplers[NonUniformResourceIndex(pc.samplerSlot)];

    // Sample SkyView LUT.
    Texture2D svLut  = g_textures[NonUniformResourceIndex(pc.skyViewLutSlot)];
    float2 svUV      = dirToSkyViewUV(rayDir, camPosKm);
    float3 skyColor  = svLut.SampleLevel(samp, svUV, 0).rgb;

    // Sun disk.
    float3 sunDir  = normalize(pc.sunWorldDirIntensity.xyz);
    float  cosView = dot(rayDir, sunDir);
    const float kCosSunRadius = 0.99998f;
    const float kCosSunEdge   = 0.99990f;
    if (cosView > kCosSunEdge) {
        float  camAlt   = max(0.0f, getAltitude(camPosKm));
        float  cosSunUp = dot(normalize(camPosKm), sunDir);
        Texture2D tlut  = g_textures[NonUniformResourceIndex(pc.transmittanceLutSlot)];
        float3 Tsun     = sampleTransmittanceLUT(tlut, samp, camAlt, cosSunUp);
        float  mask     = smoothstep(kCosSunEdge, kCosSunRadius, cosView);
        skyColor += Tsun * pc.sunWorldDirIntensity.w * 14706.0f * mask;
    }

    payload.color = skyColor;
    payload.hitT  = -1.0f;
}
