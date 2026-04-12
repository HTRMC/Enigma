// atmosphere_transmittance.hlsl
// =============================
// Computes the Transmittance LUT (256×64, R11G11B10_UFLOAT).
// Stores T(h, cos_zenith) = exp(-∫ extinction ds) along the ray from
// the point at altitude h km toward the direction given by cos_zenith,
// integrating until the ray exits the atmosphere (or hits the planet).
//
// UV parameterization:
//   u = (cos_zenith + 1) / 2   → cos_zenith ∈ [-1, 1]
//   v = sqrt(h / H_ATM)        → h ∈ [0, H_ATM] km, sqrt for surface precision

#include "atmosphere_common.hlsl"

[[vk::binding(1, 0)]] RWTexture2D<float4> g_storageImages[] : register(u0, space0);

struct PushBlock {
    uint storageSlot;
    uint _pad0, _pad1, _pad2;
};
[[vk::push_constant]] PushBlock pc;

float3 computeTransmittance(float h, float cosZenith) {
    float3 origin = float3(0.0f, R_EARTH + h, 0.0f);
    float  sinZenith = sqrt(max(0.0f, 1.0f - cosZenith * cosZenith));
    float3 dir    = float3(sinZenith, cosZenith, 0.0f);

    // Find exit distance — top of atmosphere
    float t0, t1;
    float tMax = 0.0f;
    if (raySphereIntersect(origin, dir, R_TOP, t0, t1)) {
        tMax = max(0.0f, t1);
    } else {
        return float3(1.0f, 1.0f, 1.0f); // outside atmosphere
    }

    // If ray hits planet before TOA, transmittance is near-zero (shadow column)
    float et0, et1;
    if (raySphereIntersect(origin, dir, R_EARTH, et0, et1) && et0 > 0.0f) {
        return float3(0.0f, 0.0f, 0.0f);
    }

    // Ray-march and accumulate optical depth
    const int STEPS = 256;
    float dt = tMax / float(STEPS);
    float3 opticalDepth = float3(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < STEPS; ++i) {
        float  t   = (float(i) + 0.5f) * dt;
        float3 pos = origin + dir * t;
        float  alt = max(0.0f, getAltitude(pos));
        opticalDepth += totalExtinction(alt) * dt;
    }

    return exp(-opticalDepth);
}

[numthreads(8, 8, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    uint2 dims;
    g_storageImages[pc.storageSlot].GetDimensions(dims.x, dims.y);
    if (tid.x >= dims.x || tid.y >= dims.y) return;

    float2 uv = (float2(tid.xy) + 0.5f) / float2(dims);
    float h, cosZenith;
    transmittanceUVToParams(uv, h, cosZenith);

    float3 T = computeTransmittance(h, cosZenith);
    g_storageImages[pc.storageSlot][tid.xy] = float4(T, 1.0f);
}
