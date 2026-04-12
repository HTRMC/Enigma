// atmosphere_multiscatter.hlsl
// ============================
// Computes the Multi-Scattering LUT (32×32, R11G11B10_UFLOAT).
// Based on Hillaire 2020, Section 5.2: approximates infinite-order
// multiple scattering as a 2D function of (altitude, cos_sun_zenith).
//
// Algorithm (per pixel):
//   1. Sample N uniformly distributed sphere directions.
//   2. For each direction d_i: march a ray, accumulating
//      - L_scatter: single-scatter radiance in d_i (using transmittance LUT)
//      - f_ms:      fraction of in-scattered energy available for re-scatter
//   3. ψ_ms = mean(L_scatter) / (1 − mean(f_ms))   (geometric series limit)
//
// UV parameterization (same as transmittance LUT):
//   u = (cos_sun_zenith + 1) / 2   → cos_sun_zenith ∈ [-1, 1]
//   v = sqrt(h / H_ATM)            → h ∈ [0, H_ATM] km

#include "atmosphere_common.hlsl"

[[vk::binding(0, 0)]] Texture2D           g_textures[]     : register(t0, space0);
[[vk::binding(1, 0)]] RWTexture2D<float4> g_storageImages[]: register(u0, space0);
[[vk::binding(3, 0)]] SamplerState        g_samplers[]     : register(s0, space0);

struct PushBlock {
    float3 sunDirection;       // canonical sun world dir (not used directly here)
    float  sunIntensity;
    uint   transmittanceLutSlot;
    uint   multiScatterStoreSlot;
    uint   samplerSlot;
    uint   _pad;
};
[[vk::push_constant]] PushBlock pc;

// Uniformly distributed sphere direction from index i out of N.
// Uses the Fibonacci / Archimedes spiral heuristic.
float3 uniformSphereDir(uint i, uint N) {
    float phi      = 2.0f * PI * float(i) / float(N);
    float cosTheta = 1.0f - (2.0f * float(i) + 1.0f) / float(N);
    float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
    return float3(sinTheta * cos(phi), cosTheta, sinTheta * sin(phi));
}

struct ScatterResult {
    float3 Lms;   // multi-scatter luminance contribution
    float3 fms;   // re-scatter fraction
};

ScatterResult marchDirection(float3 pos0, float3 rd, float3 sunDir,
                              Texture2D tlut, SamplerState samp) {
    ScatterResult r;
    r.Lms = float3(0, 0, 0);
    r.fms = float3(0, 0, 0);

    // Find march length to TOA (possibly shortened by planet intersection)
    float t0, t1;
    float tMax = 0.0f;
    if (raySphereIntersect(pos0, rd, R_TOP, t0, t1))
        tMax = max(0.0f, t1);
    else
        return r;

    float et0, et1;
    if (raySphereIntersect(pos0, rd, R_EARTH, et0, et1) && et0 > 0.0f)
        tMax = min(tMax, et0);

    if (tMax <= 0.0f) return r;

    const int N_STEPS = 32;
    float dt = tMax / float(N_STEPS);
    float3 accOptDepth = float3(0, 0, 0);

    for (int s = 0; s < N_STEPS; ++s) {
        float  t   = (float(s) + 0.5f) * dt;
        float3 sp  = pos0 + rd * t;
        float  alt = max(0.0f, getAltitude(sp));

        float3 ext = totalExtinction(alt);
        float3 T   = exp(-accOptDepth);

        float3 sigmaR; float sigmaM;
        getScatteringCoeffs(alt, sigmaR, sigmaM);
        float3 scatter = sigmaR + sigmaM;

        // Transmittance from sp toward sun (sampled from LUT)
        float cosSunAtSp = dot(normalize(sp), sunDir);
        float3 Tsun = sampleTransmittanceLUT(tlut, samp, alt, cosSunAtSp);

        // Isotropic phase (1/4π) — used to preintegrate multi-scatter equally
        // over all directions (the actual phase is applied at SkyView LUT time)
        float uniformPhase = 1.0f / (4.0f * PI);
        r.Lms += T * Tsun * scatter * uniformPhase * pc.sunIntensity * dt;

        // f_ms: fraction of scattered energy surviving per scatter event
        float3 safeExt = max(float3(1e-9f, 1e-9f, 1e-9f), ext);
        r.fms += T * scatter / safeExt * dt;

        accOptDepth += ext * dt;
    }
    return r;
}

[numthreads(8, 8, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    uint2 dims;
    g_storageImages[pc.multiScatterStoreSlot].GetDimensions(dims.x, dims.y);
    if (tid.x >= dims.x || tid.y >= dims.y) return;

    float2 uv          = (float2(tid.xy) + 0.5f) / float2(dims);
    float cosSunZenith = 2.0f * uv.x - 1.0f;
    float h            = (R_TOP - R_EARTH) * uv.y * uv.y;

    float3 pos0   = float3(0.0f, R_EARTH + h, 0.0f);
    float3 sunDir = float3(sqrt(max(0.0f, 1.0f - cosSunZenith * cosSunZenith)),
                           cosSunZenith, 0.0f);

    Texture2D    tlut = g_textures[pc.transmittanceLutSlot];
    SamplerState samp = g_samplers[pc.samplerSlot];

    float3 Lms_sum = float3(0, 0, 0);
    float3 fms_sum = float3(0, 0, 0);

    const uint N_DIRS = 64;
    for (uint i = 0; i < N_DIRS; ++i) {
        float3 rd = uniformSphereDir(i, N_DIRS);
        ScatterResult sr = marchDirection(pos0, rd, sunDir, tlut, samp);
        Lms_sum += sr.Lms;
        fms_sum += sr.fms;
    }

    float3 Lms_mean = Lms_sum / float(N_DIRS);
    float3 fms_mean = fms_sum / float(N_DIRS);

    // Infinite-order approximation: geometric series sum
    float3 psi_ms = Lms_mean / max(float3(1e-9f, 1e-9f, 1e-9f), 1.0f - fms_mean);

    g_storageImages[pc.multiScatterStoreSlot][tid.xy] = float4(psi_ms, 1.0f);
}
