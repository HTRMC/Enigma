// atmosphere_skyview.hlsl
// =======================
// Computes the SkyView LUT (192×108, R11G11B10_UFLOAT).
// Stores the sky radiance for any view direction as seen from the camera
// position, integrating single-scatter + multi-scatter contributions.
//
// UV parameterization (Hillaire 2020 "horizon-concentrated" mapping):
//   u = azimuth / (2π)
//   v = 0.5 + 0.5 * sign(cosZenith) * sqrt(|cosZenith|)
//       (packs both hemispheres; more samples near horizon where v ≈ 0.5)

#include "atmosphere_common.hlsl"

[[vk::binding(0, 0)]] Texture2D           g_textures[]      : register(t0, space0);
[[vk::binding(1, 0)]] RWTexture2D<float4> g_storageImages[] : register(u0, space0);
[[vk::binding(3, 0)]] SamplerState        g_samplers[]      : register(s0, space0);

struct PushBlock {
    float3 sunDirection;        // canonical sun dir FROM surface TO sun (world space)
    float  sunIntensity;
    float3 cameraWorldPos;      // world position of camera (in km from planet center)
    float  _pad;
    uint   transmittanceLutSlot;
    uint   multiScatterLutSlot;
    uint   skyViewStoreSlot;
    uint   samplerSlot;
};
[[vk::push_constant]] PushBlock pc;

// Decode SkyView LUT UV to view direction in a local frame where Y is up
// relative to the planet surface at the camera position.
float3 skyViewUVToDir(float2 uv) {
    float azimuth  = uv.x * 2.0f * PI;
    // v = 0.5 + 0.5 * sign(cos) * sqrt(|cos|) → invert:
    float vOff  = 2.0f * uv.y - 1.0f;             // ∈ [-1, 1]
    float cosZenith = sign(vOff) * vOff * vOff;    // ∈ [-1, 1]
    float sinZenith = sqrt(max(0.0f, 1.0f - cosZenith * cosZenith));
    return float3(sinZenith * cos(azimuth), cosZenith, sinZenith * sin(azimuth));
}

float3 computeSkyRadiance(float3 camPos, float3 viewDir, float3 sunDir) {
    float camAlt = max(0.0f, getAltitude(camPos));

    // Find ray entry/exit against top of atmosphere and planet
    float t0, t1;
    float tMax = 0.0f;
    if (raySphereIntersect(camPos, viewDir, R_TOP, t0, t1))
        tMax = max(0.0f, t1);
    else
        return float3(0.0f, 0.0f, 0.0f);

    float et0, et1;
    bool hitsGround = raySphereIntersect(camPos, viewDir, R_EARTH, et0, et1)
                   && et0 > 0.0f;
    if (hitsGround) tMax = min(tMax, et0);

    Texture2D    tlut  = g_textures[pc.transmittanceLutSlot];
    Texture2D    mslut = g_textures[pc.multiScatterLutSlot];
    SamplerState samp  = g_samplers[pc.samplerSlot];

    const int N_STEPS = 32;
    float dt = tMax / float(N_STEPS);

    float cosViewSun = dot(viewDir, sunDir);
    float phaseR = rayleighPhase(cosViewSun);
    float phaseM = miePhase(cosViewSun);

    float3 L          = float3(0, 0, 0); // accumulated in-scatter radiance
    float3 Ttotal     = float3(1, 1, 1); // transmittance along view ray

    for (int i = 0; i < N_STEPS; ++i) {
        float  t   = (float(i) + 0.5f) * dt;
        float3 sp  = camPos + viewDir * t;
        float  alt = max(0.0f, getAltitude(sp));

        float3 ext = totalExtinction(alt);
        float3 stepT = exp(-ext * dt);

        // Transmittance from sample point toward sun
        float cosSunAtSp = dot(normalize(sp), sunDir);
        float3 Tsun = sampleTransmittanceLUT(tlut, samp, alt, cosSunAtSp);

        // Single scatter: Rayleigh + Mie
        float3 sigmaR; float sigmaM;
        getScatteringCoeffs(alt, sigmaR, sigmaM);
        float3 singleScatter = (sigmaR * phaseR + sigmaM * phaseM)
                             * Tsun * pc.sunIntensity;

        // Multi-scatter: read from the 2D LUT
        float2 msUV = float2((cosSunAtSp + 1.0f) * 0.5f,
                             sqrt(saturate(alt / (R_TOP - R_EARTH))));
        float3 multiScatter = mslut.SampleLevel(samp, msUV, 0).rgb
                            * (sigmaR + sigmaM);

        // Integrate using exact analytic transmittance for each step segment
        float3 scatterInteg = (singleScatter + multiScatter) * Ttotal;
        float3 extSafe = max(float3(1e-9f, 1e-9f, 1e-9f), ext);
        L += scatterInteg * (1.0f - stepT) / extSafe;

        Ttotal *= stepT;
    }

    return L;
}

[numthreads(8, 8, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    uint2 dims;
    g_storageImages[pc.skyViewStoreSlot].GetDimensions(dims.x, dims.y);
    if (tid.x >= dims.x || tid.y >= dims.y) return;

    float2 uv = (float2(tid.xy) + 0.5f) / float2(dims);

    // Build local frame: Y = up from planet center at camera position
    float3 camPos = pc.cameraWorldPos;
    if (dot(camPos, camPos) < 1.0f)
        camPos = float3(0.0f, R_EARTH + 0.001f, 0.0f); // fallback at origin

    float3 up      = normalize(camPos);
    // Check cross-product length BEFORE normalising (same pattern as sky_background.hlsl).
    float3 rightRaw = cross(up, float3(0, 0, 1));
    if (length(rightRaw) < 0.01f) rightRaw = cross(up, float3(1, 0, 0));
    float3 right = normalize(rightRaw);
    float3 forward = cross(right, up);

    // Decode view direction in local frame
    float3 localDir = skyViewUVToDir(uv);
    float3 viewDir  = normalize(localDir.x * right + localDir.y * up + localDir.z * forward);

    float3 L = computeSkyRadiance(camPos, viewDir, pc.sunDirection);
    g_storageImages[pc.skyViewStoreSlot][tid.xy] = float4(L, 1.0f);
}
