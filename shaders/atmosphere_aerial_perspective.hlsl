// atmosphere_aerial_perspective.hlsl
// ===================================
// Bakes the Aerial Perspective volume (32×32×32, RGBA16F).
// Each froxel stores (in-scatter RGB, transmittance mono) for
// the segment from the camera to the froxel center.
//
// XY: screen-space normalized [0,1] × [0,1]
// Z: log-distributed slice depth from near to apSliceFar km
//
// Set 0: global bindless (transmittance + multi-scatter LUTs, samplers)
// Set 1, binding 0: RWTexture3D<float4> g_apVolume (write)

#include "atmosphere_common.hlsl"

[[vk::binding(0, 0)]] Texture2D           g_textures[]  : register(t0, space0);
[[vk::binding(3, 0)]] SamplerState        g_samplers[]  : register(s0, space0);

// Dedicated set 1 for the 3D volume (Texture2D[] cannot hold Texture3D in HLSL)
[[vk::binding(0, 1)]] RWTexture3D<float4> g_apVolume    : register(u0, space1);

struct PushBlock {
    float4x4 invViewProj;        // inverse view-projection for world pos reconstruction
    float3   sunDirection;
    float    sunIntensity;
    float3   cameraWorldPos;     // km from planet center
    float    apSliceFar;         // world-space distance of the far slice (km)
    uint     transmittanceLutSlot;
    uint     multiScatterLutSlot;
    uint     samplerSlot;
    uint     _pad;
};
[[vk::push_constant]] PushBlock pc;

// Slice index → world-space depth (km from camera), log-distributed
float sliceToDepth(float sliceNorm) {
    // sliceNorm ∈ [0,1], slice 0 = AP_NEAR km, slice 1 = apSliceFar km
    return AP_NEAR * pow(pc.apSliceFar / AP_NEAR, sliceNorm);
}

[numthreads(4, 4, 4)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    uint3 dims;
    g_apVolume.GetDimensions(dims.x, dims.y, dims.z);
    if (any(tid >= dims)) return;

    // Froxel center in normalized [0,1]^3 space
    float3 uvw = (float3(tid) + 0.5f) / float3(dims);

    // Reconstruct world position of froxel center
    float depth    = sliceToDepth(uvw.z);
    float2 ndc     = uvw.xy * 2.0f - 1.0f;
    float4 clipPos = float4(ndc.x, ndc.y, 0.5f, 1.0f);
    float4 worldH  = mul(pc.invViewProj, clipPos);
    float3 worldDir = normalize(worldH.xyz / worldH.w - pc.cameraWorldPos);
    float3 froxelPos = pc.cameraWorldPos + worldDir * depth;

    // Ray march from camera to froxel center, accumulating in-scatter and transmittance
    float3 camPos = pc.cameraWorldPos;
    float3 marchDir = worldDir;

    // Clamp march to atmosphere boundary
    float tMax = depth;
    float et0, et1;
    if (raySphereIntersect(camPos, marchDir, R_EARTH, et0, et1) && et0 > 0.0f)
        tMax = min(tMax, et0);
    float at0, at1;
    if (raySphereIntersect(camPos, marchDir, R_TOP, at0, at1))
        tMax = min(tMax, max(0.0f, at1));

    Texture2D    tlut  = g_textures[pc.transmittanceLutSlot];
    Texture2D    mslut = g_textures[pc.multiScatterLutSlot];
    SamplerState samp  = g_samplers[pc.samplerSlot];

    const int N_STEPS = 8;
    float dt = tMax / float(N_STEPS);

    float3 inScatter   = float3(0, 0, 0);
    float3 Ttotal      = float3(1, 1, 1);

    for (int i = 0; i < N_STEPS; ++i) {
        float  t   = (float(i) + 0.5f) * dt;
        float3 sp  = camPos + marchDir * t;
        float  alt = max(0.0f, getAltitude(sp));

        float3 ext   = totalExtinction(alt);
        float3 stepT = exp(-ext * dt);

        float3 sigmaR; float sigmaM;
        getScatteringCoeffs(alt, sigmaR, sigmaM);
        float3 scat = sigmaR + sigmaM;

        float cosSunAtSp = dot(normalize(sp), pc.sunDirection);
        float3 Tsun = sampleTransmittanceLUT(tlut, samp, alt, cosSunAtSp);

        float2 msUV = float2((cosSunAtSp + 1.0f) * 0.5f,
                             sqrt(saturate(alt / (R_TOP - R_EARTH))));
        float3 ms   = mslut.SampleLevel(samp, msUV, 0).rgb;

        // Isotropic phase for AP (angular detail comes from SkyView LUT at apply time)
        float uniformPhase = 1.0f / (4.0f * PI);
        float3 scatter = (Tsun * pc.sunIntensity + ms) * scat * uniformPhase;

        float3 extSafe = max(float3(1e-9f, 1e-9f, 1e-9f), ext);
        inScatter += Ttotal * scatter * (1.0f - stepT) / extSafe;
        Ttotal    *= stepT;
    }

    // Pack: rgb = in-scatter, a = average transmittance (luminance)
    float transmittanceMono = dot(Ttotal, float3(0.2126f, 0.7152f, 0.0722f));
    g_apVolume[tid] = float4(inScatter, transmittanceMono);
}
