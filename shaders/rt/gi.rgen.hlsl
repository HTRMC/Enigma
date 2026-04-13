// gi.rgen.hlsl
// ============
// Ray generation shader for RT global illumination. Traces hemisphere-
// distributed rays from G-buffer surface points to gather single-bounce
// diffuse indirect illumination.

#include "../common.hlsl"

// --- Bindless resource arrays ---
[[vk::binding(0, 0)]]
Texture2D g_textures[] : register(t0, space0);

[[vk::binding(1, 0)]]
RWTexture2D<float4> g_storageImages[] : register(u0, space0);

[[vk::binding(2, 0)]]
StructuredBuffer<float4> g_buffers[] : register(t0, space1);

[[vk::binding(3, 0)]]
SamplerState g_samplers[] : register(s0, space0);

[[vk::binding(4, 0)]]
RaytracingAccelerationStructure g_tlas[] : register(t0, space2);

// Must exactly match reflection.rmiss.hlsl PushBlock (shared miss shader needs 64 bytes).
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

struct GIPayload {
    float3 color;
    float  hitT;
};

static const uint N_RAYS = 4;

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

// Simple cosine-weighted hemisphere sample using a hash-based RNG.
float3 cosineHemisphere(float3 N, uint2 pixel, uint rayIdx) {
    // Simple hash for pseudo-random direction.
    uint seed = pixel.x * 1973u + pixel.y * 9277u + rayIdx * 26699u;
    float u1 = frac(float(seed) / 4096.0);
    seed = seed * 747796405u + 2891336453u;
    float u2 = frac(float(seed) / 4096.0);

    float phi       = 2.0 * 3.14159265 * u1;
    float cosTheta  = sqrt(1.0 - u2);
    float sinTheta  = sqrt(u2);

    float3 tangent  = abs(N.y) < 0.999 ? normalize(cross(N, float3(0, 1, 0)))
                                        : normalize(cross(N, float3(1, 0, 0)));
    float3 binormal = cross(N, tangent);

    return normalize(tangent * cos(phi) * sinTheta +
                     binormal * sin(phi) * sinTheta +
                     N * cosTheta);
}

[shader("raygeneration")]
void RayGenMain() {
    uint2 launchID   = DispatchRaysIndex().xy;
    uint2 launchSize = DispatchRaysDimensions().xy;

    float2 uv = (float2(launchID) + 0.5) / float2(launchSize);

    // Sample G-buffer directly (no sampler needed for RT path).
    float4 normalSample = g_textures[NonUniformResourceIndex(pc.normalSlot)].Load(int3(launchID, 0));
    float  depth        = g_textures[NonUniformResourceIndex(pc.depthSlot)].Load(int3(launchID, 0)).r;

    // Reverse-Z: far = 0. Skip background pixels.
    if (depth == 0.0) {
        g_storageImages[NonUniformResourceIndex(pc.outputSlot)][launchID] = float4(0, 0, 0, 0);
        return;
    }

    // Decode world normal.
    float3 N = normalize(normalSample.rgb * 2.0 - 1.0);

    // Reconstruct world position from depth.
    float2 ndc    = uv * 2.0 - 1.0;
    float4 ndcPos = float4(ndc.x, ndc.y, depth, 1.0);
    CameraData cam = loadCamera(pc.cameraSlot);
    float4 worldPos4 = mul(cam.invViewProj, ndcPos);
    float3 worldPos  = worldPos4.xyz / worldPos4.w;

    // Trace N_RAYS hemisphere-distributed rays for diffuse GI.
    float3 irradiance = float3(0, 0, 0);
    for (uint i = 0; i < N_RAYS; ++i) {
        float3 dir = cosineHemisphere(N, launchID, i);

        RayDesc ray;
        ray.Origin    = worldPos + N * 0.001;
        ray.Direction = dir;
        ray.TMin      = 0.001;
        ray.TMax      = 100.0;

        GIPayload payload;
        payload.color = float3(0, 0, 0);
        payload.hitT  = -1.0;

        TraceRay(
            g_tlas[NonUniformResourceIndex(pc.tlasSlot)],
            RAY_FLAG_FORCE_OPAQUE,
            0xFF,
            0, 1, 0,
            ray,
            payload);

        irradiance += payload.color;
    }
    irradiance /= float(N_RAYS);

    g_storageImages[NonUniformResourceIndex(pc.outputSlot)][launchID] =
        float4(irradiance, 1.0);
}
