// wetroad.rgen.hlsl
// =================
// Ray generation shader for wet road reflections. Traces low-angle
// reflection rays from road-like surfaces, modulated by wetnessFactor.

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

// First 64 bytes match reflection.rmiss.hlsl PushBlock (shared miss shader).
// wetnessFactor lives after the miss-shader region at offset 64.
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
    float  wetnessFactor;
    uint   _pad0;
    uint   _pad1;
    uint   _pad2;
};

[[vk::push_constant]] PushBlock pc;

struct ReflectionPayload {
    float3 color;
    float  hitT;
};

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

[shader("raygeneration")]
void RayGenMain() {
    uint2 launchID   = DispatchRaysIndex().xy;
    uint2 launchSize = DispatchRaysDimensions().xy;

    float2 uv = (float2(launchID) + 0.5) / float2(launchSize);

    float4 normalSample = g_textures[NonUniformResourceIndex(pc.normalSlot)].Load(int3(launchID, 0));
    float  depth        = g_textures[NonUniformResourceIndex(pc.depthSlot)].Load(int3(launchID, 0)).r;

    // Skip background and non-road pixels.
    // Road surface classification: normal.y > 0.8 (roughly horizontal).
    float3 N = normalize(normalSample.rgb * 2.0 - 1.0);
    if (depth == 0.0 || N.y < 0.8 || pc.wetnessFactor < 0.001) {
        g_storageImages[NonUniformResourceIndex(pc.outputSlot)][launchID] = float4(0, 0, 0, 0);
        return;
    }

    // Reconstruct world position.
    float2 ndc    = uv * 2.0 - 1.0;
    float4 ndcPos = float4(ndc.x, ndc.y, depth, 1.0);
    CameraData cam = loadCamera(pc.cameraSlot);
    float4 worldPos4 = mul(cam.invViewProj, ndcPos);
    float3 worldPos  = worldPos4.xyz / worldPos4.w;

    // Compute reflection direction, perturbed by wetness.
    float3 V = normalize(cam.worldPos.xyz - worldPos);
    float3 reflectDir = reflect(-V, N);

    // Perturb reflection direction based on wetnessFactor (more wet = sharper reflection).
    uint seed = launchID.x * 1973u + launchID.y * 9277u + 12345u;
    float u1 = frac(float(seed) / 4096.0);
    seed = seed * 747796405u + 2891336453u;
    float u2 = frac(float(seed) / 4096.0);

    float roughness = 1.0 - pc.wetnessFactor; // dry = rough, wet = smooth
    float perturbAngle = roughness * 0.2;

    float3 tangent  = abs(reflectDir.y) < 0.999 ? normalize(cross(reflectDir, float3(0, 1, 0)))
                                                  : normalize(cross(reflectDir, float3(1, 0, 0)));
    float3 binormal = cross(reflectDir, tangent);

    float phi      = 2.0 * 3.14159265 * u1;
    float cosTheta = cos(perturbAngle * u2);
    float sinTheta = sin(perturbAngle * u2);

    float3 perturbedDir = normalize(reflectDir * cosTheta +
                                    tangent * cos(phi) * sinTheta +
                                    binormal * sin(phi) * sinTheta);

    RayDesc ray;
    ray.Origin    = worldPos + N * 0.001;
    ray.Direction = perturbedDir;
    ray.TMin      = 0.001;
    ray.TMax      = 500.0;

    ReflectionPayload payload;
    payload.color = float3(0, 0, 0);
    payload.hitT  = -1.0;

    TraceRay(
        g_tlas[NonUniformResourceIndex(pc.tlasSlot)],
        RAY_FLAG_FORCE_OPAQUE,
        0xFF,
        0, 1, 0,
        ray,
        payload);

    float3 result = payload.color * pc.wetnessFactor;
    g_storageImages[NonUniformResourceIndex(pc.outputSlot)][launchID] =
        float4(result, pc.wetnessFactor);
}
