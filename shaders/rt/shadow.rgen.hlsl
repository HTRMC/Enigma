// shadow.rgen.hlsl
// ================
// Ray generation shader for RT shadows. Traces shadow rays from surface
// to directional sun light with cone jitter for soft penumbra.

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

struct PushBlock {
    uint  normalSlot;
    uint  depthSlot;
    uint  cameraSlot;
    uint  tlasSlot;
    uint  outputSlot;
    uint  _pad0;
    uint  _pad1;
    uint  _pad2;
    float4 lightDirIntensity; // xyz = sun dir, w = cone half-angle (radians)
};

[[vk::push_constant]] PushBlock pc;

struct ShadowPayload {
    float visibility; // 1.0 = lit, 0.0 = shadowed
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

[shader("raygeneration")]
void RayGenMain() {
    uint2 launchID   = DispatchRaysIndex().xy;
    uint2 launchSize = DispatchRaysDimensions().xy;

    float2 uv = (float2(launchID) + 0.5) / float2(launchSize);

    float4 normalSample = g_textures[NonUniformResourceIndex(pc.normalSlot)].Load(int3(launchID, 0));
    float  depth        = g_textures[NonUniformResourceIndex(pc.depthSlot)].Load(int3(launchID, 0)).r;

    // Reverse-Z: far = 0. Skip background pixels — fully lit.
    if (depth == 0.0) {
        g_storageImages[NonUniformResourceIndex(pc.outputSlot)][launchID] = float4(1, 1, 1, 1);
        return;
    }

    float3 N = normalize(normalSample.rgb * 2.0 - 1.0);

    // Reconstruct world position.
    float2 ndc    = uv * 2.0 - 1.0;
    float4 ndcPos = float4(ndc.x, ndc.y, depth, 1.0);
    CameraData cam = loadCamera(pc.cameraSlot);
    float4 worldPos4 = mul(cam.invViewProj, ndcPos);
    float3 worldPos  = worldPos4.xyz / worldPos4.w;

    float3 lightDir      = normalize(pc.lightDirIntensity.xyz);
    float  coneHalfAngle = pc.lightDirIntensity.w;

    // Build a tangent frame around the light direction for cone jitter.
    float3 tangent  = abs(lightDir.y) < 0.999 ? normalize(cross(lightDir, float3(0, 1, 0)))
                                               : normalize(cross(lightDir, float3(1, 0, 0)));
    float3 binormal = cross(lightDir, tangent);

    float totalVis = 0.0;
    for (uint i = 0; i < N_RAYS; ++i) {
        // Jitter direction within the cone.
        uint seed = launchID.x * 1973u + launchID.y * 9277u + i * 26699u;
        float u1 = frac(float(seed) / 4096.0);
        seed = seed * 747796405u + 2891336453u;
        float u2 = frac(float(seed) / 4096.0);

        float phi      = 2.0 * 3.14159265 * u1;
        float cosTheta = cos(coneHalfAngle * u2);
        float sinTheta = sin(coneHalfAngle * u2);

        float3 jitteredDir = normalize(lightDir * cosTheta +
                                       tangent * cos(phi) * sinTheta +
                                       binormal * sin(phi) * sinTheta);

        RayDesc ray;
        ray.Origin    = worldPos + N * 0.001;
        ray.Direction = jitteredDir;
        ray.TMin      = 0.001;
        ray.TMax      = 1000.0;

        ShadowPayload payload;
        payload.visibility = 0.0; // default: shadowed (hit = occluded)

        TraceRay(
            g_tlas[NonUniformResourceIndex(pc.tlasSlot)],
            RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
            0xFF,
            0, 1, 0,
            ray,
            payload);

        totalVis += payload.visibility;
    }

    float shadowTerm = totalVis / float(N_RAYS);
    g_storageImages[NonUniformResourceIndex(pc.outputSlot)][launchID] = float4(shadowTerm, shadowTerm, shadowTerm, 1.0);
}
