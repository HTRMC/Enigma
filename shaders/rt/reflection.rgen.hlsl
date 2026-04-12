// reflection.rgen.hlsl
// ====================
// Ray generation shader for RT reflections. Reads G-buffer world
// positions + normals, casts reflection rays, writes to an RGBA16F
// storage image.

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
    uint   normalSlot;
    uint   depthSlot;
    uint   cameraSlot;
    uint   samplerSlot;
    uint   tlasSlot;
    uint   outputSlot;
    uint   skyViewLutSlot;        // SkyView LUT slot for miss shader sky sampling
    uint   transmittanceLutSlot;  // Transmittance LUT slot for sun disk
    float4 sunWorldDirIntensity;  // xyz = sun direction, w = intensity
    float4 cameraWorldPosKm;      // xyz = camera position in km from planet centre
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

    SamplerState samp = g_samplers[NonUniformResourceIndex(pc.samplerSlot)];

    // Sample G-buffer.
    float4 normalSample = g_textures[NonUniformResourceIndex(pc.normalSlot)].SampleLevel(samp, uv, 0);
    float  depth        = g_textures[NonUniformResourceIndex(pc.depthSlot)].SampleLevel(samp, uv, 0).r;

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

    // Compute reflection direction.
    float3 V          = normalize(cam.worldPos.xyz - worldPos);
    float3 reflectDir = reflect(-V, N);

    // Trace reflection ray.
    RayDesc ray;
    ray.Origin    = worldPos + N * 0.001; // small offset to avoid self-intersection
    ray.Direction = reflectDir;
    ray.TMin      = 0.001;
    ray.TMax      = 1000.0;

    ReflectionPayload payload;
    payload.color = float3(0, 0, 0);
    payload.hitT  = -1.0;

    TraceRay(
        g_tlas[NonUniformResourceIndex(pc.tlasSlot)],
        RAY_FLAG_FORCE_OPAQUE,
        0xFF,   // instance mask
        0,      // SBT offset (hit group 0)
        1,      // SBT stride
        0,      // miss index
        ray,
        payload);

    g_storageImages[NonUniformResourceIndex(pc.outputSlot)][launchID] =
        float4(payload.color, payload.hitT >= 0.0 ? 1.0 : 0.0);
}
