// visibility_buffer.task.hlsl
// ===========================
// Amplification (task) shader for the visibility buffer pipeline. Each
// workgroup processes TASK_GROUP_SIZE meshlets, performs frustum culling,
// and dispatches surviving meshlets to the mesh shader stage.
//
// Dispatch: one workgroup per batch of TASK_GROUP_SIZE meshlets.
// Depth convention: reverse-Z (far = 0, near = 1).

#include "common.hlsl"

#define TASK_GROUP_SIZE 32

// --- Bindless resource arrays ---
[[vk::binding(2, 0)]]
StructuredBuffer<float4> g_buffers[] : register(t0, space1);

[[vk::binding(5, 0)]]
RWByteAddressBuffer g_rwBuffers[] : register(u1, space0);

// --- Push constants (unified with visibility_buffer.mesh.hlsl) ---
struct PushBlock {
    uint instanceBufferSlot;    // both task+mesh: GpuInstance[]
    uint meshletBufferSlot;     // both task+mesh: Meshlet[]
    uint survivingIdsSlot;      // task: surviving global meshlet IDs from gpu_cull
    uint meshletVerticesSlot;   // mesh: vertex index remapping (u32[])
    uint meshletTrianglesSlot;  // mesh: packed u8 triangle indices
    uint cameraSlot;            // both: camera matrices
    uint totalSurviving;        // task: total surviving meshlet count
    uint instanceCount;         // task: number of GpuInstance entries
};

[[vk::push_constant]] PushBlock pc;

// --- GPU struct accessors (same layout as gpu_cull.comp.hlsl) ---
struct GpuInstance {
    float4x4 transform;
    uint     meshlet_offset;
    uint     meshlet_count;
    uint     material_index;
    uint     vertex_buffer_slot;
};

GpuInstance loadInstance(uint idx) {
    StructuredBuffer<float4> buf = g_buffers[NonUniformResourceIndex(pc.instanceBufferSlot)];
    const uint base = idx * 5;
    GpuInstance inst;
    inst.transform = transpose(float4x4(buf[base + 0], buf[base + 1], buf[base + 2], buf[base + 3]));
    uint4 packed   = asuint(buf[base + 4]);
    inst.meshlet_offset    = packed.x;
    inst.meshlet_count     = packed.y;
    inst.material_index    = packed.z;
    inst.vertex_buffer_slot = packed.w;
    return inst;
}

struct Meshlet {
    uint   vertex_offset;
    uint   triangle_offset;
    uint   vertex_count;
    uint   triangle_count;
    float3 center;
    float  radius;
    float3 cone_axis;
    float  cone_cutoff;
};

Meshlet loadMeshlet(uint idx) {
    StructuredBuffer<float4> buf = g_buffers[NonUniformResourceIndex(pc.meshletBufferSlot)];
    const uint base = idx * 3;
    uint4  m0 = asuint(buf[base + 0]);
    float4 m1 = buf[base + 1];
    float4 m2 = buf[base + 2];

    Meshlet m;
    m.vertex_offset   = m0.x;
    m.triangle_offset = m0.y;
    m.vertex_count    = m0.z;
    m.triangle_count  = m0.w;
    m.center          = m1.xyz;
    m.radius          = m1.w;
    m.cone_axis       = m2.xyz;
    m.cone_cutoff     = m2.w;
    return m;
}

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

// --- Frustum ---
struct Frustum {
    float4 planes[6];
};

Frustum extractFrustum(float4x4 vp) {
    Frustum f;
    f.planes[0] = float4(vp[0][3] + vp[0][0], vp[1][3] + vp[1][0], vp[2][3] + vp[2][0], vp[3][3] + vp[3][0]); // Left
    f.planes[1] = float4(vp[0][3] - vp[0][0], vp[1][3] - vp[1][0], vp[2][3] - vp[2][0], vp[3][3] - vp[3][0]); // Right
    f.planes[2] = float4(vp[0][3] + vp[0][1], vp[1][3] + vp[1][1], vp[2][3] + vp[2][1], vp[3][3] + vp[3][1]); // Bottom
    f.planes[3] = float4(vp[0][3] - vp[0][1], vp[1][3] - vp[1][1], vp[2][3] - vp[2][1], vp[3][3] - vp[3][1]); // Top
    f.planes[4] = float4(vp[0][3] - vp[0][2], vp[1][3] - vp[1][2], vp[2][3] - vp[2][2], vp[3][3] - vp[3][2]); // Near
    f.planes[5] = float4(vp[0][2], vp[1][2], vp[2][2], vp[3][2]);                                               // Far

    [unroll]
    for (int i = 0; i < 6; ++i) {
        float len = length(f.planes[i].xyz);
        f.planes[i] /= len;
    }
    return f;
}

bool sphereInsideFrustum(Frustum frust, float3 center, float radius) {
    [unroll]
    for (int i = 0; i < 6; ++i) {
        if (dot(frust.planes[i].xyz, center) + frust.planes[i].w < -radius)
            return false;
    }
    return true;
}

// --- Mesh payload (shared with visibility_buffer.mesh.hlsl) ---
struct MeshPayload {
    uint meshlet_indices[TASK_GROUP_SIZE]; // global meshlet IDs
    uint instance_ids[TASK_GROUP_SIZE];    // corresponding instance IDs
};

groupshared MeshPayload s_payload;
groupshared uint s_surviving_count;

// --- Find which instance owns a global meshlet index ---
// Walks the instance buffer linearly. The gpu_cull pass already filtered by
// totalSurviving so the surviving IDs buffer is compact.
void findInstanceAndLocal(uint globalMeshletIdx, uint instanceCount,
                          out uint instanceId, out uint localMeshletIdx) {
    uint offset = 0;
    for (uint i = 0; i < instanceCount; ++i) {
        StructuredBuffer<float4> buf = g_buffers[NonUniformResourceIndex(pc.instanceBufferSlot)];
        uint4 packed = asuint(buf[i * 5 + 4]);
        uint count = packed.y;
        if (globalMeshletIdx < offset + count) {
            instanceId      = i;
            localMeshletIdx = globalMeshletIdx - offset;
            return;
        }
        offset += count;
    }
    instanceId      = 0;
    localMeshletIdx = 0;
}

// --- Main ---
[numthreads(TASK_GROUP_SIZE, 1, 1)]
void ASMain(
    in uint3 groupId    : SV_GroupID,
    in uint  groupIndex : SV_GroupIndex
) {
    // Initialize shared counter.
    if (groupIndex == 0)
        s_surviving_count = 0;
    GroupMemoryBarrierWithGroupSync();

    // Global meshlet slot for this thread (from the surviving IDs written by gpu_cull).
    uint batchBase   = groupId.x * TASK_GROUP_SIZE;
    uint meshletSlot = batchBase + groupIndex;

    bool valid = meshletSlot < pc.totalSurviving;

    // Read the global meshlet ID from the surviving IDs buffer.
    uint globalMeshletId = 0;
    uint instanceId = 0;
    bool visible = false;

    if (valid) {
        RWByteAddressBuffer survivingBuf = g_rwBuffers[NonUniformResourceIndex(pc.survivingIdsSlot)];
        globalMeshletId = survivingBuf.Load(meshletSlot * 4);

        // Resolve instance from global meshlet ID for payload.
        uint localMeshletId;
        findInstanceAndLocal(globalMeshletId, pc.instanceCount, instanceId, localMeshletId);

        visible = true; // Already culled in gpu_cull; pass through.
    }

    // Compact surviving meshlets into groupshared payload using wave ballot.
    uint laneCount = WaveActiveCountBits(visible);
    uint laneIndex = WavePrefixCountBits(visible);

    if (visible) {
        s_payload.meshlet_indices[laneIndex] = globalMeshletId;
        s_payload.instance_ids[laneIndex]    = instanceId;
    }

    // Dispatch mesh shader workgroups for each surviving meshlet.
    DispatchMesh(laneCount, 1, 1, s_payload);
}
