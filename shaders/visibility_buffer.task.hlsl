// visibility_buffer.task.hlsl
// ===========================
// Amplification (task) shader for the visibility buffer pipeline. Each
// workgroup processes TASK_GROUP_SIZE entries from the surviving meshlet IDs
// buffer written by the GPU cull compute pass and dispatches surviving meshlets
// to the mesh shader stage.
//
// Dispatch: ceil(totalMeshlets / TASK_GROUP_SIZE) workgroups.
// The GPU count buffer (countBufferSlot) holds the actual surviving count;
// threads whose slot index >= actualSurviving output nothing.

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
    uint survivingIdsSlot;      // task: surviving (globalMeshletId, instanceId) pairs from gpu_cull
    uint meshletVerticesSlot;   // mesh: vertex index remapping (u32[])
    uint meshletTrianglesSlot;  // mesh: packed u8 triangle indices
    uint cameraSlot;            // both: camera matrices
    uint countBufferSlot;       // task: GPU buffer slot holding actual surviving count
    uint instanceCount;         // task: number of GpuInstance entries (unused — instanceId from buffer)
};

[[vk::push_constant]] PushBlock pc;

// --- Mesh payload (shared with visibility_buffer.mesh.hlsl) ---
struct MeshPayload {
    uint meshlet_indices[TASK_GROUP_SIZE]; // global meshlet IDs
    uint instance_ids[TASK_GROUP_SIZE];    // corresponding instance IDs
};

groupshared MeshPayload s_payload;

// --- Main ---
// Each task group processes up to TASK_GROUP_SIZE entries from the surviving
// IDs buffer.  Wave ballot compacts valid entries into the payload and
// dispatches one mesh group per surviving meshlet in this batch.
[numthreads(TASK_GROUP_SIZE, 1, 1)]
void ASMain(
    in uint3 groupId    : SV_GroupID,
    in uint  groupIndex : SV_GroupIndex
) {
    // Read actual surviving count from the GPU count buffer.
    RWByteAddressBuffer countBuf = g_rwBuffers[NonUniformResourceIndex(pc.countBufferSlot)];
    uint actualSurviving = countBuf.Load(0);

    // Each thread in this group handles one slot in the surviving IDs buffer.
    uint batchBase   = groupId.x * TASK_GROUP_SIZE;
    uint meshletSlot = batchBase + groupIndex;

    // Pre-initialize payload to sentinel so stale LDS from a prior wave never
    // leaks into the mesh shader if DispatchMesh(0,...) misbehaves on a driver.
    s_payload.meshlet_indices[groupIndex] = 0xFFFFFFFFu;
    s_payload.instance_ids[groupIndex]    = 0u;
    GroupMemoryBarrierWithGroupSync();

    bool visible = false;
    uint globalMeshletId = 0;
    uint instanceId      = 0;

    if (meshletSlot < actualSurviving) {
        RWByteAddressBuffer survivingBuf = g_rwBuffers[NonUniformResourceIndex(pc.survivingIdsSlot)];
        // Read (globalMeshletId, instanceId) packed by gpu_cull — stride 8 bytes.
        uint2 entry     = survivingBuf.Load2(meshletSlot * 8);
        globalMeshletId = entry.x;
        instanceId      = entry.y;
        visible = true;
    }

    // Compact surviving meshlets into groupshared payload using wave ballot.
    uint laneCount = WaveActiveCountBits(visible);
    uint laneIndex = WavePrefixCountBits(visible);

    if (visible) {
        s_payload.meshlet_indices[laneIndex] = globalMeshletId;
        s_payload.instance_ids[laneIndex]    = instanceId;
    }

    // Ensure all payload writes are visible before mesh shader reads them.
    // Required by the EXT_mesh_shader spec: "The last write to any component
    // of the task payload variable must be completed before the call to
    // OpEmitMeshTasksEXT." On NVIDIA warp=32=TASK_GROUP_SIZE this is a no-op
    // but is required for correctness on other hardware and future drivers.
    GroupMemoryBarrierWithGroupSync();

    // All invocations MUST call DispatchMesh exactly once (EXT_mesh_shader spec).
    // Calling it conditionally (only when laneCount > 0) is undefined behavior.
    // When laneCount == 0, DispatchMesh(0,...) dispatches no mesh groups on a
    // conformant driver.  Stale-LDS safety is handled by the sentinel init above
    // and the 0xFFFFFFFF guard in the mesh shader.
    DispatchMesh(laneCount, 1, 1, s_payload);
}
