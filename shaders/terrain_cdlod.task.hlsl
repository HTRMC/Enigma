// terrain_cdlod.task.hlsl
// =======================
// Amplification (task) shader for the CDLOD terrain visibility-buffer pipeline.
// Structurally identical to visibility_buffer.task.hlsl — each workgroup
// processes TASK_GROUP_SIZE entries from the surviving-meshlet-ID buffer
// written by GpuCullPass and dispatches surviving meshlets to the terrain
// mesh shader stage.
//
// Terrain patches flow through the SAME GpuCullPass as regular meshes; the
// cull compute writes (globalMeshletId, instanceId) pairs the same way, so
// this task shader shares the cull output format with visibility_buffer.task.hlsl.
// The only reason a separate task shader exists is to pair with the
// terrain mesh shader (which uses different push constants for the
// shared-topology SSBO slots).
//
// Dispatch: ceil(totalMeshlets / TASK_GROUP_SIZE) workgroups.

#include "common.hlsl"

#define TASK_GROUP_SIZE 32

// --- Bindless resource arrays ---
[[vk::binding(2, 0)]]
StructuredBuffer<float4> g_buffers[] : register(t0, space1);

[[vk::binding(5, 0)]]
RWByteAddressBuffer g_rwBuffers[] : register(u1, space0);

// --- Push constants (unified with terrain_cdlod.mesh.hlsl) ---
// Mirrors visibility_buffer.task/mesh PushBlock, then appends the two
// terrain-specific topology slots at the end. The mesh shader consumes
// topologyVerticesSlot/topologyTrianglesSlot; the task shader only needs
// the base cull-output fields.
struct PushBlock {
    uint instanceBufferSlot;     // both task+mesh: GpuInstance[]
    uint meshletBufferSlot;      // both task+mesh: Meshlet[]
    uint survivingIdsSlot;       // task: surviving (globalMeshletId, instanceId) pairs from gpu_cull
    uint meshletVerticesSlot;    // mesh: unused for terrain (topology comes from topologyVerticesSlot)
    uint meshletTrianglesSlot;   // mesh: unused for terrain (topology comes from topologyTrianglesSlot)
    uint cameraSlot;             // both: camera matrices
    uint countBufferSlot;        // task: GPU buffer slot holding actual surviving count
    uint instanceCount;          // task: number of GpuInstance entries (unused — instanceId read from surviving buf)
    uint topologyVerticesSlot;   // mesh: shared per-LOD meshlet_vertices[] SSBO
    uint topologyTrianglesSlot;  // mesh: shared per-LOD meshlet_triangles packed-u8 SSBO
};

[[vk::push_constant]] PushBlock pc;

// --- Mesh payload (shared with terrain_cdlod.mesh.hlsl) ---
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
