// gpu_cull.comp.hlsl
// ==================
// Compute shader for GPU-driven meshlet frustum culling. One thread per
// meshlet globally. Surviving meshlets emit an indirect draw command
// (VkDrawMeshTasksIndirectCommandEXT = 3 uints) and store their global
// meshlet index for the mesh shader to look up via SV_GroupID.
//
// Dispatch: ceil(totalMeshlets / 64) workgroups of 64 threads each.
// Depth convention: reverse-Z (far = 0, near = 1).

#include "common.hlsl"

// --- Bindless resource arrays ---
[[vk::binding(2, 0)]]
StructuredBuffer<float4> g_buffers[] : register(t0, space1);

[[vk::binding(5, 0)]]
RWByteAddressBuffer g_rwBuffers[] : register(u1, space0);

// --- Push constants ---
struct PushBlock {
    uint instanceBufferSlot;   // StructuredBuffer<GpuInstance> (read as float4)
    uint meshletBufferSlot;    // StructuredBuffer<Meshlet>     (read as float4)
    uint commandsBufferSlot;   // RWByteAddressBuffer — DrawMeshTasksIndirectCommandEXT (3 uints)
    uint countBufferSlot;      // RWByteAddressBuffer — atomic counter at offset 0
    uint survivingIdsSlot;     // RWByteAddressBuffer — surviving global meshlet ids
    uint cameraSlot;           // StructuredBuffer<float4> camera data
    uint totalMeshlets;        // total meshlet count across all instances
    uint instanceCount;        // number of GpuInstance entries
};

[[vk::push_constant]] PushBlock pc;

// --- GPU struct accessors ---
// GpuInstance: 18 float4s (float4x4 transform = 4, then 4 uints packed as 1 float4)
// Layout: transform[0..3], {meshlet_offset, meshlet_count, material_index, vertex_buffer_slot}
struct GpuInstance {
    float4x4 transform;
    uint     meshlet_offset;
    uint     meshlet_count;
    uint     material_index;
    uint     vertex_buffer_slot;
};

GpuInstance loadInstance(uint idx) {
    StructuredBuffer<float4> buf = g_buffers[NonUniformResourceIndex(pc.instanceBufferSlot)];
    const uint base = idx * 5; // 4 float4 for mat4 + 1 float4 for 4 uints = 5 float4
    GpuInstance inst;
    inst.transform = transpose(float4x4(buf[base + 0], buf[base + 1], buf[base + 2], buf[base + 3]));
    uint4 packed   = asuint(buf[base + 4]);
    inst.meshlet_offset    = packed.x;
    inst.meshlet_count     = packed.y;
    inst.material_index    = packed.z;
    inst.vertex_buffer_slot = packed.w;
    return inst;
}

// Meshlet: 3 float4s (12 floats / uints)
// Layout: {vertex_offset, triangle_offset, vertex_count, triangle_count},
//         {bounding_sphere_center.xyz, bounding_sphere_radius},
//         {cone_axis.xyz, cone_cutoff}
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

// --- Camera load (column-major GLM -> HLSL row-major) ---
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

// --- Frustum plane extraction from viewProj (Gribb-Hartmann method) ---
// Each plane stored as float4(A, B, C, D) where Ax+By+Cz+D >= 0 means inside.
struct Frustum {
    float4 planes[6];
};

Frustum extractFrustum(float4x4 vp) {
    // Gribb-Hartmann frustum extraction.
    // DXC compiles with -Zpc (column-major), so vp[col][row] — i.e. vp[i][j] = VP_{row=j, col=i}.
    // clip = vp * world  →  clip.x = row0·world, clip.w = row3·world.
    // Left plane:   clip.w + clip.x ≥ 0  →  (row3 + row0)·world ≥ 0
    //   A = VP_{3,0} + VP_{0,0} = vp[0][3] + vp[0][0], etc.
    Frustum f;
    f.planes[0] = float4(vp[0][3]+vp[0][0], vp[1][3]+vp[1][0], vp[2][3]+vp[2][0], vp[3][3]+vp[3][0]); // Left
    f.planes[1] = float4(vp[0][3]-vp[0][0], vp[1][3]-vp[1][0], vp[2][3]-vp[2][0], vp[3][3]-vp[3][0]); // Right
    f.planes[2] = float4(vp[0][3]+vp[0][1], vp[1][3]+vp[1][1], vp[2][3]+vp[2][1], vp[3][3]+vp[3][1]); // Bottom
    f.planes[3] = float4(vp[0][3]-vp[0][1], vp[1][3]-vp[1][1], vp[2][3]-vp[2][1], vp[3][3]-vp[3][1]); // Top
    f.planes[4] = float4(vp[0][3]-vp[0][2], vp[1][3]-vp[1][2], vp[2][3]-vp[2][2], vp[3][3]-vp[3][2]); // Near
    f.planes[5] = float4(vp[0][2],           vp[1][2],           vp[2][2],           vp[3][2]);          // Far

    // Skip plane[5] (far) — for reverse-Z infinite projection, row2 of VP is
    // (0, 0, 0, near), so planes[5].xyz = 0 and normalization would divide by
    // zero.  Nothing is ever beyond an infinite far plane, so the test always
    // passes and we simply omit it.
    [unroll]
    for (int i = 0; i < 5; ++i) {
        float len = length(f.planes[i].xyz);
        f.planes[i] /= len;
    }
    return f;
}

// Sphere-vs-frustum test. Returns true if the sphere is at least partially inside.
bool sphereInsideFrustum(Frustum frust, float3 center, float radius) {
    [unroll]
    for (int i = 0; i < 5; ++i) { // plane[5] (far) skipped — infinite projection, always passes
        float dist = dot(frust.planes[i].xyz, center) + frust.planes[i].w;
        if (dist < -radius)
            return false;
    }
    return true;
}

// --- Find which instance a global meshlet index belongs to ---
// Uses meshlet_offset + meshlet_count directly from each instance — correct
// regardless of whether the GpuInstance buffer is in the same order as the
// meshlet buffer (which is NOT guaranteed when GLTF nodes reference meshes
// out of their array order). An accumulated-offset walk would fail in that case.
void findInstanceAndLocal(uint globalMeshletIdx,
                          out uint instanceId,
                          out uint localMeshletIdx) {
    for (uint i = 0; i < pc.instanceCount; ++i) {
        StructuredBuffer<float4> buf = g_buffers[NonUniformResourceIndex(pc.instanceBufferSlot)];
        uint4 packed          = asuint(buf[i * 5 + 4]);
        uint meshlet_offset   = packed.x;
        uint meshlet_count    = packed.y;
        if (globalMeshletIdx >= meshlet_offset && globalMeshletIdx < meshlet_offset + meshlet_count) {
            instanceId      = i;
            localMeshletIdx = globalMeshletIdx - meshlet_offset;
            return;
        }
    }
    // Should not reach here if totalMeshlets is correct.
    instanceId      = 0;
    localMeshletIdx = 0;
}

// --- Main ---
[numthreads(64, 1, 1)]
void CSMain(uint3 dispatchId : SV_DispatchThreadID) {
    uint globalId = dispatchId.x;
    if (globalId >= pc.totalMeshlets)
        return;

    // Determine which instance and local meshlet this global ID maps to.
    uint instanceId, localMeshletId;
    findInstanceAndLocal(globalId, instanceId, localMeshletId);

    GpuInstance inst = loadInstance(instanceId);
    Meshlet meshlet  = loadMeshlet(inst.meshlet_offset + localMeshletId);

    // Transform meshlet bounding sphere centre to world space.
    float3 worldCenter = mul(inst.transform, float4(meshlet.center, 1.0)).xyz;

    // Scale radius by the maximum axis scale of the world matrix.
    // With -Zpc (column-major), transform[col][row], so column vectors are:
    //   col0 = (transform[0][0], transform[0][1], transform[0][2]), etc.
    // Column lengths give the actual axis scales; row lengths do not.
    float3 col0 = float3(inst.transform[0][0], inst.transform[0][1], inst.transform[0][2]);
    float3 col1 = float3(inst.transform[1][0], inst.transform[1][1], inst.transform[1][2]);
    float3 col2 = float3(inst.transform[2][0], inst.transform[2][1], inst.transform[2][2]);
    float maxScale = max(length(col0), max(length(col1), length(col2)));
    float worldRadius = meshlet.radius * maxScale;

    // Frustum cull.
    CameraData cam = loadCamera(pc.cameraSlot);
    Frustum frustum = extractFrustum(cam.viewProj);

    if (!sphereInsideFrustum(frustum, worldCenter, worldRadius))
        return;

    // Meshlet survived — append to output.
    RWByteAddressBuffer countBuf      = g_rwBuffers[NonUniformResourceIndex(pc.countBufferSlot)];
    RWByteAddressBuffer commandsBuf   = g_rwBuffers[NonUniformResourceIndex(pc.commandsBufferSlot)];
    RWByteAddressBuffer survivingBuf  = g_rwBuffers[NonUniformResourceIndex(pc.survivingIdsSlot)];

    uint prevCount;
    countBuf.InterlockedAdd(0, 1u, prevCount);

    // Write VkDrawMeshTasksIndirectCommandEXT: { groupCountX=1, groupCountY=1, groupCountZ=1 }
    commandsBuf.Store3(prevCount * 12, uint3(1, 1, 1));

    // Store (globalMeshletId, instanceId) as a uint2 so the task shader can use both
    // without re-running findInstanceAndLocal (which fails for shared primitives).
    survivingBuf.Store2(prevCount * 8, uint2(globalId, instanceId));
}
