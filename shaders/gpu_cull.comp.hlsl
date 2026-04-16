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
// Two-pass CDLOD batching: (instanceOffset, instanceCount) scopes findInstanceAndLocal
// to a contiguous GpuInstance range; (meshletOffset, meshletCount) scopes the
// dispatch to the meshlet range produced by that instance set. For a one-pass
// full-scene cull pass (offset=0, count=total, meshletOffset=0, meshletCount=total).
struct PushBlock {
    uint instanceBufferSlot;   // StructuredBuffer<GpuInstance> (read as float4)
    uint meshletBufferSlot;    // StructuredBuffer<Meshlet>     (read as float4)
    uint commandsBufferSlot;   // RWByteAddressBuffer — DrawMeshTasksIndirectCommandEXT (3 uints)
    uint countBufferSlot;      // RWByteAddressBuffer — atomic counter at offset 0
    uint survivingIdsSlot;     // RWByteAddressBuffer — surviving global meshlet ids
    uint cameraSlot;           // StructuredBuffer<float4> camera data
    uint meshletCount;         // number of meshlets in THIS batch (dispatch range)
    uint instanceCount;        // number of GpuInstance entries in THIS batch
    uint instanceOffset;       // first GpuInstance index searched by findInstanceAndLocal
    uint meshletOffset;        // global meshlet index that threadID=0 maps to
};

[[vk::push_constant]] PushBlock pc;

// --- GPU struct accessors ---
// GpuInstance: 6 float4s (float4x4 transform = 4, then 2 float4s of packed fields).
// Layout (matches GpuSceneBuffer.h):
//   transform[0..3]
//   pack0 = {meshlet_offset, meshlet_count, material_index, vertex_buffer_slot}
//   pack1 = {vertex_base_offset, patch_quad_size (float), verts_per_edge, _pad}
// The cull pass only consumes the first two pack1 fields (the rest are terrain-specific).
struct GpuInstance {
    float4x4 transform;
    uint     meshlet_offset;
    uint     meshlet_count;
    uint     material_index;
    uint     vertex_buffer_slot;
    uint     vertex_base_offset;
};

GpuInstance loadInstance(uint idx) {
    StructuredBuffer<float4> buf = g_buffers[NonUniformResourceIndex(pc.instanceBufferSlot)];
    const uint base = idx * 6; // 4 float4 for mat4 + 2 float4 for packed fields = 6 float4
    GpuInstance inst;
    inst.transform = transpose(float4x4(buf[base + 0], buf[base + 1], buf[base + 2], buf[base + 3]));
    uint4 pack0    = asuint(buf[base + 4]);
    inst.meshlet_offset     = pack0.x;
    inst.meshlet_count      = pack0.y;
    inst.material_index     = pack0.z;
    inst.vertex_buffer_slot = pack0.w;
    uint4 pack1             = asuint(buf[base + 5]);
    inst.vertex_base_offset = pack1.x;
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

// (Gribb-Hartmann frustum extraction removed — see view-space test in CSMain.)

// --- Find which instance a global meshlet index belongs to ---
// Uses meshlet_offset + meshlet_count directly from each instance — correct
// regardless of whether the GpuInstance buffer is in the same order as the
// meshlet buffer (which is NOT guaranteed when GLTF nodes reference meshes
// out of their array order). An accumulated-offset walk would fail in that case.
void findInstanceAndLocal(uint globalMeshletIdx,
                          out uint instanceId,
                          out uint localMeshletIdx) {
    // Search only the (instanceOffset, instanceCount) batch passed for this
    // dispatch. For a single-pass cull over the whole scene, instanceOffset=0
    // and instanceCount=totalInstances — the original behaviour.
    for (uint k = 0; k < pc.instanceCount; ++k) {
        uint i = pc.instanceOffset + k;
        StructuredBuffer<float4> buf = g_buffers[NonUniformResourceIndex(pc.instanceBufferSlot)];
        uint4 packed          = asuint(buf[i * 6 + 4]);
        uint meshlet_offset   = packed.x;
        uint meshlet_count    = packed.y;
        if (globalMeshletIdx >= meshlet_offset && globalMeshletIdx < meshlet_offset + meshlet_count) {
            instanceId      = i;
            localMeshletIdx = globalMeshletIdx - meshlet_offset;
            return;
        }
    }
    // No instance owns this meshlet — it is an orphaned range (e.g. a terrain
    // patch retired-but-pending-reclamation whose GpuInstance is no longer in
    // the scene buffer). Signal "not found" with UINT32_MAX so CSMain can
    // discard the thread rather than falling back to instance 0 (the car).
    instanceId      = 0xFFFFFFFFu;
    localMeshletIdx = 0;
}

// --- Main ---
[numthreads(64, 1, 1)]
void CSMain(uint3 dispatchId : SV_DispatchThreadID) {
    // Shift threadID by meshletOffset so two-pass CDLOD batching can target a
    // sub-range of the global meshlet buffer (e.g. scene meshlets then terrain
    // meshlets). For a single-batch dispatch meshletOffset=0 — identity shift.
    uint globalId = dispatchId.x + pc.meshletOffset;
    if (dispatchId.x >= pc.meshletCount)
        return;

    // Determine which instance and local meshlet this global ID maps to.
    uint instanceId, localMeshletId;
    findInstanceAndLocal(globalId, instanceId, localMeshletId);

    // Orphaned meshlet (retired-but-pending-reclamation terrain range with no
    // current GpuInstance). Discard rather than falling back to instance 0.
    if (instanceId == 0xFFFFFFFFu)
        return;

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

    // Frustum cull — view-space approach.
    //
    // The previous Gribb-Hartmann approach extracted world-space planes by
    // indexing vp[col][row] into the transposed VP matrix.  Under DXC -Zpc
    // targeting SPIR-V the float4x4 constructor treats its arguments as
    // COLUMN vectors, so after transpose() the indexing gives VP_math[col][row]
    // instead of the required VP_math[row][col], silently producing wrong plane
    // normals that fail in a rotation-dependent way.
    //
    // This approach avoids that entirely:
    //   • mul(cam.view, ...) — same transpose-load pattern as the mesh shader,
    //     proven correct (objects render at right screen positions).
    //   • cam.proj[0][0] / cam.proj[1][1] — diagonal elements, identical under
    //     row-major or column-major constructor interpretation.
    //
    // DISABLE_FRUSTUM_CULL: set to 1 to bypass the test entirely.
    // DIAG_PER_PLANE_CULL:  set to 1 to count per-plane culls into
    //   countBuffer offsets 4,8,12,16,20.  Plane order: L, R, Bot, Top, Near.
#define DISABLE_FRUSTUM_CULL 0
#define DIAG_PER_PLANE_CULL  0

    CameraData cam = loadCamera(pc.cameraSlot);

#if DISABLE_FRUSTUM_CULL == 0
    {
        // Transform bounding sphere centre to view space.
        float4 viewPos4  = mul(cam.view, float4(worldCenter, 1.0));
        float  vx        = viewPos4.x;
        float  vy        = viewPos4.y;
        float  depth     = -viewPos4.z;  // positive = in front (right-handed -Z forward)

        // Frustum half-extents at this depth.
        // proj[0][0] = f/aspect,  proj[1][1] = -f  (Vulkan +Y-down flip applied in proj).
        float fPerAspect = cam.proj[0][0];           // > 0
        float hw         = depth / fPerAspect;        // half-width  of frustum at depth
        float hh         = depth / (-cam.proj[1][1]); // half-height of frustum at depth

        // Signed distances (positive = inside that half-space):
        //   [0] left    [1] right   [2] bottom   [3] top   [4] near / behind-camera
        float dists[5];
        dists[0] =  vx + hw;  // inside left  boundary: vx >= -hw
        dists[1] =  hw - vx;  // inside right boundary: vx <=  hw
        dists[2] =  vy + hh;  // inside bottom:         vy >= -hh
        dists[3] =  hh - vy;  // inside top:            vy <=  hh
        dists[4] =  depth;    // not behind camera:      depth >= 0

  #if DIAG_PER_PLANE_CULL
        {
            RWByteAddressBuffer diagBuf = g_rwBuffers[NonUniformResourceIndex(pc.countBufferSlot)];
            [unroll]
            for (int pi = 0; pi < 5; ++pi) {
                if (dists[pi] < -worldRadius) {
                    uint dummy;
                    diagBuf.InterlockedAdd((pi + 1) * 4, 1u, dummy); // offsets 4,8,12,16,20
                    return;
                }
            }
        }
  #else
        // Screen-edge planes (left/right/bottom/top): the frustum plane normals in this
        // linearised test are UNNORMALISED.  The left/right plane normal has length
        // sec(halfFovX) = sqrt(1 + tan²(halfFovX)), and the top/bottom plane normal has
        // length sec(halfFovY).  Comparing worldRadius against the raw `dists` values
        // effectively shrinks the sphere by cos(halfFov), causing over-aggressive culling
        // for meshlets near the screen edge (partially-visible quads get popped).
        //
        // Fix: scale worldRadius by the plane-normal length before the comparison so
        // the test is equivalent to a true signed-distance sphere-plane check.
        // For the near plane the normal IS (0,0,1) with unit length — no correction needed.
        float tanHFovX   = 1.0f / fPerAspect;                   // tan(halfFovX)
        float tanHFovY   = 1.0f / (-cam.proj[1][1]);            // tan(halfFovY)
        float worldRadiusLR = worldRadius * sqrt(1.0f + tanHFovX * tanHFovX); // worldRadius * sec(halfFovX)
        float worldRadiusTB = worldRadius * sqrt(1.0f + tanHFovY * tanHFovY); // worldRadius * sec(halfFovY)

        // 1m world-space guard band on top of the corrected radius so meshlets that
        // graze the screen edge don't flicker due to sub-texel sphere inaccuracies.
        static const float kEdgeGuard = 1.0f;
        if (dists[0] < -(worldRadiusLR + kEdgeGuard)) return;  // left
        if (dists[1] < -(worldRadiusLR + kEdgeGuard)) return;  // right
        if (dists[2] < -(worldRadiusTB + kEdgeGuard)) return;  // bottom
        if (dists[3] < -(worldRadiusTB + kEdgeGuard)) return;  // top
        // Near / behind-camera plane: no guard band — discard anything behind camera.
        if (dists[4] < -worldRadius)
            return;
  #endif
    }
#endif

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
