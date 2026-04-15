// visibility_buffer.mesh.hlsl
// ============================
// Mesh shader for the visibility buffer pipeline. One workgroup per surviving
// meshlet. Reads vertex positions, transforms to clip space, and outputs
// triangles. The pixel shader writes a packed instance+triangle ID to the
// R32_UINT visibility buffer.
//
// Vertex data layout: 3 float4 per vertex (same as mesh.hlsl):
//   [vid*3+0] = (position.x, position.y, position.z, normal.x)
//   [vid*3+1] = (normal.y,   normal.z,   uv.x,       uv.y)
//   [vid*3+2] = (tangent.x,  tangent.y,  tangent.z,  tangent.w)
//
// Depth convention: reverse-Z (far = 0, near = 1).

#include "common.hlsl"

#define TASK_GROUP_SIZE 32
#define MAX_VERTICES    64
#define MAX_TRIANGLES   124

// --- Bindless resource arrays ---
[[vk::binding(2, 0)]]
StructuredBuffer<float4> g_buffers[] : register(t0, space1);

[[vk::binding(5, 0)]]
RWByteAddressBuffer g_rwBuffers[] : register(u1, space0);

// --- Push constants (unified with visibility_buffer.task.hlsl) ---
struct PushBlock {
    uint instanceBufferSlot;    // both task+mesh: GpuInstance[]
    uint meshletBufferSlot;     // both task+mesh: Meshlet[]
    uint survivingIdsSlot;      // task only (unused in mesh stage)
    uint meshletVerticesSlot;   // mesh: vertex index remapping (u32[])
    uint meshletTrianglesSlot;  // mesh: packed u8 triangle indices
    uint cameraSlot;            // both: camera matrices
    uint countBufferSlot;       // task only (unused in mesh stage)
    uint instanceCount;         // task only (unused in mesh stage)
};

[[vk::push_constant]] PushBlock pc;

// --- GPU struct accessors ---
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

// --- Mesh payload (shared with visibility_buffer.task.hlsl) ---
struct MeshPayload {
    uint meshlet_indices[TASK_GROUP_SIZE]; // global meshlet IDs
    uint instance_ids[TASK_GROUP_SIZE];    // corresponding instance IDs
};

// --- Vertex output ---
struct VertexOutput {
    float4 pos : SV_Position;
    nointerpolation uint vis_value : TEXCOORD0; // globalMeshletId << 7 (lower 7 bits zeroed)
};

// --- Per-primitive output ---
// localTriId is the per-meshlet triangle index [0, triangle_count).
// Declared nointerpolation so DXC emits PerPrimitiveEXT in SPIR-V.
// SV_PrimitiveID in the pixel shader would be the global draw-call counter on
// NVIDIA hardware — using an explicit per-primitive attribute avoids that bug.
struct PrimAttrib {
    nointerpolation uint localTriId : TEXCOORD1;
};

// --- Mesh shader main ---
[numthreads(MAX_TRIANGLES, 1, 1)]
[outputtopology("triangle")]
void MSMain(
    in uint3 groupId    : SV_GroupID,
    in uint  groupIndex : SV_GroupIndex,
    in payload MeshPayload p,
    out vertices  VertexOutput verts[MAX_VERTICES],
    out indices   uint3        tris[MAX_TRIANGLES],
    out primitives PrimAttrib  prims[MAX_TRIANGLES]
) {
    uint globalMeshletId = p.meshlet_indices[groupId.x];
    uint instanceId      = p.instance_ids[groupId.x];

    // Sentinel check: if the task shader wrote 0xFFFFFFFF (uninitialized slot),
    // output nothing rather than rendering garbage geometry.
    if (globalMeshletId == 0xFFFFFFFFu) {
        SetMeshOutputCounts(0, 0);
        return;
    }

    GpuInstance inst = loadInstance(instanceId);
    Meshlet meshlet  = loadMeshlet(globalMeshletId);

    CameraData cam = loadCamera(pc.cameraSlot);

    SetMeshOutputCounts(meshlet.vertex_count, meshlet.triangle_count);

    // Each thread processes one vertex (threads beyond vertex_count are idle).
    if (groupIndex < meshlet.vertex_count) {
        // meshlet_vertices is a flat u32 array packed as StructuredBuffer<float4>.
        // Each float4 holds 4 u32 vertex indices. Compute the float4 element and component.
        StructuredBuffer<float4> meshletVertBuf = g_buffers[NonUniformResourceIndex(pc.meshletVerticesSlot)];
        uint flatIdx    = meshlet.vertex_offset + groupIndex;
        uint float4Idx  = flatIdx / 4;
        uint component  = flatIdx % 4;
        uint4 packed    = asuint(meshletVertBuf[float4Idx]);
        uint globalVertIdx;
        if      (component == 0) globalVertIdx = packed.x;
        else if (component == 1) globalVertIdx = packed.y;
        else if (component == 2) globalVertIdx = packed.z;
        else                     globalVertIdx = packed.w;

        // Load vertex position from the instance's vertex SSBO.
        StructuredBuffer<float4> vbuf = g_buffers[NonUniformResourceIndex(inst.vertex_buffer_slot)];
        float3 position = vbuf[globalVertIdx * 3 + 0].xyz;

        // Transform to clip space.
        float4 worldPos = mul(inst.transform, float4(position, 1.0));
        float4 clipPos  = mul(cam.viewProj, worldPos);

        // Vis buffer encoding (32 bits):
        //   [31:7] globalMeshletId (25 bits, up to 33M total meshlets)
        //   [6:0]  localPrimId     (7 bits, MAX_TRIANGLES=124 fits)
        // Instance is resolved in material_eval by range-checking meshlet_offset.
        VertexOutput v;
        v.pos       = clipPos;
        v.vis_value = globalMeshletId << 7u;
        verts[groupIndex] = v;
    }

    // Each thread processes one triangle (threads beyond triangle_count are idle).
    if (groupIndex < meshlet.triangle_count) {
        // Meshlet triangles are packed as 3 u8 indices per triangle in a ByteAddressBuffer.
        // Each triangle = 3 bytes. We load a uint and extract bytes.
        RWByteAddressBuffer triBuf = g_rwBuffers[NonUniformResourceIndex(pc.meshletTrianglesSlot)];
        uint byteOffset = meshlet.triangle_offset + groupIndex * 3;
        uint wordOffset = byteOffset & ~3u; // align to 4 bytes
        uint shift      = (byteOffset & 3u) * 8;

        uint word0 = triBuf.Load(wordOffset);
        uint word1 = triBuf.Load(wordOffset + 4);

        // Extract 3 consecutive bytes starting at byteOffset.
        // Guard shift == 0: HLSL masks shift amounts mod 32, so
        // word1 << (32 - 0) becomes word1 << 0 = word1 (not 0).
        uint bits;
        if (shift == 0) {
            bits = word0;
        } else {
            bits = (word0 >> shift) | (word1 << (32 - shift));
        }
        uint i0 = bits & 0xFF;
        uint i1 = (bits >> 8) & 0xFF;
        uint i2 = (bits >> 16) & 0xFF;

        tris[groupIndex] = uint3(i0, i1, i2);

        // Write the per-meshlet local triangle index as a per-primitive attribute.
        // The pixel shader reads this instead of SV_PrimitiveID (which on NVIDIA
        // hardware is the global draw-call primitive counter, not the per-meshlet
        // local index, causing vis buffer corruption for all meshlets after the first).
        prims[groupIndex].localTriId = groupIndex;
    }
}

// --- Pixel shader ---
// Writes packed visibility value to R32_UINT render target.
// Encoding: [31:7] globalMeshletId | [6:0] localPrimId
//
// localTriId comes from the per-primitive attribute written by MSMain.
// Do NOT use SV_PrimitiveID here: on NVIDIA EXT_mesh_shader hardware it is
// the global draw-call primitive counter (accumulates across meshlets), so
// `& 0x7F` wraps incorrectly for every meshlet after the first.
uint PSMain(VertexOutput input, PrimAttrib primInput) : SV_Target0 {
    return input.vis_value | (primInput.localTriId & 0x7Fu);
}
