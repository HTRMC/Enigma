// terrain_cdlod.mesh.hlsl
// ========================
// Mesh shader for the CDLOD terrain visibility-buffer pipeline. One workgroup
// per surviving terrain meshlet. Reads vertex positions from a per-LOD vertex
// pool (shared across all patches of a given LOD) at offset
// `inst.vertex_base_offset + localVertInPatch`, and reads meshlet topology
// (vertex-index list + packed-u8 triangle-index list) from per-LOD
// shared-topology SSBOs indexed by push-constant slots.
//
// Per-LOD vertex pool layout (one FLOAT — Y only — per vertex):
//   heightIdx = inst.vertex_base_offset + localVertIdx   (float index, stride 4B)
//   Read by bindless StructuredBuffer<float4> as:
//     g_buffers[inst.vertex_buffer_slot][heightIdx / 4][heightIdx & 3]
//   i.e. packed 4 heights per float4 element. World X and Z are reconstructed
//   from localVertIdx + inst.verts_per_edge + inst.patch_quad_size (see below).
//
// Shared topology layout (same byte encoding as the regular
// GpuMeshletBuffer::m_vertices / m_triangles):
//   topologyVerticesSlot  : flat u32 vertex-index array, packed as float4
//                           (4 u32 per float4 element). Indices address the
//                           CURRENT patch's vertex block — they are local to
//                           the patch, not global — hence the per-instance
//                           vertex_base_offset is added at fetch time.
//   topologyTrianglesSlot : packed 3-byte triangles in a ByteAddressBuffer.
//
// vis_value encoding is IDENTICAL to visibility_buffer.mesh.hlsl:
//   [31:7] globalMeshletId (25 bits)
//   [6:0]  localTriId      (7 bits, MAX_TRIANGLES=124 fits)
// MaterialEvalPass decodes terrain pixels via the same shift/mask used for
// regular meshes — the terrain branch is taken via MATERIAL_FLAG_TERRAIN on
// the resolved material, not via any special vis_value bit.
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

// --- Push constants (unified with terrain_cdlod.task.hlsl) ---
// Base fields mirror visibility_buffer.mesh.hlsl so the host can reuse the
// same push-constant emission helpers; the two topology slots at the tail
// are the terrain-specific extension.
struct PushBlock {
    uint instanceBufferSlot;     // both task+mesh: GpuInstance[]
    uint meshletBufferSlot;      // both task+mesh: Meshlet[]
    uint survivingIdsSlot;       // task only (unused in mesh stage)
    uint meshletVerticesSlot;    // mesh: unused for terrain (topology comes from topologyVerticesSlot)
    uint meshletTrianglesSlot;   // mesh: unused for terrain (topology comes from topologyTrianglesSlot)
    uint cameraSlot;             // both: camera matrices
    uint countBufferSlot;        // task only (unused in mesh stage)
    uint instanceCount;          // task only (unused in mesh stage)
    uint topologyVerticesSlot;   // mesh: shared per-LOD meshlet_vertices[] SSBO
    uint topologyTrianglesSlot;  // mesh: shared per-LOD meshlet_triangles packed-u8 SSBO
};

[[vk::push_constant]] PushBlock pc;

// --- GPU struct accessors ---
// GpuInstance: 6 float4s — layout identical to visibility_buffer.mesh.hlsl.
// For terrain:
//   vertex_buffer_slot  -> per-LOD pool SSBO
//   vertex_base_offset  -> float index (stride 4B) of this patch's Y-values
//   patch_quad_size     -> world-space quad size (patch_size / quadsPerPatch)
//   verts_per_edge      -> vertices per patch edge (quadsPerPatch + 1)
struct GpuInstance {
    float4x4 transform;
    uint     meshlet_offset;
    uint     meshlet_count;
    uint     material_index;
    uint     vertex_buffer_slot;
    uint     vertex_base_offset;
    float    patch_quad_size;
    uint     verts_per_edge;
    uint     _pad;
};

GpuInstance loadInstance(uint idx) {
    StructuredBuffer<float4> buf = g_buffers[NonUniformResourceIndex(pc.instanceBufferSlot)];
    const uint base = idx * 6;
    GpuInstance inst;
    // DXC -Zpc -spirv: float4x4(v0..v3) takes COLUMNS. transpose() compensates
    // so `inst.transform[row][col]` matches the row-major HLSL convention the
    // rest of the pipeline uses (matches visibility_buffer.mesh.hlsl).
    inst.transform = transpose(float4x4(buf[base + 0], buf[base + 1], buf[base + 2], buf[base + 3]));
    uint4 pack0    = asuint(buf[base + 4]);
    inst.meshlet_offset     = pack0.x;
    inst.meshlet_count      = pack0.y;
    inst.material_index     = pack0.z;
    inst.vertex_buffer_slot = pack0.w;
    float4 pack1f           = buf[base + 5];
    uint4  pack1u           = asuint(pack1f);
    inst.vertex_base_offset = pack1u.x;
    inst.patch_quad_size    = pack1f.y;
    inst.verts_per_edge     = pack1u.z;
    inst._pad               = pack1u.w;
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

// --- Mesh payload (shared with terrain_cdlod.task.hlsl) ---
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
// Same pattern as visibility_buffer.mesh.hlsl: localTriId is a per-primitive
// attribute (declared nointerpolation so DXC emits PerPrimitiveEXT in SPIR-V).
// Never use SV_PrimitiveID in the pixel shader — on NVIDIA EXT_mesh_shader
// hardware it is the global draw-call counter, not the per-meshlet local index.
struct PrimAttrib {
    nointerpolation uint localTriId : TEXCOORD1;
};

// --- Mesh shader main ---
// numthreads = MAX_TRIANGLES (124) — matches visibility_buffer.mesh.hlsl.
// This is deliberately MAX_TRIANGLES, not MAX_VERTICES: a previous regression
// (MAX_VERTICES=64 < triangle_count possible on certain meshlets) left the
// per-triangle LDS slots beyond thread 63 uninitialized and produced confetti.
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

    // --- Vertex processing ---
    // Each thread processes one vertex (threads beyond vertex_count are idle).
    if (groupIndex < meshlet.vertex_count) {
        // Shared-topology meshlet_vertices is a flat u32 array packed as
        // StructuredBuffer<float4>.  Each float4 holds 4 u32 patch-local
        // vertex indices. Compute the float4 element and component.
        StructuredBuffer<float4> topoVertBuf =
            g_buffers[NonUniformResourceIndex(pc.topologyVerticesSlot)];
        uint flatIdx    = meshlet.vertex_offset + groupIndex;
        uint float4Idx  = flatIdx / 4;
        uint component  = flatIdx % 4;
        uint4 packed    = asuint(topoVertBuf[float4Idx]);
        uint localVertInPatch;
        if      (component == 0) localVertInPatch = packed.x;
        else if (component == 1) localVertInPatch = packed.y;
        else if (component == 2) localVertInPatch = packed.z;
        else                     localVertInPatch = packed.w;

        // Fetch Y (height) from this patch's slot inside the per-LOD vertex pool.
        // Pool layout: one FLOAT per vertex. Read through the bindless
        // StructuredBuffer<float4> array by indexing the float4 element and
        // picking the appropriate component. X and Z are reconstructed below
        // from the patch-local vertex index — they were NOT stored in the pool.
        StructuredBuffer<float4> vbuf =
            g_buffers[NonUniformResourceIndex(inst.vertex_buffer_slot)];
        uint heightIdx   = inst.vertex_base_offset + localVertInPatch;
        uint heightF4Idx = heightIdx / 4u;
        uint heightComp  = heightIdx & 3u;
        float4 pack      = vbuf[heightF4Idx];
        float worldY;
        if      (heightComp == 0) worldY = pack.x;
        else if (heightComp == 1) worldY = pack.y;
        else if (heightComp == 2) worldY = pack.z;
        else                      worldY = pack.w;

        // Reconstruct patch-local X and Z from the grid-layout vertex index.
        // Template grid is row-major: localVertInPatch = z * verts_per_edge + x.
        uint vpe    = inst.verts_per_edge;
        uint gridX  = localVertInPatch % vpe;
        uint gridZ  = localVertInPatch / vpe;
        float localX = float(gridX) * inst.patch_quad_size;
        float localZ = float(gridZ) * inst.patch_quad_size;
        float3 position = float3(localX, worldY, localZ);

        // Transform to clip space. inst.transform is the patch's world
        // matrix (typically a translation to the patch's worldMin corner;
        // the local-space X/Z above are relative to that corner).
        // Final clip = viewProj * world * local.
        float4 worldPos = mul(inst.transform, float4(position, 1.0));
        float4 clipPos  = mul(cam.viewProj, worldPos);

        VertexOutput v;
        v.pos       = clipPos;
        v.vis_value = globalMeshletId << 7u;
        verts[groupIndex] = v;
    }

    // --- Triangle processing ---
    // Each thread processes one triangle (threads beyond triangle_count are idle).
    if (groupIndex < meshlet.triangle_count) {
        // Shared-topology triangles are packed as 3 u8 indices per triangle
        // in a ByteAddressBuffer (same encoding as GpuMeshletBuffer::m_triangles).
        RWByteAddressBuffer triBuf =
            g_rwBuffers[NonUniformResourceIndex(pc.topologyTrianglesSlot)];
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
        // See visibility_buffer.mesh.hlsl for the NVIDIA SV_PrimitiveID caveat
        // — the same bug applies identically to terrain meshlets.
        prims[groupIndex].localTriId = groupIndex;
    }
}

// --- Pixel shader ---
// Writes packed visibility value to R32_UINT render target.
// Encoding: [31:7] globalMeshletId | [6:0] localTriId
// Identical to visibility_buffer.mesh.hlsl — MaterialEvalPass decodes both
// terrain and regular pixels through the same unpack path.
uint PSMain(VertexOutput input, PrimAttrib primInput) : SV_Target0 {
    return input.vis_value | (primInput.localTriId & 0x7Fu);
}
