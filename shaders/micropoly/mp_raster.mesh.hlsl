// mp_raster.mesh.hlsl
// =====================
// Mesh shader for the Micropoly HW raster pipeline (M3.3). Consumes the task
// payload written by mp_raster.task.hlsl, reads a cluster's vertices and
// triangles from the PageCache, transforms vertices to clip space, and emits
// triangles. The fragment shader at the bottom of this file does the 64-bit
// atomic-min write to the visibility image.
//
// Rasterisation model: HW raster (standard vkCmdDrawMeshTasksIndirectEXT
// path). No color attachment; the fragment stage's side-effect is an
// InterlockedMax on the R64_UINT storage image registered at
// visImageBindlessIndex. Under reverse-Z (see
// shaders/micropoly/mp_vis_pack.hlsl banner) the LARGEST packed
// (depth, clusterId, triId) tuple wins — that is the nearest sample.
//
// DXC -Zpc -spirv: float4x4(row0..row3) takes COLUMN vectors, so every
// SSBO matrix load uses transpose(float4x4(...)). Grep the rest of the
// engine's shaders/ tree for the identical pattern.

#include "../common.hlsl"
#include "mp_vis_pack.hlsl"
#include "mp_cluster_layout.hlsl"

// Must match MP_MESH_GROUP_TRIS in the task shader.
#define MP_MAX_TRIANGLES 128u
// Each ClusterOnDisk can carry up to 128 vertices (bake-time cap — see
// MpAssetFormat.h). Sized so triangles with all-unique indices fit.
#define MP_MAX_VERTICES  128u

// --- Bindless resource arrays ----------------------------------------------
[[vk::binding(2, 0)]]
StructuredBuffer<float4> g_buffers[] : register(t0, space1);

[[vk::binding(5, 0)]]
RWByteAddressBuffer g_rwBuffers[] : register(u1, space0);

// R64_UINT storage image array for atomic-min writes. DXC -spirv maps
// RWTexture2D<uint64_t> to OpTypeImage Unknown=R64ui with AtomicInt64Image
// support. When the device only advertises R32G32_UINT atomics we'd need a
// CAS-based fallback — that path is stubbed as an error log at pass
// construction time (see MicropolyRasterPass::create; plan §M4).
[[vk::binding(1, 0)]]
RWTexture2D<uint64_t> g_visImages[] : register(u0, space0);

// --- Push constants (must match mp_raster.task.hlsl + the C++ PushBlock) ---
struct PushBlock {
    uint indirectBufferBindlessIndex;
    uint dagBufferBindlessIndex;
    uint pageToSlotBufferBindlessIndex;
    uint pageCacheBufferBindlessIndex;
    uint cameraSlot;
    uint visImageBindlessIndex;
    uint pageSlotBytes;
    uint pageCount;
    uint dagNodeCount;
};
[[vk::push_constant]] PushBlock pc;

// --- Task payload (shape mirrors mp_raster.task.hlsl::TaskPayload) --------
// M4.5: localClusterIdx added. The mesh shader consumes only clusterIdx +
// the pre-computed page/cluster byte offsets today, but the field must be
// present for the task/mesh payload ABI to match.
struct TaskPayload {
    uint clusterIdx;
    uint pageSlotOffsetB;
    uint clusterOnDiskOffB;
    uint localClusterIdx;
    uint triangleBlobOffB;
};

// --- Camera load (matches visibility_buffer.mesh.hlsl pattern) ------------
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

// --- ClusterOnDisk reader --------------------------------------------------
// Reads the subset of fields the mesh shader actually needs. Full layout
// documented in src/asset/MpAssetFormat.h::ClusterOnDisk — 76 bytes:
//   0  : u32 vertexCount
//   4  : u32 triangleCount
//   8  : u32 vertexOffset     (byte offset in this page's vertex blob)
//   12 : u32 triangleOffset   (byte offset in this page's triangle blob)
//   16 : f32[4] boundsSphere
//   32 : f32[3] coneApex
//   44 : f32[3] coneAxis
//   56 : f32    coneCutoff
//   60 : f32    maxSimplificationError
//   64 : u32    dagLodLevel
//   68 : u32    materialIndex
//   72 : u32    _pad1
struct ClusterFields {
    uint vertexCount;
    uint triangleCount;
    uint vertexOffsetInBlob;
    uint triangleOffsetInBlob;
};

ClusterFields loadClusterFields(uint pageByteOffset, uint clusterOnDiskOff) {
    RWByteAddressBuffer pageBuf = g_rwBuffers[NonUniformResourceIndex(pc.pageCacheBufferBindlessIndex)];
    const uint base = pageByteOffset + clusterOnDiskOff;
    uint4 first4 = pageBuf.Load4(base);
    ClusterFields c;
    c.vertexCount          = first4.x;
    c.triangleCount        = first4.y;
    c.vertexOffsetInBlob   = first4.z;
    c.triangleOffsetInBlob = first4.w;
    return c;
}

// Compute the byte offset of this page's concatenated vertex blob. Per the
// .mpa v1 contract, the vertex blob starts immediately after all
// ClusterOnDisk entries: PagePayloadHeader + clusterCount * ClusterOnDisk.
// We read clusterCount from the PagePayloadHeader at offset 0 of the page.
uint pageVertexBlobStart(uint pageByteOffset) {
    RWByteAddressBuffer pageBuf = g_rwBuffers[NonUniformResourceIndex(pc.pageCacheBufferBindlessIndex)];
    uint clusterCount = pageBuf.Load(pageByteOffset + 0u);
    // Defensive cap — a corrupt clusterCount shouldn't let us walk off the
    // end of the page. The reader's validation gate rejects pages with
    // clusterCount beyond the bake's architectural limit.
    if (clusterCount > 4096u) clusterCount = 4096u;
    return pageByteOffset + MP_PAGE_PAYLOAD_HEADER_BYTES
         + clusterCount * MP_CLUSTER_ON_DISK_STRIDE;
}

// Compute the byte offset of this page's triangle blob: starts right after
// the vertex blob, which itself is clusterVertexCountSum * 32. For M3.3 we
// keep the "one cluster per page" assumption from the task shader; so the
// triangle blob starts at vertexBlobStart + cluster.vertexCount*32.
uint pageTriangleBlobStart(uint vertexBlobStart, uint totalVertexCount) {
    return vertexBlobStart + totalVertexCount * 32u;
}

// Read one vertex (position only — normal/uv are ignored in M3.3 because
// the vis buffer only needs clip-space depth to win atomic-min). Each
// vertex is 32 B: vec3 pos (12) + vec3 normal (12) + vec2 uv (8).
float3 loadVertexPos(uint vertexBlobStart, uint vertexIndexInBlob) {
    RWByteAddressBuffer pageBuf = g_rwBuffers[NonUniformResourceIndex(pc.pageCacheBufferBindlessIndex)];
    const uint vertexAddr = vertexBlobStart + vertexIndexInBlob * 32u;
    uint3 bits = pageBuf.Load3(vertexAddr);
    return float3(asfloat(bits.x), asfloat(bits.y), asfloat(bits.z));
}

// Read one packed u8 triangle index. Triangle blob is 3 bytes per tri; we
// need to handle the misaligned reads by loading u32 words and bit-masking.
// Returns the 3 local vertex indices (0..vertexCount-1) as u32.
uint3 loadTriangleIndices(uint triangleBlobStart, uint clusterTriOffsetBytes, uint triIndex) {
    RWByteAddressBuffer pageBuf = g_rwBuffers[NonUniformResourceIndex(pc.pageCacheBufferBindlessIndex)];
    const uint byteOffset  = triangleBlobStart + clusterTriOffsetBytes + triIndex * 3u;
    const uint wordOffset  = byteOffset & ~3u;      // align to 4 bytes
    const uint shift       = (byteOffset & 3u) * 8u;

    uint word0 = pageBuf.Load(wordOffset);
    uint word1 = pageBuf.Load(wordOffset + 4u);

    // Extract 3 consecutive bytes starting at byteOffset — mirrors the
    // pattern in visibility_buffer.mesh.hlsl (see its shift-==0 guard).
    uint bits;
    if (shift == 0u) {
        bits = word0;
    } else {
        bits = (word0 >> shift) | (word1 << (32u - shift));
    }
    return uint3(bits & 0xFFu, (bits >> 8u) & 0xFFu, (bits >> 16u) & 0xFFu);
}

// --- Mesh shader output structs --------------------------------------------
// Fragment input: clusterId flows through as a per-primitive attribute so
// the vis-packing in PSMain is stable even when adjacent primitives in the
// same mesh group straddle the same pixel (atomic-min sorts it out).
struct VertexOutput {
    float4 pos : SV_Position;
};

struct PrimAttrib {
    nointerpolation uint clusterId  : CLUSTER_ID;
    nointerpolation uint triangleId : TRIANGLE_ID;
};

// --- Mesh shader main ------------------------------------------------------
// One thread per triangle — up to MP_MAX_TRIANGLES. Threads below
// vertex_count also participate in vertex transform (parallel load).
[numthreads(MP_MAX_TRIANGLES, 1, 1)]
[outputtopology("triangle")]
void MSMain(
    in uint3 gid        : SV_GroupID,
    in uint  groupIndex : SV_GroupIndex,
    in payload TaskPayload payload,
    out vertices  VertexOutput verts[MP_MAX_VERTICES],
    out indices   uint3        tris[MP_MAX_TRIANGLES],
    out primitives PrimAttrib  prims[MP_MAX_TRIANGLES]
) {
    ClusterFields cluster = loadClusterFields(payload.pageSlotOffsetB,
                                              payload.clusterOnDiskOffB);
    const uint vertexBlobStart   = pageVertexBlobStart(payload.pageSlotOffsetB);
    const uint triangleBlobStart = payload.pageSlotOffsetB + payload.triangleBlobOffB;
    uint vertexCount   = cluster.vertexCount;
    uint triangleCount = cluster.triangleCount;
    if (vertexCount   > MP_MAX_VERTICES)  vertexCount   = MP_MAX_VERTICES;
    if (triangleCount > MP_MAX_TRIANGLES) triangleCount = MP_MAX_TRIANGLES;
    if (vertexCount == 0u) triangleCount = 0u;
    SetMeshOutputCounts(vertexCount, triangleCount);

    CameraData cam = loadCamera(pc.cameraSlot);

    // Per-vertex transform — one thread per vertex, threads beyond
    // vertexCount idle. No instance transform for M3.3: micropoly is
    // static geometry (Principle 5), so object-space positions are
    // world-space positions. An instance matrix will land with M4's
    // MicropolyInstance widening.
    if (groupIndex < vertexCount) {
        const uint vertexIndexInBlob =
            cluster.vertexOffsetInBlob / 32u + groupIndex;
        const float3 pos = loadVertexPos(vertexBlobStart, vertexIndexInBlob);
        // Match the engine-wide -90° Y correction applied to glTF scenes
        // in Application.cpp (glb +X → physics +Z). The bake writes
        // positions in the source glTF's native axes; without this
        // rotation the mpa geometry lands 90° off vs the glTF mesh, so
        // almost every triangle loses depth inside the glTF envelope.
        const float3 posC = float3(-pos.z, pos.y, pos.x);
        float4 clipPos = mul(cam.viewProj, float4(posC, 1.0f));
        VertexOutput v;
        v.pos = clipPos;
        verts[groupIndex] = v;
    }

    // Per-triangle emission — one thread per triangle.
    if (groupIndex < triangleCount) {
        uint3 idx = loadTriangleIndices(triangleBlobStart,
                                        cluster.triangleOffsetInBlob,
                                        groupIndex);
        // Ensure indices stay inside the emitted vertex range (defensive).
        idx.x = min(idx.x, vertexCount - 1u);
        idx.y = min(idx.y, vertexCount - 1u);
        idx.z = min(idx.z, vertexCount - 1u);
        tris[groupIndex] = idx;

        PrimAttrib pa;
        pa.clusterId  = payload.clusterIdx;
        pa.triangleId = groupIndex;
        prims[groupIndex] = pa;
    }

    // gid is unused in the single-group-per-cluster shape but referenced
    // here to silence DXC's unused-parameter warning under /W4.
    (void)gid;
}

// --- Fragment shader: 64-bit atomic-min write to the vis image ------------
// One fragment per rasterised pixel; clusterId + triangleId arrive as
// per-primitive attributes (no interpolation). SV_Position provides
// screen-space coords + NDC depth in [0,1].
void PSMain(
    in VertexOutput input,
    in PrimAttrib   prim
) {
    const uint2 pixelCoord = uint2(input.pos.xy);
    const float depth      = input.pos.z;

    // M4.1 — vis-pack v2: rasterClass is the new 2nd arg. HW raster path
    // (this shader is the sole producer today) always tags samples with
    // kMpRasterClassHw; the SW raster kernel landing in M4.2 will pass
    // kMpRasterClassSw so the M4.6 DebugMode::MicropolyRasterClass overlay
    // can distinguish the two paths pixel-by-pixel.
    const uint64_t packed = PackMpVis64(asuint(depth),
                                        kMpRasterClassHw,
                                        prim.clusterId,
                                        prim.triangleId);

    // R64_UINT atomic-MAX on the bindless storage image. Under reverse-Z
    // (far=0, near=1) the LARGEST packed value is the NEAREST sample, so
    // atomic-max picks the winner in a single RMW. The vis image is cleared
    // to 0 (kMpVisEmpty) each frame so any real fragment wins the first
    // write trivially (depth > 0 for almost every rasterised sample).
    // Requires VK_EXT_shader_image_atomic_int64 — the runtime gates
    // pipeline construction on supportsShaderImageInt64() (see MicropolyCaps).
    // DXC lowers this to OpImageTexelPointer + OpAtomicUMax on the 64-bit
    // texel when the image format is R64ui (vk 1.3 + the extension).
    InterlockedMax(g_visImages[NonUniformResourceIndex(pc.visImageBindlessIndex)][pixelCoord],
                   packed);
}
