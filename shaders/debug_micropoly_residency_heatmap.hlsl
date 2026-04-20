// debug_micropoly_residency_heatmap.hlsl
// =========================================
// Per-pixel residency overlay: samples the 64-bit Micropoly vis image,
// decodes the cluster -> pageId via the DAG SSBO, then looks up the
// page's PageCache slot index in the `pageToSlotBuffer` to colour the
// pixel by residency state:
//
//   green   : page resident (slotIndex != UINT32_MAX)
//   magenta : page NOT resident (geometry rendered from a page that was
//             evicted between raster and this debug pass, or a wiring
//             bug — M6.1 uses this as "something strange happened here")
//   black   : empty vis pixel (no micropoly contribution)
//
// The plan (§3.M6 line 568) specifies per-slot AGE visualisation (fresh =
// green, oldest-retained = red). ResidencyManager does not yet expose a
// per-slot age — when it does, this shader will become the obvious
// extension point. The resident-vs-non-resident dichotomy here is a
// strictly simpler approximation that still surfaces streaming hiccups.
//
// Availability: same gate as MicropolyRasterClass / LodHeatmap — the
// Renderer won't construct / route this unless shaderImageInt64 is
// supported and a HW raster pass is active.

#include "common.hlsl"
#include "micropoly/mp_vis_pack.hlsl"

// fragmentStoresAndAtomics is enabled at Device creation so the capability
// surface is satisfied even though the PS only reads the vis image.
[[vk::binding(1, 0)]]
RWTexture2D<uint64_t> g_storageImages64[] : register(u0, space0);

// StructuredBuffer<float4> bindless alias — same binding as mp_cluster_cull.
[[vk::binding(2, 0)]]
StructuredBuffer<float4> g_buffers[] : register(t0, space1);

struct PushBlock {
    uint visImage64Bindless;
    uint dagBufferBindless;
    uint pageToSlotBindless;
    uint dagNodeCount;
    uint pageCount;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};
[[vk::push_constant]] PushBlock pc;

struct VSOut {
    float4 pos : SV_Position;
    float2 uv  : TEXCOORD0;
};

VSOut VSMain(uint vid : SV_VertexID) {
    float2 uv = float2((vid << 1) & 2, vid & 2);
    VSOut o;
    o.pos = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    o.uv  = uv;
    return o;
}

// Look up the PageCache slot index for a given pageId. Mirrors
// mp_raster.task.hlsl::slotForPage and mp_cluster_cull.comp.hlsl::
// classSlotForPage — same float4-packed u32 layout. Returns UINT32_MAX
// when pageId is out of range or the page is non-resident.
uint residencySlotForPage(uint pageId) {
    if (pageId >= pc.pageCount) return 0xFFFFFFFFu;
    StructuredBuffer<float4> buf = g_buffers[NonUniformResourceIndex(pc.pageToSlotBindless)];
    const uint float4Idx = pageId >> 2u;
    const uint component = pageId & 3u;
    uint4 packed = asuint(buf[float4Idx]);
    if      (component == 0u) return packed.x;
    else if (component == 1u) return packed.y;
    else if (component == 2u) return packed.z;
    else                      return packed.w;
}

float4 PSMain(VSOut vs) : SV_Target {
    uint2 pixelCoord = uint2(vs.pos.xy);
    RWTexture2D<uint64_t> vis = g_storageImages64[NonUniformResourceIndex(pc.visImage64Bindless)];
    uint64_t packed = vis[pixelCoord];
    if (packed == kMpVisEmpty) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }

    uint depth32, rasterClass, clusterIdx, triIdx;
    UnpackMpVis64(packed, depth32, rasterClass, clusterIdx, triIdx);

    // Wiring-sanity guards: if the DAG buffer or pageToSlot buffer isn't
    // wired (UINT32_MAX sentinel), or the clusterIdx escapes the DAG, we
    // can't answer the residency question. Emit a distinct yellow so the
    // mis-wiring is visible rather than silently defaulting to "resident".
    if (pc.dagBufferBindless == 0xFFFFFFFFu
        || pc.pageToSlotBindless == 0xFFFFFFFFu
        || clusterIdx >= pc.dagNodeCount) {
        return float4(1.0, 1.0, 0.0, 1.0);
    }

    // MpDagNode GPU layout — 5 float4 per node (M4 widening → m3 for SSE
    // errors; M4-fix widening → m4 for parentCenter; see
    // mp_cluster_cull.comp.hlsl::loadDagNode). Only m2.w is needed here:
    //   m2.w = asfloat(pageId(low 24) | lodLevel(high 8)).
    StructuredBuffer<float4> dag = g_buffers[NonUniformResourceIndex(pc.dagBufferBindless)];
    float4 m2 = dag[clusterIdx * 5u + 2u];
    uint   packedWord = asuint(m2.w);
    uint   pageId     = packedWord & 0x00FFFFFFu;

    const uint slotIndex = residencySlotForPage(pageId);
    if (slotIndex == 0xFFFFFFFFu) {
        // Non-resident. Under normal steady-state operation this is
        // rare — the cluster cull pass already filters out non-resident
        // clusters before emitting an indirect draw, so a magenta pixel
        // here implies eviction-racing-raster or a DAG/pageCount bound
        // mismatch. Either way: flag it loud.
        return float4(1.0, 0.0, 1.0, 1.0);
    }
    // Resident. Until ResidencyManager tracks per-slot age, all resident
    // pages paint uniform green.
    return float4(0.0, 1.0, 0.0, 1.0);
}
