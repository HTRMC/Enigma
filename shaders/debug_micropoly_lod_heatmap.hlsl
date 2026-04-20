// debug_micropoly_lod_heatmap.hlsl
// =================================
// Per-pixel heatmap visualising the LOD level of the DAG node that produced
// each Micropoly fragment. Complements debug_micropoly_raster_class.hlsl
// (M4.6) which colours by raster path; this M6.1 overlay colours by DAG
// depth so streaming decisions are visible at a glance.
//
// Pipeline:
//   1. Sample the 64-bit vis image at the pixel coord.
//   2. Reject kMpVisEmpty -> black.
//   3. UnpackMpVis64 -> clusterIdx (the DAG node index).
//   4. Load MpDagNode.m2.w from the DAG SSBO, decode lodLevel (high 8 bits).
//   5. Normalise lodLevel to [0,1] and map through a simple hot-to-cold
//      gradient (blue = leaf / high-res, red = coarse / top-level).
//
// Availability: gated on Device::supportsShaderImageInt64() AND an active
// m_micropolyRasterPass AND a populated DAG SSBO on the C++ side. When
// any of those are missing the Renderer never routes this mode — but we
// defensively bail when `dagBufferBindless == UINT32_MAX` or when the
// decoded clusterIdx is out of range.
//
// Push-block shape is a dedicated struct (not the shared DebugVisPushBlock)
// because only three u32s are needed and it keeps the shader surface tight.

#include "common.hlsl"
#include "micropoly/mp_vis_pack.hlsl"

// 64-bit storage image alias — matches material_eval.comp.hlsl's MP_ENABLE
// path and debug_micropoly_raster_class.hlsl's vis read. fragmentStoresAnd-
// Atomics is enabled at Device creation so the capability surface is satisfied.
[[vk::binding(1, 0)]]
RWTexture2D<uint64_t> g_storageImages64[] : register(u0, space0);

// StructuredBuffer<float4> bindless alias for the DAG SSBO. Matches the
// `g_buffers` declaration used by mp_cluster_cull.comp.hlsl (binding 2/0).
[[vk::binding(2, 0)]]
StructuredBuffer<float4> g_buffers[] : register(t0, space1);

struct PushBlock {
    uint visImage64Bindless;  // R64_UINT vis image slot
    uint dagBufferBindless;   // StructuredBuffer<MpDagNode> as 3 float4 per node
    uint dagNodeCount;        // defensive bounds check
    uint _pad;
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

// Simple hot-to-cold gradient. t in [0,1]:
//   t=0 (leaf, high res)   -> blue
//   t~0.5                  -> green
//   t=1 (coarse top-level) -> red
// Each channel ramps with smoothstep so the transitions are visually
// continuous rather than the hard hue boundaries of a naive HSV mapping.
float3 lodHeatmap(float t) {
    t = saturate(t);
    float r = smoothstep(0.3, 1.0, t);
    float g = smoothstep(0.0, 0.5, t) * (1.0 - smoothstep(0.5, 1.0, t));
    float b = smoothstep(0.0, 0.3, 1.0 - t);
    return float3(r, g, b);
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

    // Defensive: when the DAG SSBO is not yet wired through, or the
    // packed clusterIdx is out of range, emit a distinct magenta so the
    // mis-wiring is visible rather than silently miscoloured.
    if (pc.dagBufferBindless == 0xFFFFFFFFu || clusterIdx >= pc.dagNodeCount) {
        return float4(1.0, 0.0, 1.0, 1.0);
    }

    // MpDagNode GPU layout: 3 float4 per node (see mp_cluster_cull.comp.hlsl).
    //   float4 m0 = {center.xyz,    radius}
    //   float4 m1 = {coneApex.xyz,  coneCutoff}
    //   float4 m2 = {coneAxis.xyz,  asfloat(packed)} where
    //               packed = pageId(low 24) | lodLevel(high 8).
    StructuredBuffer<float4> dag = g_buffers[NonUniformResourceIndex(pc.dagBufferBindless)];
    float4 m2 = dag[clusterIdx * 3u + 2u];
    uint   packedWord = asuint(m2.w);
    uint   lodLevel   = (packedWord >> 24u) & 0xFFu;

    // Normalise to 0..1 across 8 LOD levels (plan cap). 7.0 is the divisor
    // so lodLevel=7 maps to 1.0; levels beyond that saturate.
    float t = saturate(float(lodLevel) / 7.0);
    return float4(lodHeatmap(t), 1.0);
}
