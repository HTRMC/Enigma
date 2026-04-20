// debug_micropoly_raster_class.hlsl
// ===================================
// Per-pixel R/G debug overlay decoding the 2-bit rasterClassBits field from
// the 64-bit Micropoly visibility image (M4.6).
//
//   red   : HW-rasterised sample (kMpRasterClassHw == 0)
//   green : SW-rasterised sample (kMpRasterClassSw == 1)
//   blue  : reserved raster-class tag (2..3 — not emitted today)
//   black : empty pixel (packed == kMpVisEmpty)
//
// Availability: this debug mode is gated on Device::supportsShaderImageInt64()
// on the C++ side. When the device lacks the Int64 storage-image capability,
// the Renderer never constructs the pass nor routes the frame through this
// shader — matching the existing MicropolyRasterPass capability gate.
//
// Push-block shape mirrors DebugVisPushBlock from DebugVisualizationPass.cpp.
// Unused fields are ignored.

#include "common.hlsl"
#include "micropoly/mp_vis_pack.hlsl"

// 64-bit storage image alias of binding 1/0 — matches material_eval.comp.hlsl's
// MP_ENABLE path. The vis image lives at the bindless slot passed via the push
// block (visOrHdrSlot, repurposed as visImage64Bindless for this mode).
//
// NOTE: intentionally NOT marked [[vk::readonly]] — DXC doesn't recognise the
// attribute on images (emits `unknown attribute` warning). fragmentStoresAnd-
// Atomics is enabled at Device creation so the capability surface is already
// satisfied; the PS is effectively read-only at the call sites even without
// the SPIR-V NonWritable decoration.
[[vk::binding(1, 0)]]
RWTexture2D<uint64_t> g_storageImages64[] : register(u0, space0);

struct PushBlock {
    uint _unused0;
    uint _unused1;
    uint _unused2;
    uint _unused3;
    uint _unused4;
    uint _samplerSlot;
    uint visImage64Bindless; // aliased onto DebugVisPushBlock::visOrHdrSlot
    uint _pad;
    float4 _a;
    float4 _b;
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

float4 PSMain(VSOut vs) : SV_Target {
    uint2 pixelCoord = uint2(vs.pos.xy);
    RWTexture2D<uint64_t> vis = g_storageImages64[NonUniformResourceIndex(pc.visImage64Bindless)];
    uint64_t packed = vis[pixelCoord];
    if (packed == kMpVisEmpty) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }
    uint depth32, rasterClass, clusterIdx, triIdx;
    UnpackMpVis64(packed, depth32, rasterClass, clusterIdx, triIdx);
    float3 color =
        (rasterClass == kMpRasterClassHw) ? float3(1.0, 0.0, 0.0) :
        (rasterClass == kMpRasterClassSw) ? float3(0.0, 1.0, 0.0) :
                                            float3(0.0, 0.0, 1.0); // reserved
    return float4(color, 1.0);
}
