// debug_micropoly_bin_overflow.hlsl
// ==================================
// Per-pixel overlay colouring each screen-space 8x8 tile by its SW-raster
// bin fill level. Produced in M6 to complete the debug-overlay inventory
// (BinOverflowHeat is the 5th radio entry — see plan §3.M6).
//
// Colour key:
//   black                          : empty tile (binCount == 0 and no spill)
//   green (dim -> bright gradient) : 1 .. MP_SW_TILE_BIN_CAP-1 entries
//   yellow                         : tile saturated at MP_SW_TILE_BIN_CAP
//   red                            : tile had >=1 spilled entry (dominates
//                                    yellow/green so saturation + overflow
//                                    is always flagged loud)
//
// Source SSBOs:
//   tileBinCount  : RWByteAddressBuffer, u32 * numTiles. Post-cap count per
//                   tile written by sw_raster_bin.comp.hlsl.
//   spillBuffer   : RWByteAddressBuffer, header {u32 spillCount,
//                   u32 spillDroppedCount} then array of
//                   {u32 tileIdx, u32 triRef} pairs.
//
// Availability (Renderer side): gated on m_micropolySwRasterPass != nullptr
// so this shader is never routed unless both bin buffers exist. The shader
// itself is defensive against UINT32_MAX bindless slots so a wiring bug
// shows up as black instead of sampling an invalid descriptor.
//
// Per-pixel spill scan cost: O(min(spillCount, MP_SW_SPILL_CAP,
// MP_SW_SPILL_SCAN_CAP)). The MP_SW_SPILL_SCAN_CAP bound (256) is an
// absolute ceiling applied after the runtime cap so an adversarial input
// that fills the 65,536-entry spill list can't make the debug overlay
// dominate frame time — the overlay still accurately reports "this tile
// had a spill" when at least one hit falls inside the scanned prefix.

#include "common.hlsl"
#include "micropoly/mp_vis_pack.hlsl"

// Mirror the #defines at the top of sw_raster_bin.comp.hlsl. Any change
// there MUST be reflected here — mismatched tile sizes or bin caps would
// silently mis-classify tiles.
#define MP_SW_TILE_X         8u
#define MP_SW_TILE_Y         8u
#define MP_SW_TILE_BIN_CAP   256u
#define MP_SW_SPILL_CAP      65536u
// Per-pixel spill scan ceiling. Caps the worst-case spill iteration so
// adversarial asset inputs don't make this overlay O(pixels * MP_SW_SPILL_CAP).
// Previously 2048 — at 1080p that was ~2B ops and tripped the Windows 2s
// TDR on large assets (real-world observation: BMW GT3 asset device-lost
// when BinOverflow overlay was selected). Dropped to 256 so the worst-case
// per-pixel cost is ~236M ops (~50ms), comfortably under TDR. The overlay
// still accurately reports "this tile had a spill" when at least one hit
// falls inside the scanned prefix — debug visualisation, not audit.
#define MP_SW_SPILL_SCAN_CAP 256u

// Raw byte-addressable alias of the engine's bindless RW byte-buffer slot.
// The engine's descriptor layout puts RWByteAddressBuffer at binding (5, 0)
// — binding (1, 0) is RWTexture2D storage-image descriptors. Mismatching
// the binding routes Loads through the storage-image descriptor and
// reinterprets texel bytes as spill-list counters, producing multi-billion
// iteration counts and an instant Windows TDR. See sw_raster_bin.comp.hlsl
// for the peer writer that uses the same (5, 0) binding.
[[vk::binding(5, 0)]]
RWByteAddressBuffer g_rwBuffers[] : register(u1, space0);

struct PushBlock {
    uint tileBinCountBindless;
    uint spillBufferBindless;
    uint tilesX;
    uint tilesY;
    uint screenWidth;
    uint screenHeight;
    uint _pad0;
    uint _pad1;
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
    // Defensive: if either bin buffer isn't wired the Renderer shouldn't
    // route us here — fall through to black so the wiring bug surfaces
    // rather than sampling an invalid bindless slot.
    if (pc.tileBinCountBindless == 0xFFFFFFFFu
        || pc.spillBufferBindless == 0xFFFFFFFFu
        || pc.tilesX == 0u
        || pc.tilesY == 0u) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }

    const uint2 pixelCoord = uint2(vs.pos.xy);
    const uint  tileX      = pixelCoord.x / MP_SW_TILE_X;
    const uint  tileY      = pixelCoord.y / MP_SW_TILE_Y;
    const uint  tileIdx    = tileY * pc.tilesX + tileX;
    const uint  numTiles   = pc.tilesX * pc.tilesY;
    if (tileIdx >= numTiles) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }

    RWByteAddressBuffer tileCount = g_rwBuffers[NonUniformResourceIndex(pc.tileBinCountBindless)];
    const uint binCount = tileCount.Load(tileIdx * 4u);

    // Scan the spill list to decide whether THIS tile was one of the
    // overflow victims. The list is unsorted — linear scan is the only
    // option without auxiliary indexing.
    RWByteAddressBuffer spill = g_rwBuffers[NonUniformResourceIndex(pc.spillBufferBindless)];
    const uint spillCount = spill.Load(0u);
    const uint scanLimit  = min(min(spillCount, MP_SW_SPILL_CAP), MP_SW_SPILL_SCAN_CAP);
    bool spilled = false;
    for (uint i = 0u; i < scanLimit; ++i) {
        // Entry layout: 8 B = {u32 tileIdx, u32 triRef} at header offset 8.
        const uint entryTileIdx = spill.Load(8u + i * 8u);
        if (entryTileIdx == tileIdx) {
            spilled = true;
            break;
        }
    }

    // Red dominates: a spilled tile flags overflow loud even if its binCount
    // didn't reach the cap (the overflow went straight to the spill list
    // because some earlier triangle already pushed it to BIN_CAP).
    if (spilled) {
        return float4(1.0, 0.2, 0.2, 1.0);
    }
    if (binCount == 0u) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }
    if (binCount >= MP_SW_TILE_BIN_CAP) {
        return float4(1.0, 0.85, 0.0, 1.0); // yellow — saturated
    }
    // Green gradient: dim at 1 entry, bright at BIN_CAP-1.
    const float fill = saturate(float(binCount) / float(MP_SW_TILE_BIN_CAP));
    return float4(0.0, fill, 0.0, 1.0);
}
