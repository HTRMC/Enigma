// mp_vis_pack.hlsl
// ==================
// Shared pack/unpack helpers for the 64-bit Micropoly visibility image.
// Included by HW raster shaders (M3.3), SW raster shaders (M4.2+), and
// the modified material_eval pass (M3.4 + the M4.6 debug overlay). This
// file is INCLUDE-ONLY and must NEVER appear in the spirv_diff test
// manifest — it produces no standalone SPIR-V blob.
//
// Encoding contract (plan §3.M4 layout, v2 — adds rasterClassBits):
//
//   bits 63..32 : depth           (32 bits — asuint(f32), reverse-Z)
//   bits 31..30 : rasterClassBits (2 bits — 0=HW, 1=SW, 2..3 reserved)
//   bits 29..7  : clusterId       (23 bits — up to 8,388,608 clusters)
//   bits  6..0  : triangleId      (7 bits  — 128 tris max per cluster)
//
// The 23-bit cluster field is still ~128× the current
// kMpMaxIndirectDrawClusters = 65536 cap enforced by
// MicropolyCullPass.h (there's a compile-time static_assert on the C++
// side that trips if anyone raises the cap past (1<<23)).
//
// Depth-comparison rule (reverse-Z convention, matches the rest of the
// Enigma engine — see material_eval.comp.hlsl, gpu_cull.comp.hlsl,
// visibility_buffer.mesh.hlsl, terrain_cdlod.mesh.hlsl, and
// mp_cluster_cull.comp.hlsl):
//
//   NEAREST sample has the LARGEST depth (far=0, near=1). We compare packed
//   64-bit values as unsigned integers; because the depth lives in the high
//   32 bits and is non-negative IEEE-754, bitwise uint ordering matches
//   numeric ordering — so the LARGEST packed value wins InterlockedMax,
//   which is the NEAREST sample. Readers must use `>` to decide "mp wins
//   over meshlet", mirroring the same convention.
//
// v2 note (M4.1): the rasterClassBits field was carved out of the former
// 25-bit clusterId space rather than tacked on the low end so the depth
// stays in the high 32 bits — preserving the reverse-Z / InterlockedMax
// ordering semantics with zero atomic-op changes. The sentinel
// kMpVisEmpty = 0 still loses to any real fragment (depth=0 AND class=HW
// AND cluster=0 AND tri=0 is strictly smaller than any real sample).

#ifndef MICROPOLY_VIS_PACK_HLSL
#define MICROPOLY_VIS_PACK_HLSL

// --- Raster-class enum (2 bits) --------------------------------------------
// Extend with 2=reserved / 3=reserved if a future milestone needs additional
// sample provenance tags (e.g. CAS-fallback path, compute-raster variant).
static const uint kMpRasterClassHw = 0u;
static const uint kMpRasterClassSw = 1u;

// --- Bit layout constants --------------------------------------------------
// Named so any future layout tweak is a one-file edit. Downstream code MUST
// route through these rather than hard-coding literal shifts/masks.
static const uint kMpVisTriBits         = 7u;     // 128 tris/cluster (plan cap)
static const uint kMpVisTriMask         = (1u << kMpVisTriBits) - 1u;   // 0x7F
static const uint kMpClusterShift       = kMpVisTriBits;                // 7
static const uint kMpVisClusterBits     = 23u;    // 8,388,608 clusters (v2)
static const uint kMpClusterMask        = (1u << kMpVisClusterBits) - 1u;  // 0x007FFFFF
static const uint kMpRasterClassShift   = kMpClusterShift + kMpVisClusterBits;  // 30
static const uint kMpRasterClassBits    = 2u;
static const uint kMpRasterClassMask    = (1u << kMpRasterClassBits) - 1u; // 0x3

// Pack depth + raster-class + cluster + triangle into a 64-bit vis value.
// The returned uint64_t is fed directly into InterlockedMax on the vis
// image / buffer. The LARGEST value wins — under reverse-Z (see file
// banner) that is the NEAREST sample.
//
// depth32: asuint(ndc-z float). Caller owns the reverse-Z convention.
// rasterClass: one of kMpRasterClassHw / kMpRasterClassSw. Masked to 2 bits
//              defensively — a malformed argument cannot leak into the
//              cluster field.
// clusterIdx / triangleIdx: masked to their field widths, again defensive.
//                           8M clusters is well above the current 65,536 cap
//                           but the `& kMpClusterMask` keeps the contract
//                           strict if a future bake emits more.
uint64_t PackMpVis64(uint depth32,
                     uint rasterClass,
                     uint clusterIdx,
                     uint triangleIdx) {
    const uint payload =
          ((rasterClass  & kMpRasterClassMask)  << kMpRasterClassShift)
        | ((clusterIdx   & kMpClusterMask)      << kMpClusterShift)
        |  (triangleIdx  & kMpVisTriMask);
    return (((uint64_t)depth32) << 32) | (uint64_t)payload;
}

// Reverse of PackMpVis64. Reverse-Z handling is the caller's problem —
// see banner. Emits all four fields; callers that don't care about
// rasterClass (the M3.4 magenta stub, for example) may ignore it.
void UnpackMpVis64(uint64_t packed,
                   out uint depth32,
                   out uint rasterClass,
                   out uint clusterIdx,
                   out uint triangleIdx) {
    depth32     = (uint)(packed >> 32);
    const uint payload = (uint)(packed & 0xFFFFFFFFull);
    rasterClass = (payload >> kMpRasterClassShift) & kMpRasterClassMask;
    clusterIdx  = (payload >> kMpClusterShift)     & kMpClusterMask;
    triangleIdx =  payload                         & kMpVisTriMask;
}

// Sentinel "no sample" value used to clear the vis image at frame start
// and to recognize pixels that received no micropoly contribution. Under
// reverse-Z + InterlockedMax semantics the empty slot must be the SMALLEST
// possible 64-bit value (0), so any real fragment (whose depth, being a
// rasterised sample inside the [0,1] NDC range, has a non-zero packed
// representation for anything not at the exact far plane) wins the first
// atomic-max write. MaterialEvalPass (M3.4) treats any value == kMpVisEmpty
// as "fall back to the existing 32-bit vis path." The v2 layout preserves
// this: class=HW(0) + cluster=0 + tri=0 + depth=0 packs to 0 identically.
static const uint64_t kMpVisEmpty = 0ull;

#endif // MICROPOLY_VIS_PACK_HLSL
