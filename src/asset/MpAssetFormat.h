#pragma once

// MpAssetFormat.h
// ================
// On-disk schema for Micropoly Asset files (`.mpa`). Produced by the offline
// `enigma-mpbake` tool, consumed by the runtime `MpAssetReader`. Version 1
// of the format is what M1 ships; version bumps are backwards-incompatible.
//
// Design constraints (cf. .omc/plans/ralplan-micropolygon.md §3.M1):
//   * Deterministic on-disk layout across MSVC / Clang / GCC. We reach for
//     `#pragma pack(push, 1)` + explicit padding rather than relying on the
//     compiler's default ABI; pack(1) is portable across the three Windows
//     toolchains we ship against and across Linux when the tool is built
//     there for a CI bake.
//   * Every struct has a static_assert on sizeof in MpAssetFormat.cpp so
//     layout drift is caught at compile time, not at first-read time.
//   * No pointers, no std::string, no std::vector — these are POD-only
//     views onto a memory-mapped byte range.
//
// File layout (M1c):
//   offset 0                     : MpAssetHeader (40 B)
//   header.dagByteOffset         : MpDagNode[dagNodeCount] (36 B each)
//   header.pagesByteOffset       : MpPageEntry[pageCount] (32 B each)
//   header.boundsByteOffset      : reserved (empty in v1 — future bounds
//                                   hierarchy; always equals
//                                   pagesByteOffset + pageCount*32)
//   after page table             : raw zstd-compressed page payloads,
//                                   variable size, per MpPageEntry.
//
// Per-page payload (after zstd-decompress):
//   offset 0                : PagePayloadHeader (16 B)
//   offset 16               : ClusterOnDisk[clusterCount] (76 B each)
//   then                    : concatenated vertex blob (32 B per vertex:
//                              vec3 pos, vec3 normal, vec2 uv) in
//                              cluster-order. Offsets inside clusters
//                              are byte-relative to the start of this
//                              concatenated blob.
//   then                    : concatenated triangle blob (3 bytes per
//                              triangle, u8 local indices). Offsets in
//                              ClusterOnDisk are byte-relative to the
//                              start of THIS blob.

#include "core/Types.h"

#include <cstddef>

namespace enigma::asset {

// Magic four-char-code that every `.mpa` file starts with. The literal is
// "MPA1" — the "1" suffix is the format generation, independent of the
// `version` field below. Together they let the loader distinguish a
// first-format-generation file from a future "MPA2" re-cut without needing
// the version field to roll over.
inline constexpr char kMpAssetMagic[4] = { 'M', 'P', 'A', '1' };

// Current supported format version. Write this into `MpAssetHeader::version`
// at bake time; the reader rejects anything else.
inline constexpr u32 kMpAssetVersion = 1;

// Per-page payload format version. Distinct from the file-format version so
// we can iterate on per-page layout without rotating the outer container
// magic. The reader rejects anything else.
inline constexpr u32 kMpPagePayloadVersion = 1;

// Raw zstd blobs are appended with 16-byte alignment after the page table.
// This is a file-layout hygiene knob (SIMD-friendly, non-crossing-page reads);
// it is NOT required by zstd itself. The PageWriter pads with zero bytes to
// reach this stride before each payload.
inline constexpr u64 kMpPagePayloadAlignment = 16u;

// Per-vertex size in the concatenated vertex blob: vec3 pos + vec3 normal +
// vec2 uv = 8 floats = 32 bytes.
inline constexpr u32 kMpVertexStride = 32u;

// Hard cap on per-page decompressed size. Page payloads should be well under
// 1 MiB for M1; this generous cap protects against disk corruption and future
// adversarial inputs without constraining legitimate growth.
//
// IMPORTANT: this is a corruption-protection cap at the reader level, NOT a
// cache-sizing hint. Do NOT use this constant as a PageCache slot size — typical
// decompressed pages are 25-80 KiB (M1c DamagedHelmet stats). Using 64 MiB as
// a slot size would give only ~16 slots on a 1 GiB pool, wasting ~99.9% of VRAM.
inline constexpr u32 kMpMaxPageDecompressedBytes = 64u * 1024u * 1024u;  // 64 MiB

#pragma pack(push, 1)

// On-disk header. Placed at file offset 0. Byte-offsets to the three trailing
// sections (pages, DAG, bounds) let the reader mmap + jump without a
// sequential parse.
struct MpAssetHeader {
    char  magic[4];            // must equal kMpAssetMagic
    u32   version;             // must equal kMpAssetVersion
    u32   dagNodeCount;        // number of MpDagNode entries in the DAG section
    u32   pageCount;           // number of variable-size compressed pages
    u64   pagesByteOffset;     // file offset of the first page descriptor
    u64   dagByteOffset;       // file offset of the first DAG node
    u64   boundsByteOffset;    // file offset of the bounds-hierarchy root (reserved in v1)
};

// One node in the cluster-group DAG. `boundsSphere` = (center.xyz, radius).
// `maxError` is the screen-space worst-case simplification error at the
// time of the bake; at runtime the selector walks the DAG until child error
// <= screen-space-error threshold.
//
// `parentGroupId` is UINT32_MAX for DAG roots. `firstChildNode` +
// `childCount` form an inclusive child range; a leaf has `childCount == 0`.
// `pageId` indexes into the pages section and says which compressed page
// carries this node's cluster triangle data.
struct MpDagNode {
    // Center xyz + radius w. Stored as plain f32[4] rather than glm::vec4
    // because glm::vec4 can route through an SSE-aligned storage variant
    // (`alignas(16)`) when `GLM_FORCE_ALIGNED_GENTYPES` is defined; that
    // would fight `#pragma pack(1)` and silently change layout. Readers
    // that want a glm::vec4 can memcpy these four floats after load.
    f32       boundsSphere[4];
    f32       maxError;        // worst-case screen-space error (pixels @ bake res)
    u32       parentGroupId;   // UINT32_MAX == root
    u32       firstChildNode;  // index into DAG array
    u32       childCount;      // leaf iff == 0
    u32       pageId;          // index into pages array
};

// One entry in the page index table. Points at a variable-size zstd blob
// in the trailing payload region. `decompressedSize` is the size the caller
// must allocate before calling `fetchPage()`; `compressedSize` is the exact
// number of bytes zstd will read starting at `payloadByteOffset`.
//
// `firstDagNodeIdx` lets the runtime locate the cluster-range this page
// serves without a separate mapping table. `groupId` is debug-only (and
// `pageId`-equivalent in v1, since we emit exactly one page per group).
struct MpPageEntry {
    u64 payloadByteOffset;   // file offset to start of zstd blob
    u32 compressedSize;      // size of the zstd blob on disk
    u32 decompressedSize;    // size after zstd decompress (caller allocates this)
    u32 clusterCount;        // how many clusters in this page
    u32 firstDagNodeIdx;     // index into MpDagNode array of the first cluster in this page
    u32 groupId;             // which DAG group this page serves (for debug / stats)
    u32 _pad;                // align to 32 bytes; reserved for future flags
};

// Per-page payload header, placed at offset 0 of each decompressed page.
// Identifies the payload version + cluster count so readers can bounds-
// check the trailing ClusterOnDisk array.
struct PagePayloadHeader {
    u32 clusterCount;
    u32 version;
    u32 _pad0;
    u32 _pad1;
};

// Per-cluster descriptor inside a decompressed page. Offsets are byte-
// relative to the START of the page's concatenated vertex / triangle
// blobs respectively (NOT relative to the page start).
//
// The fields mirror the in-memory `enigma::mpbake::ClusterData` but in a
// fixed, packed layout suitable for mmap. Bounds + normal cone fields
// are plain float arrays for the same reason as MpDagNode::boundsSphere.
struct ClusterOnDisk {
    u32 vertexCount;        // 0..128                                     (4)
    u32 triangleCount;      // 0..128                                     (4)
    u32 vertexOffset;       // byte offset into per-page vertex blob      (4)
    u32 triangleOffset;     // byte offset into per-page triangle blob    (4)
    f32 boundsSphere[4];    // center.xyz, radius                         (16)
    f32 coneApex[3];        // normal-cone apex                           (12)
    f32 coneAxis[3];        // normal-cone axis                           (12)
    f32 coneCutoff;         // cos(half-angle)                            (4)
    f32 maxSimplificationError;                                        // (4)
    u32 dagLodLevel;                                                   // (4)
    u32 materialIndex;      // index into scene material buffer          (4)
    u32 _pad1;              // align to 76-byte stride                    (4)
};

#pragma pack(pop)

// Compile-time layout checks — defined in the .cpp so they fire exactly
// once per TU graph. See MpAssetFormat.cpp.
//
// Sizes (with pack(1)):
//   MpAssetHeader    : 4 + 4 + 4 + 4 + 8 + 8 + 8               = 40 bytes
//   MpDagNode        : 16 + 4 + 4 + 4 + 4 + 4                  = 36 bytes
//   MpPageEntry      : 8 + 4 + 4 + 4 + 4 + 4 + 4               = 32 bytes
//   PagePayloadHeader: 4 + 4 + 4 + 4                           = 16 bytes
//   ClusterOnDisk    : 4 + 4 + 4 + 4 + 16 + 12 + 12 + 4 + 4 + 4 + 4 = 72 bytes
//                      wait — that's 72, but we want 76. The explicit `_pad`
//                      plus the cone triplets being stored as f32[3] (12 B
//                      each) keeps alignment checks honest. 4+4+4+4 +
//                      16 + 12 + 12 + 4 + 4 + 4 + 4 = 72. Add 4 B of pad = 76.
// These exact sizes are what M1c will rely on.
inline constexpr usize kMpAssetHeaderSize    = 40;
inline constexpr usize kMpDagNodeSize        = 36;
inline constexpr usize kMpPageEntrySize      = 32;
inline constexpr usize kPagePayloadHeaderSize = 16;
inline constexpr usize kClusterOnDiskSize    = 76;

} // namespace enigma::asset
