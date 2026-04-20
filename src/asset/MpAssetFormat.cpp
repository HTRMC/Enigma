// MpAssetFormat.cpp
// =================
// Compile-time layout assertions for the MpAssetFormat.h on-disk schema.
// No runtime code — all the interesting bits are `static_assert` tripwires
// that fire if an ABI change (or an accidental dropped `#pragma pack`) would
// silently corrupt the .mpa file format.

#include "asset/MpAssetFormat.h"

#include <bit>

namespace enigma::asset {

static_assert(sizeof(MpAssetHeader) == kMpAssetHeaderSize,
              "MpAssetHeader on-disk size drifted — version-bump the format");
static_assert(sizeof(MpDagNode) == kMpDagNodeSize,
              "MpDagNode on-disk size drifted — version-bump the format");
static_assert(sizeof(MpPageEntry) == kMpPageEntrySize,
              "MpPageEntry on-disk size drifted — version-bump the format");
static_assert(sizeof(PagePayloadHeader) == kPagePayloadHeaderSize,
              "PagePayloadHeader on-disk size drifted — version-bump the format");
static_assert(sizeof(ClusterOnDisk) == kClusterOnDiskSize,
              "ClusterOnDisk on-disk size drifted — version-bump the format");

// Basic alignment sanity. With `#pragma pack(push, 1)` these are all 1 — we
// assert that explicitly so nobody reaches for alignas(...) without a
// coordinated version bump.
static_assert(alignof(MpAssetHeader) == 1,
              "MpAssetHeader must be packed — lost #pragma pack");
static_assert(alignof(MpDagNode) == 1,
              "MpDagNode must be packed — lost #pragma pack");
static_assert(alignof(MpPageEntry) == 1,
              "MpPageEntry must be packed — lost #pragma pack");
static_assert(alignof(PagePayloadHeader) == 1,
              "PagePayloadHeader must be packed — lost #pragma pack");
static_assert(alignof(ClusterOnDisk) == 1,
              "ClusterOnDisk must be packed — lost #pragma pack");

// .mpa v1 is a little-endian format. Every multi-byte field on disk is LE.
// Catch the first cross-compile to a big-endian target at compile time
// rather than silently producing garbage at read time.
static_assert(std::endian::native == std::endian::little,
              "MPA1 layout is little-endian — big-endian targets need a byteswap pass on read/write");

} // namespace enigma::asset
