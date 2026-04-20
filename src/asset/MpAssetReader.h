#pragma once

// MpAssetReader.h
// ================
// Runtime-side reader for `.mpa` Micropoly Asset files. Opens a file via
// memory map, validates header magic + version + section offsets, and
// exposes a page-fetch API that zstd-decompresses each page on demand.
//
// Contract
// --------
// - `open()` must be called before anything else; holds an OS file handle +
//   mmap'd view for the reader's lifetime.
// - Pure CPU. No Vulkan. The runtime streaming subsystem (M2) consumes
//   PageView byte ranges and uploads to GPU.
// - Not thread-safe to open concurrently; after open, `fetchPage()` is
//   re-entrant as long as each call supplies its own `outDecompressedBuffer`.
// - Destructor releases the file handle + mapping.

#include "asset/MpAssetFormat.h"
#include "core/Types.h"

#include <cstddef>
#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace enigma::asset {

// Classifier for every failure the reader may surface.
enum class MpReadErrorKind {
    FileNotFound,
    IoError,
    FileTooSmall,
    ValidateMagicFailed,
    ValidateVersionFailed,
    InvalidSectionOffsets,
    PageIndexOutOfRange,
    BufferTooSmall,
    ZstdError,
    InvalidPayload,
    NotOpen,
};

// Rich error payload.
struct MpReadError {
    MpReadErrorKind kind;
    std::string     detail;
};

// Stable string for `MpReadErrorKind`. Never null.
const char* mpReadErrorKindString(MpReadErrorKind kind) noexcept;

// View into one decompressed page. The `clusters`, `vertexBlob`, and
// `triangleBlob` spans reference bytes inside the caller-owned decompressed
// buffer passed to `fetchPage()`. Do not let that buffer go out of scope
// while a PageView derived from it is still in use.
struct PageView {
    std::span<const ClusterOnDisk> clusters;
    std::span<const std::byte>     vertexBlob;
    std::span<const std::byte>     triangleBlob;
    u32                            clusterCount = 0u;
    u32                            pageId       = 0u;
};

// MpAssetReader
// -------------
// Value type — not copyable, movable. Construct, `open()`, use, destruct.
class MpAssetReader {
public:
    MpAssetReader()  = default;
    ~MpAssetReader();

    MpAssetReader(const MpAssetReader&)            = delete;
    MpAssetReader& operator=(const MpAssetReader&) = delete;
    MpAssetReader(MpAssetReader&&) noexcept;
    MpAssetReader& operator=(MpAssetReader&&) noexcept;

    // Open the file and map it into memory. Returns an error payload on
    // IO failure, magic/version mismatch, or truncated file.
    std::expected<void, MpReadError> open(const std::filesystem::path& path);

    // Close any held mapping + handles. Safe to call multiple times.
    void close() noexcept;

    // True iff `open()` succeeded since the last `close()`.
    bool isOpen() const noexcept { return mappedBase_ != nullptr; }

    // Cheap re-check that magic + version match what we support.
    bool validate() const noexcept;

    // The loaded header. Valid iff `isOpen()`.
    const MpAssetHeader& header() const noexcept { return header_; }

    // Typed access to DAG nodes / page entries — returns empty spans if
    // not open. Callers must not outlive this reader.
    std::span<const MpDagNode>  dagNodes()   const noexcept;
    std::span<const MpPageEntry> pageTable() const noexcept;

    // M4.5: gather `firstDagNodeIdx` for every page into a dense u32 vector.
    // The runtime uploads this as a DEVICE_LOCAL SSBO (pageId -> global DAG
    // node index of that page's first cluster) so shaders can recover the
    // per-page local cluster index via `globalDagNodeIdx - firstDagNodeIdx`.
    // A plain `std::vector<u32>` is returned (not a zero-copy span over the
    // mmap'd page table) because the mmap stride is 32 B per entry, not
    // 4 B — indexing it as a u32 array would misread. Called exactly once
    // at asset load, so the copy cost is negligible. Empty when not open.
    std::vector<u32> firstDagNodeIndices() const;

    // Runtime-format DAG node for GPU upload. 80 bytes = 5×float4, matching
    // the shader's `MpDagNode` layout in mp_cluster_cull.comp.hlsl::loadDagNode:
    //   float4 m0 = (center.xyz, radius)           — from ClusterOnDisk.boundsSphere
    //   float4 m1 = (coneApex.xyz, coneCutoff)     — from ClusterOnDisk.cone*
    //   float4 m2 = (coneAxis.xyz, asfloat(packed))
    //     where packed = (pageId & 0x00FFFFFF) | (lodLevel << 24)
    //   float4 m3 = (maxError, parentMaxError, 0, 0)
    //     * maxError       = this cluster's world-space simplification error
    //     * parentMaxError = the error of the next-coarser cluster whose
    //                       group produced this one. FLT_MAX for roots so the
    //                       screen-space-error "emit when parent fails" test
    //                       always accepts a root. M4 widening over the
    //                       previous 48 B layout.
    //   float4 m4 = (parentCenter.xyz, 0)          — M4-fix widening over 64 B.
    //     * parentCenter = bounds centre of the coarser parent cluster.
    //                      Nanite's group-coherent LOD rule requires every
    //                      child in a DAG group to project its screen-space
    //                      error at the PARENT'S centre (one parent per group,
    //                      shared by every child), so siblings compute
    //                      identical errSelf / errParent and flip LOD
    //                      together — no cracks or temporal flicker at
    //                      group boundaries. For roots (parentGroupId ==
    //                      UINT32_MAX) we copy the node's own centre so
    //                      the SSE projection stays finite; combined with
    //                      parentMaxError == FLT_MAX the root fallback in
    //                      the cull shader still emits the root cluster.
    //
    // On-disk MpDagNode (36 B) carries bounds/parent/pageId + maxError; cone
    // data lives inside each page's ClusterOnDisk entries (76 B each). The
    // assembler walks the page table, decompresses every page, and joins
    // ClusterOnDisk cone+bounds+error with MpDagNode.pageId + parentGroupId
    // to produce one runtime node per global DAG node. PageWriter guarantees
    // a 1:1 mapping: DAG node at global index (firstDagNodeIdx + i) ↔
    // ClusterOnDisk[i] in that page (see PageWriter.cpp:317 + DagBuilder's
    // stable sort by parentGid).
    struct RuntimeDagNode {
        f32 m0[4];  // center.xyz, radius
        f32 m1[4];  // coneApex.xyz, coneCutoff
        f32 m2[4];  // coneAxis.xyz, asfloat(pageId | lodLevel<<24)
        f32 m3[4];  // maxError, parentMaxError, 0, 0
        f32 m4[4];  // parentCenter.xyz, 0  — group-coherent LOD anchor (M4-fix)
    };

    // Assemble the full 80 B runtime DAG node array (one entry per
    // `MpDagNode` in the on-disk DAG section). Called exactly once at asset
    // load by MicropolyStreaming::attachDagNodeBuffer. Decompresses every
    // page via `fetchPage()` + reuses a single scratch buffer across pages
    // (one allocation per run). Returns empty vector / std::unexpected on
    // bounds mismatch, zstd failure, or not-open.
    //
    // Resulting size: header_.dagNodeCount * sizeof(RuntimeDagNode)
    //                 == dagNodeCount * 80 B.
    std::expected<std::vector<RuntimeDagNode>, MpReadError>
    assembleRuntimeDagNodes() const;

    // Fetch + decompress page `pageId` into `outDecompressedBuffer`. The
    // buffer is resized to the page's decompressed size if needed. On
    // success the returned `PageView` spans point into that buffer.
    std::expected<PageView, MpReadError>
    fetchPage(u32 pageId, std::vector<u8>& outDecompressedBuffer) const;

private:
    // Zero out internal state. Used by close() + move-assign.
    void reset_() noexcept;

    MpAssetHeader header_ {};

    // Raw bytes of the entire file, mapped read-only. We keep an explicit
    // pointer + size because std::span has no ownership semantics.
    const u8*   mappedBase_ = nullptr;
    u64         mappedSize_ = 0u;

    // Platform handles. void* so we don't leak <windows.h> into the header.
    void*       osFileHandle_    = nullptr;  // CreateFileW HANDLE
    void*       osMappingHandle_ = nullptr;  // CreateFileMappingW HANDLE
};

} // namespace enigma::asset
