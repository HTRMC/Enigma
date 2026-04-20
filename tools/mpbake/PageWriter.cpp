// PageWriter.cpp
// ==============
// M1c implementation of the `.mpa` file writer. See PageWriter.h for the
// contract and DagBuilder.h for the upstream DAG shape.
//
// Pipeline
// --------
//   1. Group the DAG nodes by (lodLevel, parentGroupId). Each distinct
//      group becomes exactly one page. Orphaned nodes (parentGroupId ==
//      UINT32_MAX) each get their own page keyed by their own node index
//      so determinism is preserved and no two orphans collide.
//   2. For each page, serialize the clusters' ClusterOnDisk descriptors +
//      concatenated vertex blob (32 B per vertex) + concatenated triangle
//      blob (3 B per triangle) into a single buffer. Prefix the buffer
//      with a PagePayloadHeader.
//   3. zstd_compress the buffer at the caller's compression level. Record
//      payload offset + compressed size + decompressed size in an
//      MpPageEntry.
//   4. Update each DAG node's `pageId` field to point at its page.
//   5. Write the file: header (with placeholder offsets) -> DAG nodes ->
//      page table -> padded payloads. Seek back and patch the header's
//      section offsets.
//
// Hardening
// ---------
// - NaN / Inf sanitization on every f32 field written to disk (we route
//   through `sanitizeFloat` below).
// - Overflow guards on vertex / triangle / page-payload sizes (u32 bounds).
// - Deterministic ordering via `std::map` (ordered) keyed on
//   (lodLevel, parentGroupId) + fallback orphan key.
// - Single-pass compression per page so no internal zstd state carries
//   across pages (determinism + reproducibility across re-bakes).

#include "PageWriter.h"

#include "asset/MpAssetFormat.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <zstd.h>

namespace enigma::mpbake {

const char* pageWriteErrorKindString(PageWriteErrorKind kind) noexcept {
    switch (kind) {
        case PageWriteErrorKind::EmptyDag:            return "EmptyDag";
        case PageWriteErrorKind::ClusterOverflow:     return "ClusterOverflow";
        case PageWriteErrorKind::ZstdError:           return "ZstdError";
        case PageWriteErrorKind::IoError:             return "IoError";
        case PageWriteErrorKind::InvariantViolation:  return "InvariantViolation";
    }
    return "Unknown";
}

namespace {

// Replace NaN / Inf with 0. Finite values pass through unchanged. We use
// this on every f32 field that hits disk so corrupted in-memory state can't
// poison the on-disk format silently.
inline enigma::f32 sanitizeFloat(float v) noexcept {
    return std::isfinite(v) ? v : 0.0f;
}

// Append a trivially-copyable value to a byte vector. Caller-aligned — no
// padding inserted. Uses memcpy so the output stream respects pack(1).
template <typename T>
void append_bytes(std::vector<std::uint8_t>& buf, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>,
        "append_bytes only supports trivially-copyable types");
    const std::size_t before = buf.size();
    buf.resize(before + sizeof(T));
    std::memcpy(buf.data() + before, &value, sizeof(T));
}

// Compose the page-table key for a DAG node. Nodes with a real parent
// group id are grouped together; orphans each get their own page keyed
// by node index so they stay deterministic and distinct.
//
// Key layout: (lodLevel, sortKey) where sortKey is either the
// parentGroupId (for grouped nodes) or kOrphanMarker + nodeIdx (for
// orphans). `kOrphanMarker` is set to a value guaranteed to exceed every
// legitimate parentGroupId (UINT32_MAX's range is avoided by using a
// 64-bit key for sortKey and reserving the top bit for the orphan flag).
struct PageKey {
    std::uint32_t lodLevel;
    std::uint64_t sortKey;

    friend bool operator<(const PageKey& a, const PageKey& b) noexcept {
        if (a.lodLevel != b.lodLevel) return a.lodLevel < b.lodLevel;
        return a.sortKey < b.sortKey;
    }
};

// Orphans map to (kOrphanBit | nodeIdx). Real parent group ids map to
// (parentGroupId). Since parentGroupId is u32, setting the top bit of a
// u64 makes orphan keys strictly greater than any real parent key at the
// same lodLevel — deterministic and collision-free.
constexpr std::uint64_t kOrphanBit = 0x8000'0000'0000'0000ull;

} // namespace

std::expected<MpWriteStats, PageWriteError>
PageWriter::write(const DagResult& dag,
                  const std::filesystem::path& outPath,
                  const PageWriteOptions& opts) {
    using Err = PageWriteError;

    // --- Pre-flight. ---
    if (dag.nodes.empty() || dag.clusters.empty()) {
        return std::unexpected(Err{
            PageWriteErrorKind::EmptyDag,
            std::string{"DagResult is empty (nodes="}
                + std::to_string(dag.nodes.size())
                + " clusters=" + std::to_string(dag.clusters.size())
                + ")",
        });
    }
    if (dag.nodes.size() != dag.clusters.size()) {
        return std::unexpected(Err{
            PageWriteErrorKind::InvariantViolation,
            std::string{"DagResult.nodes.size()="}
                + std::to_string(dag.nodes.size())
                + " != clusters.size()=" + std::to_string(dag.clusters.size()),
        });
    }

    // --- Step 1: group nodes by page key. Every node gets slotted into a
    // page. Within a page, nodes are in ascending node-index order
    // (std::map iteration order + push_back preserves this). ---
    std::map<PageKey, std::vector<std::uint32_t>> pageGroups;
    for (std::uint32_t nodeIdx = 0; nodeIdx < dag.nodes.size(); ++nodeIdx) {
        const DagNode& n = dag.nodes[nodeIdx];
        PageKey key;
        key.lodLevel = n.lodLevel;
        if (n.parentGroupId == UINT32_MAX) {
            key.sortKey = kOrphanBit | static_cast<std::uint64_t>(nodeIdx);
        } else {
            key.sortKey = static_cast<std::uint64_t>(n.parentGroupId);
        }
        pageGroups[key].push_back(nodeIdx);
    }

    // --- Step 2: build per-page payload buffers + compressed blobs. ---
    // We compute MpPageEntry fields here (except payloadByteOffset, which
    // depends on the file layout resolved later).

    // Fix G: guard BEFORE cast so the overflow check is not vacuous. The
    // previous code narrowed to u32 first, then checked the original size —
    // the check could never fire because pageGroups.size() was already
    // truncated in the assignment.
    if (pageGroups.size() > std::numeric_limits<std::uint32_t>::max()) {
        return std::unexpected(Err{
            PageWriteErrorKind::ClusterOverflow,
            std::string{"page count exceeds u32: "}
                + std::to_string(pageGroups.size()),
        });
    }
    const std::uint32_t pageCount =
        static_cast<std::uint32_t>(pageGroups.size());

    std::vector<asset::MpPageEntry>           pageEntries;
    std::vector<std::vector<std::uint8_t>>    compressedBlobs;
    pageEntries.reserve(pageCount);
    compressedBlobs.reserve(pageCount);

    // We also need to patch each DAG node's `pageId` before writing the DAG
    // section. Materialize the on-disk DAG node array now with pageId set.
    std::vector<asset::MpDagNode> onDiskNodes(dag.nodes.size());
    {
        // Initialize from DagResult. We'll set pageId below as pages are built.
        for (std::size_t i = 0; i < dag.nodes.size(); ++i) {
            const DagNode& src = dag.nodes[i];
            asset::MpDagNode& dst = onDiskNodes[i];
            dst.boundsSphere[0] = sanitizeFloat(src.boundsSphere.x);
            dst.boundsSphere[1] = sanitizeFloat(src.boundsSphere.y);
            dst.boundsSphere[2] = sanitizeFloat(src.boundsSphere.z);
            dst.boundsSphere[3] = sanitizeFloat(src.boundsSphere.w);
            dst.maxError        = sanitizeFloat(src.maxError);
            dst.parentGroupId   = src.parentGroupId;
            dst.firstChildNode  = src.firstChildNode;
            dst.childCount      = src.childCount;
            dst.pageId          = UINT32_MAX;   // patched below
        }
    }

    std::uint32_t pageIdCounter = 0u;
    std::uint64_t totalCompressed   = 0u;
    std::uint64_t totalDecompressed = 0u;

    for (const auto& [key, nodeIndices] : pageGroups) {
        if (nodeIndices.empty()) continue;  // defensive; std::map entries
                                            // always have >=1 element by
                                            // construction.

        // Build the decompressed payload buffer:
        //   [PagePayloadHeader]
        //   [ClusterOnDisk * N]
        //   [vertex blob]
        //   [triangle blob]
        std::vector<std::uint8_t> payload;

        // Reserve a ballpark so push_back avalanches stay cheap. N clusters
        // of up to 128 verts / 128 tris each = ~5 KB worst case.
        payload.reserve(
            asset::kPagePayloadHeaderSize +
            nodeIndices.size() * asset::kClusterOnDiskSize +
            nodeIndices.size() * 128u * asset::kMpVertexStride +
            nodeIndices.size() * 128u * 3u);

        asset::PagePayloadHeader hdr{};
        hdr.clusterCount = static_cast<std::uint32_t>(nodeIndices.size());
        hdr.version      = asset::kMpPagePayloadVersion;
        hdr._pad0        = 0u;
        hdr._pad1        = 0u;
        append_bytes(payload, hdr);

        // Reserve a contiguous slice for the ClusterOnDisk array. We fill
        // the actual offsets after the vertex/triangle concatenation is
        // done (because offsets are byte-relative into blobs that come
        // later in the same buffer).
        const std::size_t cdArrayOffset = payload.size();
        payload.resize(cdArrayOffset +
            nodeIndices.size() * asset::kClusterOnDiskSize);

        // --- Concatenate vertex blob (32 B per vertex). ---
        const std::size_t vertexBlobStart = payload.size();
        for (std::uint32_t nodeIdx : nodeIndices) {
            const DagNode& n = dag.nodes[nodeIdx];
            const ClusterData& c = dag.clusters[n.clusterId];
            if (c.positions.size() != c.normals.size() ||
                c.positions.size() != c.uvs.size()) {
                return std::unexpected(Err{
                    PageWriteErrorKind::InvariantViolation,
                    std::string{"cluster "} + std::to_string(n.clusterId)
                        + " stream size mismatch: pos="
                        + std::to_string(c.positions.size())
                        + " nrm=" + std::to_string(c.normals.size())
                        + " uv="  + std::to_string(c.uvs.size()),
                });
            }
            if (c.positions.size() > 0xFFFFu) {
                return std::unexpected(Err{
                    PageWriteErrorKind::ClusterOverflow,
                    std::string{"cluster "} + std::to_string(n.clusterId)
                        + " vertex count "
                        + std::to_string(c.positions.size())
                        + " exceeds u16 bound",
                });
            }
            for (std::size_t v = 0; v < c.positions.size(); ++v) {
                const float px = sanitizeFloat(c.positions[v].x);
                const float py = sanitizeFloat(c.positions[v].y);
                const float pz = sanitizeFloat(c.positions[v].z);
                const float nx = sanitizeFloat(c.normals[v].x);
                const float ny = sanitizeFloat(c.normals[v].y);
                const float nz = sanitizeFloat(c.normals[v].z);
                const float uu = sanitizeFloat(c.uvs[v].x);
                const float uv = sanitizeFloat(c.uvs[v].y);
                append_bytes(payload, px);
                append_bytes(payload, py);
                append_bytes(payload, pz);
                append_bytes(payload, nx);
                append_bytes(payload, ny);
                append_bytes(payload, nz);
                append_bytes(payload, uu);
                append_bytes(payload, uv);
            }
        }

        // --- Concatenate triangle blob (3 bytes per triangle, u8 local
        // indices into the cluster's local vertex table). ---
        const std::size_t triangleBlobStart = payload.size();
        for (std::uint32_t nodeIdx : nodeIndices) {
            const DagNode& n = dag.nodes[nodeIdx];
            const ClusterData& c = dag.clusters[n.clusterId];
            if (c.triangles.size() % 3u != 0u) {
                return std::unexpected(Err{
                    PageWriteErrorKind::InvariantViolation,
                    std::string{"cluster "} + std::to_string(n.clusterId)
                        + " triangle index count not multiple of 3: "
                        + std::to_string(c.triangles.size()),
                });
            }
            const std::size_t triCount = c.triangles.size() / 3u;
            if (triCount > 0xFFFFu) {
                return std::unexpected(Err{
                    PageWriteErrorKind::ClusterOverflow,
                    std::string{"cluster "} + std::to_string(n.clusterId)
                        + " triangle count " + std::to_string(triCount)
                        + " exceeds u16 bound",
                });
            }
            if (!c.triangles.empty()) {
                const std::size_t before = payload.size();
                payload.resize(before + c.triangles.size());
                std::memcpy(payload.data() + before,
                    c.triangles.data(), c.triangles.size());
            }
        }

        // --- Fill in ClusterOnDisk entries with final offsets. ---
        {
            std::size_t perClusterVertexCursor   = 0u;
            std::size_t perClusterTriangleCursor = 0u;
            for (std::size_t i = 0; i < nodeIndices.size(); ++i) {
                const std::uint32_t nodeIdx = nodeIndices[i];
                const DagNode& n = dag.nodes[nodeIdx];
                const ClusterData& c = dag.clusters[n.clusterId];

                const std::size_t vBytes =
                    c.positions.size() * asset::kMpVertexStride;
                const std::size_t tBytes = c.triangles.size();

                // Offsets are relative to their respective blobs. The
                // writer records `perClusterVertexCursor` before appending;
                // the reader recovers each cluster's blob byte-slice by
                // following these offsets.
                if (perClusterVertexCursor   > std::numeric_limits<std::uint32_t>::max() ||
                    perClusterTriangleCursor > std::numeric_limits<std::uint32_t>::max()) {
                    return std::unexpected(Err{
                        PageWriteErrorKind::ClusterOverflow,
                        std::string{"per-page blob offset exceeds u32: v="}
                            + std::to_string(perClusterVertexCursor)
                            + " t=" + std::to_string(perClusterTriangleCursor),
                    });
                }

                asset::ClusterOnDisk cd{};
                cd.vertexCount   = static_cast<std::uint32_t>(c.positions.size());
                cd.triangleCount = static_cast<std::uint32_t>(c.triangles.size() / 3u);
                cd.vertexOffset  = static_cast<std::uint32_t>(perClusterVertexCursor);
                cd.triangleOffset= static_cast<std::uint32_t>(perClusterTriangleCursor);
                cd.boundsSphere[0] = sanitizeFloat(c.boundsSphere.x);
                cd.boundsSphere[1] = sanitizeFloat(c.boundsSphere.y);
                cd.boundsSphere[2] = sanitizeFloat(c.boundsSphere.z);
                cd.boundsSphere[3] = sanitizeFloat(c.boundsSphere.w);
                cd.coneApex[0] = sanitizeFloat(c.coneApex.x);
                cd.coneApex[1] = sanitizeFloat(c.coneApex.y);
                cd.coneApex[2] = sanitizeFloat(c.coneApex.z);
                cd.coneAxis[0] = sanitizeFloat(c.coneAxis.x);
                cd.coneAxis[1] = sanitizeFloat(c.coneAxis.y);
                cd.coneAxis[2] = sanitizeFloat(c.coneAxis.z);
                cd.coneCutoff             = sanitizeFloat(c.coneCutoff);
                cd.maxSimplificationError = sanitizeFloat(c.maxSimplificationError);
                cd.dagLodLevel            = c.dagLodLevel;
                cd.materialIndex          = c.materialIndex;
                cd._pad1                  = 0u;

                const std::size_t dst = cdArrayOffset + i * asset::kClusterOnDiskSize;
                std::memcpy(payload.data() + dst, &cd, sizeof(cd));

                perClusterVertexCursor   += vBytes;
                perClusterTriangleCursor += tBytes;
            }

            // Sanity: cursor end must match blob end.
            if (vertexBlobStart + perClusterVertexCursor != triangleBlobStart) {
                return std::unexpected(Err{
                    PageWriteErrorKind::InvariantViolation,
                    std::string{"vertex blob cursor drift (expected end "}
                        + std::to_string(triangleBlobStart)
                        + " got " + std::to_string(vertexBlobStart + perClusterVertexCursor)
                        + ")",
                });
            }
            if (triangleBlobStart + perClusterTriangleCursor != payload.size()) {
                return std::unexpected(Err{
                    PageWriteErrorKind::InvariantViolation,
                    std::string{"triangle blob cursor drift (expected end "}
                        + std::to_string(payload.size())
                        + " got " + std::to_string(triangleBlobStart + perClusterTriangleCursor)
                        + ")",
                });
            }
        }

        // Fix A: enforce per-page decompressed size cap at bake time so a
        // legitimately-produced file can never exceed the reader's guard.
        if (payload.size() > asset::kMpMaxPageDecompressedBytes) {
            return std::unexpected(Err{
                PageWriteErrorKind::ClusterOverflow,
                std::string{"page payload size "} + std::to_string(payload.size())
                    + " exceeds per-page cap "
                    + std::to_string(asset::kMpMaxPageDecompressedBytes),
            });
        }
        if (payload.size() > std::numeric_limits<std::uint32_t>::max()) {
            return std::unexpected(Err{
                PageWriteErrorKind::ClusterOverflow,
                std::string{"page payload "} + std::to_string(payload.size())
                    + " exceeds u32",
            });
        }

        // --- zstd compress. ---
        const std::size_t bound = ZSTD_compressBound(payload.size());
        std::vector<std::uint8_t> compressed(bound);
        const std::size_t compressedSz = ZSTD_compress(
            compressed.data(), compressed.size(),
            payload.data(),    payload.size(),
            opts.zstdCompressionLevel);
        if (ZSTD_isError(compressedSz)) {
            return std::unexpected(Err{
                PageWriteErrorKind::ZstdError,
                std::string{"ZSTD_compress failed: "}
                    + ZSTD_getErrorName(compressedSz),
            });
        }
        if (compressedSz > std::numeric_limits<std::uint32_t>::max()) {
            return std::unexpected(Err{
                PageWriteErrorKind::ClusterOverflow,
                std::string{"compressed page size "}
                    + std::to_string(compressedSz) + " exceeds u32",
            });
        }
        compressed.resize(compressedSz);

        // --- Emit MpPageEntry. payloadByteOffset is filled in during file
        // layout resolution below; we stash the decompressedSize etc. now. ---
        asset::MpPageEntry entry{};
        entry.payloadByteOffset = 0u;  // patched in file-layout pass below
        entry.compressedSize    = static_cast<std::uint32_t>(compressedSz);
        entry.decompressedSize  = static_cast<std::uint32_t>(payload.size());
        entry.clusterCount      = static_cast<std::uint32_t>(nodeIndices.size());
        entry.firstDagNodeIdx   = nodeIndices.front();
        // Store the DAG group id in `groupId`. For orphans we store the
        // marker bit-stripped sortKey for debug traceability.
        entry.groupId           = static_cast<std::uint32_t>(
            key.sortKey & ~kOrphanBit);
        entry._pad              = 0u;

        // Patch each member node's pageId.
        const std::uint32_t pid = pageIdCounter++;
        for (std::uint32_t nodeIdx : nodeIndices) {
            onDiskNodes[nodeIdx].pageId = pid;
        }

        totalCompressed   += compressedSz;
        totalDecompressed += payload.size();

        pageEntries.push_back(entry);
        compressedBlobs.push_back(std::move(compressed));
    }

    if (pageEntries.size() != pageCount) {
        return std::unexpected(Err{
            PageWriteErrorKind::InvariantViolation,
            std::string{"built "} + std::to_string(pageEntries.size())
                + " pages, expected " + std::to_string(pageCount),
        });
    }

    // --- Step 3: resolve file layout. Write the file. ---
    // Layout:
    //   offset 0                         : MpAssetHeader
    //   dagByteOffset    = 40            : MpDagNode[dagNodeCount]
    //   pagesByteOffset  = + dag bytes   : MpPageEntry[pageCount]
    //   boundsByteOffset = + page table  : (empty in v1 — reserved)
    //   payload region   = aligned 16   : zstd blobs, each padded to 16
    //
    // We compute all section offsets up front, then write in stream order.

    const std::uint64_t headerSize   = asset::kMpAssetHeaderSize;
    const std::uint64_t dagSize      = static_cast<std::uint64_t>(
        onDiskNodes.size()) * asset::kMpDagNodeSize;
    const std::uint64_t pageTblSize  = static_cast<std::uint64_t>(
        pageEntries.size()) * asset::kMpPageEntrySize;

    const std::uint64_t dagOffset    = headerSize;
    const std::uint64_t pagesOffset  = dagOffset + dagSize;
    const std::uint64_t boundsOffset = pagesOffset + pageTblSize;

    // Align payload region to kMpPagePayloadAlignment.
    auto alignUp = [](std::uint64_t value, std::uint64_t alignment) -> std::uint64_t {
        if (alignment <= 1u) return value;
        const std::uint64_t mask = alignment - 1u;
        return (value + mask) & ~mask;
    };

    std::uint64_t payloadCursor = alignUp(boundsOffset,
        asset::kMpPagePayloadAlignment);

    // Patch each page entry with its final on-disk offset; the payload
    // itself lives at `payloadCursor`, and each subsequent page is
    // 16-byte-aligned. This keeps the reader's slice-pointer SIMD-friendly.
    for (std::size_t p = 0; p < pageEntries.size(); ++p) {
        pageEntries[p].payloadByteOffset = payloadCursor;
        payloadCursor += compressedBlobs[p].size();
        payloadCursor  = alignUp(payloadCursor,
            asset::kMpPagePayloadAlignment);
    }
    const std::uint64_t finalFileSize = payloadCursor;

    // --- Assemble the header now that all offsets are known. ---
    asset::MpAssetHeader header{};
    std::memcpy(header.magic, asset::kMpAssetMagic, sizeof(header.magic));
    header.version          = asset::kMpAssetVersion;
    header.dagNodeCount     = static_cast<std::uint32_t>(onDiskNodes.size());
    header.pageCount        = static_cast<std::uint32_t>(pageEntries.size());
    header.pagesByteOffset  = pagesOffset;
    header.dagByteOffset    = dagOffset;
    header.boundsByteOffset = boundsOffset;

    // --- Open the output file (binary, truncate). ---
    // std::ofstream with binary mode on Windows avoids the CRLF translation
    // trap; the default ios_base::trunc replaces any pre-existing file.
    std::ofstream out(outPath,
        std::ios::binary | std::ios::trunc | std::ios::out);
    if (!out.is_open()) {
        return std::unexpected(Err{
            PageWriteErrorKind::IoError,
            std::string{"failed to open output file: "} + outPath.string(),
        });
    }

    auto writeBytes = [&](const void* p, std::uint64_t n) -> bool {
        if (n == 0u) return true;
        out.write(reinterpret_cast<const char*>(p), static_cast<std::streamsize>(n));
        return out.good();
    };

    // Header.
    if (!writeBytes(&header, sizeof(header))) {
        return std::unexpected(Err{
            PageWriteErrorKind::IoError,
            std::string{"write header failed"},
        });
    }

    // DAG nodes.
    if (!onDiskNodes.empty()) {
        if (!writeBytes(onDiskNodes.data(),
            static_cast<std::uint64_t>(onDiskNodes.size()) * asset::kMpDagNodeSize)) {
            return std::unexpected(Err{
                PageWriteErrorKind::IoError,
                std::string{"write DAG nodes failed"},
            });
        }
    }

    // Page table.
    if (!pageEntries.empty()) {
        if (!writeBytes(pageEntries.data(),
            static_cast<std::uint64_t>(pageEntries.size()) * asset::kMpPageEntrySize)) {
            return std::unexpected(Err{
                PageWriteErrorKind::IoError,
                std::string{"write page table failed"},
            });
        }
    }

    // The bounds section is empty in v1. Pad bytes up to the first payload
    // offset with zeros for determinism.
    {
        const std::uint64_t currentPos = headerSize + dagSize + pageTblSize;
        if (pageEntries.empty()) {
            // No payloads — boundsOffset == currentPos == finalFileSize.
        } else {
            const std::uint64_t firstPayloadOff = pageEntries.front().payloadByteOffset;
            if (firstPayloadOff < currentPos) {
                return std::unexpected(Err{
                    PageWriteErrorKind::InvariantViolation,
                    std::string{"first payload offset "}
                        + std::to_string(firstPayloadOff)
                        + " before current stream position "
                        + std::to_string(currentPos),
                });
            }
            const std::uint64_t padBytes = firstPayloadOff - currentPos;
            if (padBytes > 0u) {
                static const std::array<std::uint8_t, 16> zeroPad{};
                std::uint64_t remaining = padBytes;
                while (remaining > 0u) {
                    const std::uint64_t chunk =
                        std::min<std::uint64_t>(remaining, zeroPad.size());
                    if (!writeBytes(zeroPad.data(), chunk)) {
                        return std::unexpected(Err{
                            PageWriteErrorKind::IoError,
                            std::string{"write bounds pad failed"},
                        });
                    }
                    remaining -= chunk;
                }
            }
        }
    }

    // Payloads with 16-byte alignment padding between them.
    for (std::size_t p = 0; p < compressedBlobs.size(); ++p) {
        if (!writeBytes(compressedBlobs[p].data(),
            static_cast<std::uint64_t>(compressedBlobs[p].size()))) {
            return std::unexpected(Err{
                PageWriteErrorKind::IoError,
                std::string{"write page payload "} + std::to_string(p) + " failed",
            });
        }
        // Alignment padding after this page (except potentially after the
        // last page — we still pad so the final file size equals
        // `finalFileSize` deterministically).
        const std::uint64_t afterBlob =
            pageEntries[p].payloadByteOffset + compressedBlobs[p].size();
        const std::uint64_t nextBoundary =
            (p + 1u < compressedBlobs.size())
                ? pageEntries[p + 1u].payloadByteOffset
                : finalFileSize;
        if (nextBoundary < afterBlob) {
            return std::unexpected(Err{
                PageWriteErrorKind::InvariantViolation,
                std::string{"payload overlap page "} + std::to_string(p),
            });
        }
        const std::uint64_t padBytes = nextBoundary - afterBlob;
        if (padBytes > 0u) {
            static const std::array<std::uint8_t, 16> zeroPad{};
            std::uint64_t remaining = padBytes;
            while (remaining > 0u) {
                const std::uint64_t chunk =
                    std::min<std::uint64_t>(remaining, zeroPad.size());
                if (!writeBytes(zeroPad.data(), chunk)) {
                    return std::unexpected(Err{
                        PageWriteErrorKind::IoError,
                        std::string{"write payload pad failed"},
                    });
                }
                remaining -= chunk;
            }
        }
    }

    out.flush();
    if (!out.good()) {
        return std::unexpected(Err{
            PageWriteErrorKind::IoError,
            std::string{"flush failed"},
        });
    }
    out.close();
    if (out.fail()) {
        return std::unexpected(Err{
            PageWriteErrorKind::IoError,
            std::string{"close failed"},
        });
    }

    MpWriteStats stats{};
    stats.pageCount              = pageCount;
    stats.dagNodeCount           = header.dagNodeCount;
    stats.totalCompressedBytes   = totalCompressed;
    stats.totalDecompressedBytes = totalDecompressed;
    stats.fileBytes              = finalFileSize;
    return stats;
}

} // namespace enigma::mpbake
