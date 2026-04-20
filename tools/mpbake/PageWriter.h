#pragma once

// PageWriter.h
// =============
// Final stage of `enigma-mpbake` (M1c): take the DAG + per-group cluster
// payload and serialize a versioned `.mpa` file. Each page holds one DAG
// group's worth of cluster triangle data, zstd-compressed. The
// `MpAssetHeader` section offsets let `MpAssetReader` jump to the right
// page without walking the stream.
//
// Determinism
// -----------
// Two bakes of the same `DagResult` with the same `PageWriteOptions`
// produce byte-identical `.mpa` files. Iteration over groups uses an
// ordered `std::map` keyed by `parentGroupId` (ordered DAG-node index for
// orphans). Zstd compression at a fixed level is deterministic given the
// same input byte sequence.

#include "DagBuilder.h"

#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>

namespace enigma::mpbake {

// Tunables for the page-writer stage. Defaults match plan §3.M1.
struct PageWriteOptions {
    // Zstd compression level. `0` resolves to zstd's default (3). Stored as
    // int to match zstd's C API. Values outside [zstd_min..zstd_max] are
    // clamped by zstd internally; values that `ZSTD_isError()` rejects cause
    // `PageWriteErrorKind::ZstdError` to surface.
    int zstdCompressionLevel = 0;
};

// Bake-time statistics surfaced to the caller. All sizes are on-disk
// (post-compression) unless labeled otherwise.
struct MpWriteStats {
    std::uint32_t pageCount              = 0u;   // == output MpAssetHeader::pageCount
    std::uint32_t dagNodeCount           = 0u;   // == output MpAssetHeader::dagNodeCount
    std::uint64_t totalCompressedBytes   = 0u;   // sum of MpPageEntry::compressedSize
    std::uint64_t totalDecompressedBytes = 0u;   // sum of MpPageEntry::decompressedSize
    std::uint64_t fileBytes              = 0u;   // final .mpa file size
};

// Classifier for every failure `PageWriter::write` may surface.
enum class PageWriteErrorKind {
    EmptyDag,             // dag.nodes or dag.clusters is empty
    ClusterOverflow,      // a cluster's vertex / triangle counts exceed u32 offsets
    ZstdError,            // zstd_compress returned an error code
    IoError,              // couldn't open / write / close the output file
    InvariantViolation,   // internal bookkeeping drift (e.g. dag node count mismatch)
};

// Rich error payload for write failures.
struct PageWriteError {
    PageWriteErrorKind kind;
    std::string        detail;
};

// Stable string for `PageWriteErrorKind`. Guaranteed never null.
const char* pageWriteErrorKindString(PageWriteErrorKind kind) noexcept;

// PageWriter
// ----------
// Stateless. Construct + `write()`.
class PageWriter {
public:
    PageWriter()  = default;
    ~PageWriter() = default;

    // Serialize `dag` to `outPath`. Creates / truncates the file. On success
    // returns bake stats; on failure returns a rich error payload and leaves
    // the (partial) file on disk — the caller may delete. The M1 exit
    // criterion is a round-trip via `MpAssetReader`.
    std::expected<MpWriteStats, PageWriteError>
    write(const DagResult& dag,
          const std::filesystem::path& outPath,
          const PageWriteOptions& opts = {});
};

} // namespace enigma::mpbake
