// Unit test for the runtime MpAssetReader class (M1c).
// Three narrow cases beyond what the writer round-trip already covers:
//   1) Non-existent file: open("nonexistent.mpa") -> FileNotFound.
//   2) Truncated file: open a file shorter than sizeof(MpAssetHeader)
//      -> FileTooSmall.
//   3) Valid file: bake DamagedHelmet to a tmp .mpa, open, verify
//      validate() returns true and header dagNodeCount > 0.
//
// Plain main, printf output, exit 0 on pass.

#include "asset/MpAssetFormat.h"
#include "asset/MpAssetReader.h"

#include "mpbake/ClusterBuilder.h"
#include "mpbake/DagBuilder.h"
#include "mpbake/GltfIngest.h"
#include "mpbake/PageWriter.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <span>
#include <string>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;

using enigma::asset::MpAssetReader;
using enigma::asset::MpReadErrorKind;
using enigma::asset::mpReadErrorKindString;
using enigma::mpbake::ClusterBuildOptions;
using enigma::mpbake::ClusterBuilder;
using enigma::mpbake::ClusterData;
using enigma::mpbake::DagBuildOptions;
using enigma::mpbake::DagBuilder;
using enigma::mpbake::GltfIngest;
using enigma::mpbake::PageWriter;
using enigma::mpbake::PageWriteOptions;

namespace {

fs::path locateDamagedHelmet(const char* argv0) {
    std::error_code ec;
    fs::path start = argv0 ? fs::absolute(argv0, ec).parent_path() : fs::current_path(ec);
    if (ec) start = fs::current_path(ec);
    for (int i = 0; i < 6 && !start.empty(); ++i) {
        fs::path candidate = start / "assets" / "DamagedHelmet.glb";
        if (fs::exists(candidate, ec)) return candidate;
        if (start == start.parent_path()) break;
        start = start.parent_path();
    }
    fs::path cwd = fs::current_path(ec) / "assets" / "DamagedHelmet.glb";
    if (fs::exists(cwd, ec)) return cwd;
    return {};
}

fs::path tmpPath(const std::string& tag) {
    std::error_code ec;
    const fs::path base = fs::temp_directory_path(ec);
    if (ec || base.empty()) return fs::path{"."} / ("mp_asset_reader_test_" + tag + ".mpa");
    return base / ("mp_asset_reader_test_" + tag + ".mpa");
}

bool testNonExistent() {
    MpAssetReader reader;
    const fs::path bogus = tmpPath("definitely_absent_do_not_create_me_please_xyzzy");
    std::error_code ec;
    fs::remove(bogus, ec);
    auto res = reader.open(bogus);
    if (res.has_value()) {
        std::fprintf(stderr,
            "[mp_asset_reader_test] case 1 FAIL: expected error, got success\n");
        return false;
    }
    if (res.error().kind != MpReadErrorKind::FileNotFound) {
        std::fprintf(stderr,
            "[mp_asset_reader_test] case 1 FAIL: expected FileNotFound, got %s\n",
            mpReadErrorKindString(res.error().kind));
        return false;
    }
    std::printf("[mp_asset_reader_test] case 1 PASS: FileNotFound surfaced as expected.\n");
    return true;
}

bool testTruncated() {
    const fs::path p = tmpPath("truncated");
    {
        std::ofstream f(p, std::ios::binary | std::ios::trunc);
        if (!f.is_open()) {
            std::fprintf(stderr,
                "[mp_asset_reader_test] case 2 FAIL: could not create tmp file\n");
            return false;
        }
        // Write only 3 bytes — well short of sizeof(MpAssetHeader).
        const char junk[3] = {'M', 'P', 'A'};
        f.write(junk, 3);
    }
    MpAssetReader reader;
    auto res = reader.open(p);
    std::error_code ec;
    fs::remove(p, ec);
    if (res.has_value()) {
        std::fprintf(stderr,
            "[mp_asset_reader_test] case 2 FAIL: expected error, got success\n");
        return false;
    }
    if (res.error().kind != MpReadErrorKind::FileTooSmall) {
        std::fprintf(stderr,
            "[mp_asset_reader_test] case 2 FAIL: expected FileTooSmall, got %s\n",
            mpReadErrorKindString(res.error().kind));
        return false;
    }
    std::printf("[mp_asset_reader_test] case 2 PASS: FileTooSmall surfaced as expected.\n");
    return true;
}

// Bakes DamagedHelmet to `out` using the offline pipeline. Returns true on
// success. Used by both the valid-file case and the firstDagNodeIdx
// corruption case. Kept local to this TU so we don't leak mpbake types.
bool bakeDamagedHelmet(const fs::path& asset, const fs::path& out) {
    GltfIngest ingest;
    auto ingestRes = ingest.load(asset);
    if (!ingestRes.has_value()) return false;
    ClusterBuilder cb;
    auto clusterRes = cb.build(*ingestRes, ClusterBuildOptions{});
    if (!clusterRes.has_value()) return false;
    DagBuilder db;
    auto dagRes = db.build(std::span<const ClusterData>(*clusterRes), DagBuildOptions{});
    if (!dagRes.has_value()) return false;
    PageWriter writer;
    auto wrRes = writer.write(*dagRes, out, PageWriteOptions{});
    return wrRes.has_value();
}

bool testValid(const fs::path& asset) {
    // Build DagResult and bake.
    GltfIngest ingest;
    auto ingestRes = ingest.load(asset);
    if (!ingestRes.has_value()) {
        std::fprintf(stderr,
            "[mp_asset_reader_test] case 3 FAIL: ingest\n");
        return false;
    }
    ClusterBuilder cb;
    auto clusterRes = cb.build(*ingestRes, ClusterBuildOptions{});
    if (!clusterRes.has_value()) {
        std::fprintf(stderr,
            "[mp_asset_reader_test] case 3 FAIL: cluster\n");
        return false;
    }
    DagBuilder db;
    auto dagRes = db.build(std::span<const ClusterData>(*clusterRes), DagBuildOptions{});
    if (!dagRes.has_value()) {
        std::fprintf(stderr,
            "[mp_asset_reader_test] case 3 FAIL: dag\n");
        return false;
    }

    const fs::path out = tmpPath("valid");
    {
        PageWriter writer;
        auto res = writer.write(*dagRes, out, PageWriteOptions{});
        if (!res.has_value()) {
            std::fprintf(stderr,
                "[mp_asset_reader_test] case 3 FAIL: write: %s\n",
                res.error().detail.c_str());
            return false;
        }
    }

    MpAssetReader reader;
    auto openRes = reader.open(out);
    std::error_code ec;
    if (!openRes.has_value()) {
        std::fprintf(stderr,
            "[mp_asset_reader_test] case 3 FAIL: open: %s: %s\n",
            mpReadErrorKindString(openRes.error().kind),
            openRes.error().detail.c_str());
        fs::remove(out, ec);
        return false;
    }
    if (!reader.validate()) {
        std::fprintf(stderr,
            "[mp_asset_reader_test] case 3 FAIL: validate() returned false\n");
        fs::remove(out, ec);
        return false;
    }
    if (reader.header().dagNodeCount == 0u || reader.header().pageCount == 0u) {
        std::fprintf(stderr,
            "[mp_asset_reader_test] case 3 FAIL: header counts zero "
            "(dagNodes=%u pages=%u)\n",
            reader.header().dagNodeCount, reader.header().pageCount);
        fs::remove(out, ec);
        return false;
    }

    std::printf(
        "[mp_asset_reader_test] case 3 PASS: valid file: dagNodes=%u pages=%u\n",
        reader.header().dagNodeCount, reader.header().pageCount);

    // Must close the reader before we can delete the mmap'd file on Windows.
    reader.close();
    fs::remove(out, ec);
    return true;
}

// M1c security fold-in case: corrupt one MpPageEntry's firstDagNodeIdx to
// UINT32_MAX and confirm open() surfaces InvalidSectionOffsets. This proves
// the bounds check added in MpAssetReader::open() for security HIGH #3.
bool testCorruptFirstDagNodeIdx(const fs::path& asset) {
    const fs::path out = tmpPath("bad_firstdagidx");
    std::error_code ec;
    fs::remove(out, ec);
    if (!bakeDamagedHelmet(asset, out)) {
        std::fprintf(stderr,
            "[mp_asset_reader_test] case 4 FAIL: bake for corruption setup\n");
        return false;
    }

    // Read the header so we know where the page table lives. We open
    // the raw bytes, not the reader, so we can patch in place.
    enigma::asset::MpAssetHeader hdr{};
    {
        std::ifstream f(out, std::ios::binary);
        if (!f.is_open()) {
            std::fprintf(stderr,
                "[mp_asset_reader_test] case 4 FAIL: reopen for read\n");
            fs::remove(out, ec);
            return false;
        }
        f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
        if (!f) {
            std::fprintf(stderr,
                "[mp_asset_reader_test] case 4 FAIL: header read\n");
            fs::remove(out, ec);
            return false;
        }
    }
    if (hdr.pageCount == 0u) {
        std::fprintf(stderr,
            "[mp_asset_reader_test] case 4 FAIL: pageCount is zero in baked file\n");
        fs::remove(out, ec);
        return false;
    }

    // Patch page 0's firstDagNodeIdx to UINT32_MAX. The layout is:
    //   MpPageEntry { u64 payloadByteOffset; u32 compressedSize;
    //                 u32 decompressedSize; u32 clusterCount;
    //                 u32 firstDagNodeIdx; u32 groupId; u32 _pad; }
    // -> firstDagNodeIdx lives at offset 8 + 4 + 4 + 4 = 20 bytes
    // inside the entry.
    constexpr std::streamoff kFirstDagIdxFieldOffset = 8 + 4 + 4 + 4;
    const std::streamoff patchPos = static_cast<std::streamoff>(hdr.pagesByteOffset)
                                  + kFirstDagIdxFieldOffset;
    {
        std::fstream f(out, std::ios::binary | std::ios::in | std::ios::out);
        if (!f.is_open()) {
            std::fprintf(stderr,
                "[mp_asset_reader_test] case 4 FAIL: reopen for patch\n");
            fs::remove(out, ec);
            return false;
        }
        f.seekp(patchPos);
        const std::uint32_t badIdx = 0xFFFFFFFFu;
        f.write(reinterpret_cast<const char*>(&badIdx), sizeof(badIdx));
        if (!f) {
            std::fprintf(stderr,
                "[mp_asset_reader_test] case 4 FAIL: patch write\n");
            fs::remove(out, ec);
            return false;
        }
    }

    MpAssetReader reader;
    auto res = reader.open(out);
    fs::remove(out, ec);
    if (res.has_value()) {
        std::fprintf(stderr,
            "[mp_asset_reader_test] case 4 FAIL: expected error, got success\n");
        return false;
    }
    if (res.error().kind != MpReadErrorKind::InvalidSectionOffsets) {
        std::fprintf(stderr,
            "[mp_asset_reader_test] case 4 FAIL: expected InvalidSectionOffsets, got %s (%s)\n",
            mpReadErrorKindString(res.error().kind),
            res.error().detail.c_str());
        return false;
    }
    std::printf(
        "[mp_asset_reader_test] case 4 PASS: bad firstDagNodeIdx caught by open().\n");
    return true;
}

// Adversarial vertexOffset fuzz gap: crafting a valid zstd-compressed page
// with a poisoned ClusterOnDisk::vertexOffset byte-for-byte requires
// re-running the zstd encoder end-to-end against a bespoke buffer. That's
// doable but fiddly enough that M2.1 does not ship a synthetic-file test
// for it. The per-cluster (vertexOffset + vertexCount*kMpVertexStride)
// <= totalVertexBytes check in MpAssetReader::fetchPage() is the guard;
// future fuzz harnesses (M3+) will cover this exhaustively via random
// mutation of a known-good .mpa. This test intentionally omitted.

} // namespace

int main(int argc, char** argv) {
    const fs::path asset = locateDamagedHelmet(argc > 0 ? argv[0] : nullptr);
    if (asset.empty()) {
        std::fprintf(stderr,
            "[mp_asset_reader_test] FAIL: could not locate assets/DamagedHelmet.glb\n");
        return 1;
    }
    bool ok = true;
    ok &= testNonExistent();
    ok &= testTruncated();
    ok &= testValid(asset);
    ok &= testCorruptFirstDagNodeIdx(asset);
    if (!ok) {
        std::fprintf(stderr, "[mp_asset_reader_test] FAILED\n");
        return 1;
    }
    std::printf("[mp_asset_reader_test] All cases passed.\n");
    return 0;
}
