// Unit test for the offline `enigma-mpbake` PageWriter stage (M1c) — the
// final piece of M1. Four cases:
//   1) Round-trip on DamagedHelmet: ingest -> cluster -> DAG -> write ->
//      MpAssetReader::open -> iterate all pages -> decompress each ->
//      verify cluster geometry (positions/normals/uvs/triangles) is
//      byte-identical to the in-memory DagResult.
//   2) Header validation: after a successful bake, corrupt the magic bytes
//      in place on disk and confirm MpAssetReader::open surfaces
//      ValidateMagicFailed.
//   3) Version validation: corrupt the `version` field and confirm
//      ValidateVersionFailed surfaces.
//   4) Determinism: bake twice to different tmp paths; byte-compare the
//      two resulting files — must be identical.
//
// Plain main, printf output, exit 0 on pass — mirrors prior stage tests.
// Built under /W4 /WX.

#include "asset/MpAssetFormat.h"
#include "asset/MpAssetReader.h"

#include "mpbake/ClusterBuilder.h"
#include "mpbake/DagBuilder.h"
#include "mpbake/GltfIngest.h"
#include "mpbake/PageWriter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <filesystem>
#include <fstream>
#include <ios>
#include <span>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace fs = std::filesystem;

using enigma::mpbake::ClusterBuildOptions;
using enigma::mpbake::ClusterBuilder;
using enigma::mpbake::ClusterData;
using enigma::mpbake::DagBuildOptions;
using enigma::mpbake::DagBuilder;
using enigma::mpbake::DagResult;
using enigma::mpbake::GltfIngest;
using enigma::mpbake::PageWriter;
using enigma::mpbake::PageWriteOptions;
using enigma::mpbake::pageWriteErrorKindString;

using enigma::asset::ClusterOnDisk;
using enigma::asset::MpAssetReader;
using enigma::asset::MpReadErrorKind;
using enigma::asset::mpReadErrorKindString;
using enigma::asset::PageView;

namespace {

// Walk upward from argv[0]'s directory looking for assets/DamagedHelmet.glb.
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

// Build the DagResult for DamagedHelmet through the full pipeline.
std::expected<DagResult, std::string>
buildDag(const fs::path& assetPath) {
    GltfIngest ingest;
    auto ingestResult = ingest.load(assetPath);
    if (!ingestResult.has_value()) {
        return std::unexpected(std::string{"ingest failed"});
    }
    ClusterBuilder cb;
    ClusterBuildOptions cbOpts{};
    auto clusterResult = cb.build(*ingestResult, cbOpts);
    if (!clusterResult.has_value()) {
        return std::unexpected(std::string{"cluster build failed"});
    }
    DagBuilder db;
    DagBuildOptions dbOpts{};
    auto dagResult = db.build(std::span<const ClusterData>(*clusterResult), dbOpts);
    if (!dagResult.has_value()) {
        return std::unexpected(std::string{"dag build failed"});
    }
    return std::move(*dagResult);
}

// Read an entire file into memory.
std::vector<std::uint8_t> readFile(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f.is_open()) return {};
    f.seekg(0, std::ios::end);
    const auto sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> buf(static_cast<std::size_t>(sz));
    if (sz > 0) f.read(reinterpret_cast<char*>(buf.data()), sz);
    return buf;
}

fs::path tmpPath(const std::string& tag) {
    std::error_code ec;
    const fs::path base = fs::temp_directory_path(ec);
    if (ec || base.empty()) return fs::path{"." } / ("page_writer_test_" + tag + ".mpa");
    return base / ("page_writer_test_" + tag + ".mpa");
}

// Case 1: Round-trip. Bake -> MpAssetReader::open -> verify each cluster
// matches the in-memory DagResult byte-for-byte (positions, normals, uvs,
// triangles, metadata).
bool testRoundTrip(const fs::path& assetPath) {
    auto dagE = buildDag(assetPath);
    if (!dagE.has_value()) {
        std::fprintf(stderr,
            "[page_writer_test] case 1 FAIL: %s\n", dagE.error().c_str());
        return false;
    }
    const DagResult& dag = *dagE;

    const fs::path out = tmpPath("roundtrip");
    {
        PageWriter writer;
        auto res = writer.write(dag, out, PageWriteOptions{});
        if (!res.has_value()) {
            std::fprintf(stderr,
                "[page_writer_test] case 1 FAIL: write: %s: %s\n",
                pageWriteErrorKindString(res.error().kind),
                res.error().detail.c_str());
            return false;
        }
    }

    MpAssetReader reader;
    auto openRes = reader.open(out);
    if (!openRes.has_value()) {
        std::fprintf(stderr,
            "[page_writer_test] case 1 FAIL: open: %s: %s\n",
            mpReadErrorKindString(openRes.error().kind),
            openRes.error().detail.c_str());
        return false;
    }

    // Header sanity.
    if (reader.header().dagNodeCount != dag.nodes.size()) {
        std::fprintf(stderr,
            "[page_writer_test] case 1 FAIL: dagNodeCount mismatch: disk=%u in-mem=%zu\n",
            reader.header().dagNodeCount, dag.nodes.size());
        return false;
    }

    // Walk page table, decompress each page, and cross-reference each
    // cluster against the in-memory DagResult. Track total clusters seen
    // so we can assert every DAG cluster was covered by exactly one page.
    std::vector<std::uint8_t> pageBuf;
    std::size_t totalClustersSeen = 0u;
    std::vector<bool> covered(dag.clusters.size(), false);

    const auto pageTbl = reader.pageTable();
    for (std::uint32_t pi = 0; pi < reader.header().pageCount; ++pi) {
        auto pvE = reader.fetchPage(pi, pageBuf);
        if (!pvE.has_value()) {
            std::fprintf(stderr,
                "[page_writer_test] case 1 FAIL: fetchPage(%u): %s: %s\n",
                pi,
                mpReadErrorKindString(pvE.error().kind),
                pvE.error().detail.c_str());
            return false;
        }
        const PageView& pv = *pvE;
        const auto& entry = pageTbl[pi];

        if (pv.clusters.size() != entry.clusterCount) {
            std::fprintf(stderr,
                "[page_writer_test] case 1 FAIL: page %u clusterCount span=%zu entry=%u\n",
                pi, pv.clusters.size(), entry.clusterCount);
            return false;
        }

        // The entry's firstDagNodeIdx tells us where in the DAG node array
        // this page's first cluster lives. Subsequent in-page clusters
        // follow in ascending node-index order within the group (all share
        // the same parent-group key, so the writer emits them in node-
        // index order).
        //
        // We rely on this ordering by scanning the DAG for nodes whose
        // pageId == pi, preserving DAG-node order, and pairing them up
        // with the on-disk cluster entries.
        std::vector<std::uint32_t> orderedNodes;
        for (std::uint32_t ni = 0; ni < reader.header().dagNodeCount; ++ni) {
            if (reader.dagNodes()[ni].pageId == pi) {
                orderedNodes.push_back(ni);
            }
        }
        if (orderedNodes.size() != entry.clusterCount) {
            std::fprintf(stderr,
                "[page_writer_test] case 1 FAIL: page %u DAG nodes w/ pageId=%u "
                "count=%zu != entry.clusterCount=%u\n",
                pi, pi, orderedNodes.size(), entry.clusterCount);
            return false;
        }

        for (std::size_t ci = 0; ci < pv.clusters.size(); ++ci) {
            const ClusterOnDisk& cd = pv.clusters[ci];
            const std::uint32_t nodeIdx = orderedNodes[ci];
            const std::uint32_t clusterId = dag.nodes[nodeIdx].clusterId;
            if (clusterId >= dag.clusters.size()) {
                std::fprintf(stderr,
                    "[page_writer_test] case 1 FAIL: clusterId %u out of range\n",
                    clusterId);
                return false;
            }
            const ClusterData& src = dag.clusters[clusterId];
            covered[clusterId] = true;
            ++totalClustersSeen;

            // Counts.
            if (cd.vertexCount != src.positions.size()) {
                std::fprintf(stderr,
                    "[page_writer_test] case 1 FAIL: page %u ci %zu vertex count disk=%u mem=%zu\n",
                    pi, ci, cd.vertexCount, src.positions.size());
                return false;
            }
            if (cd.triangleCount != src.triangles.size() / 3u) {
                std::fprintf(stderr,
                    "[page_writer_test] case 1 FAIL: page %u ci %zu triangle count disk=%u mem=%zu\n",
                    pi, ci, cd.triangleCount, src.triangles.size() / 3u);
                return false;
            }

            // Vertex bytes. Layout per-vertex: pos(vec3) + normal(vec3) + uv(vec2) = 32 B.
            const std::byte* vBase = pv.vertexBlob.data() + cd.vertexOffset;
            for (std::uint32_t v = 0; v < cd.vertexCount; ++v) {
                glm::vec3 p{};
                glm::vec3 n{};
                glm::vec2 u{};
                std::memcpy(&p, vBase + v * 32u,       sizeof(glm::vec3));
                std::memcpy(&n, vBase + v * 32u + 12u, sizeof(glm::vec3));
                std::memcpy(&u, vBase + v * 32u + 24u, sizeof(glm::vec2));
                if (p.x != src.positions[v].x ||
                    p.y != src.positions[v].y ||
                    p.z != src.positions[v].z) {
                    std::fprintf(stderr,
                        "[page_writer_test] case 1 FAIL: page %u ci %zu v %u position mismatch\n",
                        pi, ci, v);
                    return false;
                }
                if (n.x != src.normals[v].x ||
                    n.y != src.normals[v].y ||
                    n.z != src.normals[v].z) {
                    std::fprintf(stderr,
                        "[page_writer_test] case 1 FAIL: page %u ci %zu v %u normal mismatch\n",
                        pi, ci, v);
                    return false;
                }
                if (u.x != src.uvs[v].x || u.y != src.uvs[v].y) {
                    std::fprintf(stderr,
                        "[page_writer_test] case 1 FAIL: page %u ci %zu v %u uv mismatch\n",
                        pi, ci, v);
                    return false;
                }
            }

            // Triangle bytes — raw memcmp against src.triangles.
            const std::byte* tBase = pv.triangleBlob.data() + cd.triangleOffset;
            const std::size_t tBytes = static_cast<std::size_t>(cd.triangleCount) * 3u;
            if (tBytes != src.triangles.size()) {
                std::fprintf(stderr,
                    "[page_writer_test] case 1 FAIL: page %u ci %zu triangle byte mismatch\n",
                    pi, ci);
                return false;
            }
            if (tBytes > 0 && std::memcmp(tBase, src.triangles.data(), tBytes) != 0) {
                std::fprintf(stderr,
                    "[page_writer_test] case 1 FAIL: page %u ci %zu triangle bytes differ\n",
                    pi, ci);
                return false;
            }

            // LOD level and dag-lod metadata.
            if (cd.dagLodLevel != src.dagLodLevel) {
                std::fprintf(stderr,
                    "[page_writer_test] case 1 FAIL: page %u ci %zu dagLodLevel disk=%u mem=%u\n",
                    pi, ci, cd.dagLodLevel, src.dagLodLevel);
                return false;
            }
        }
    }

    if (totalClustersSeen != dag.clusters.size()) {
        std::fprintf(stderr,
            "[page_writer_test] case 1 FAIL: totalClustersSeen=%zu != dag.clusters=%zu\n",
            totalClustersSeen, dag.clusters.size());
        return false;
    }
    for (std::size_t i = 0; i < covered.size(); ++i) {
        if (!covered[i]) {
            std::fprintf(stderr,
                "[page_writer_test] case 1 FAIL: cluster %zu not emitted in any page\n", i);
            return false;
        }
    }

    std::printf(
        "[page_writer_test] case 1 PASS: roundtrip: pages=%u dagNodes=%u clusters=%zu file=%llu B\n",
        reader.header().pageCount, reader.header().dagNodeCount, dag.clusters.size(),
        static_cast<unsigned long long>(fs::file_size(out)));

    // Clean up.
    std::error_code ec;
    fs::remove(out, ec);
    return true;
}

// Case 2: corrupt the 4-byte magic. Expect ValidateMagicFailed on open.
bool testCorruptMagic(const fs::path& assetPath) {
    auto dagE = buildDag(assetPath);
    if (!dagE.has_value()) {
        std::fprintf(stderr,
            "[page_writer_test] case 2 FAIL: %s\n", dagE.error().c_str());
        return false;
    }
    const fs::path out = tmpPath("bad_magic");
    {
        PageWriter writer;
        auto res = writer.write(*dagE, out, PageWriteOptions{});
        if (!res.has_value()) {
            std::fprintf(stderr, "[page_writer_test] case 2 FAIL: write failed\n");
            return false;
        }
    }
    // Flip the first magic byte.
    {
        std::fstream f(out, std::ios::in | std::ios::out | std::ios::binary);
        if (!f.is_open()) {
            std::fprintf(stderr, "[page_writer_test] case 2 FAIL: reopen failed\n");
            return false;
        }
        char bad = 'X';
        f.seekp(0, std::ios::beg);
        f.write(&bad, 1);
        f.close();
    }
    MpAssetReader reader;
    auto openRes = reader.open(out);
    std::error_code ec;
    fs::remove(out, ec);
    if (openRes.has_value()) {
        std::fprintf(stderr, "[page_writer_test] case 2 FAIL: expected error, got success\n");
        return false;
    }
    if (openRes.error().kind != MpReadErrorKind::ValidateMagicFailed) {
        std::fprintf(stderr,
            "[page_writer_test] case 2 FAIL: expected ValidateMagicFailed, got %s\n",
            mpReadErrorKindString(openRes.error().kind));
        return false;
    }
    std::printf("[page_writer_test] case 2 PASS: ValidateMagicFailed surfaced as expected.\n");
    return true;
}

// Case 3: corrupt version. Expect ValidateVersionFailed.
bool testCorruptVersion(const fs::path& assetPath) {
    auto dagE = buildDag(assetPath);
    if (!dagE.has_value()) {
        std::fprintf(stderr,
            "[page_writer_test] case 3 FAIL: %s\n", dagE.error().c_str());
        return false;
    }
    const fs::path out = tmpPath("bad_version");
    {
        PageWriter writer;
        auto res = writer.write(*dagE, out, PageWriteOptions{});
        if (!res.has_value()) {
            std::fprintf(stderr, "[page_writer_test] case 3 FAIL: write failed\n");
            return false;
        }
    }
    // Overwrite the 4-byte `version` field (immediately after magic[4]).
    {
        std::fstream f(out, std::ios::in | std::ios::out | std::ios::binary);
        if (!f.is_open()) {
            std::fprintf(stderr, "[page_writer_test] case 3 FAIL: reopen failed\n");
            return false;
        }
        std::uint32_t bogus = 0xDEADBEEFu;
        f.seekp(4, std::ios::beg);
        f.write(reinterpret_cast<const char*>(&bogus), sizeof(bogus));
        f.close();
    }
    MpAssetReader reader;
    auto openRes = reader.open(out);
    std::error_code ec;
    fs::remove(out, ec);
    if (openRes.has_value()) {
        std::fprintf(stderr, "[page_writer_test] case 3 FAIL: expected error, got success\n");
        return false;
    }
    if (openRes.error().kind != MpReadErrorKind::ValidateVersionFailed) {
        std::fprintf(stderr,
            "[page_writer_test] case 3 FAIL: expected ValidateVersionFailed, got %s\n",
            mpReadErrorKindString(openRes.error().kind));
        return false;
    }
    std::printf("[page_writer_test] case 3 PASS: ValidateVersionFailed surfaced as expected.\n");
    return true;
}

// Case 4: Determinism — bake twice, byte-compare.
bool testDeterminism(const fs::path& assetPath) {
    auto dag1 = buildDag(assetPath);
    auto dag2 = buildDag(assetPath);
    if (!dag1.has_value() || !dag2.has_value()) {
        std::fprintf(stderr,
            "[page_writer_test] case 4 FAIL: dag build failed\n");
        return false;
    }
    const fs::path outA = tmpPath("det_a");
    const fs::path outB = tmpPath("det_b");
    PageWriter w1, w2;
    auto r1 = w1.write(*dag1, outA, PageWriteOptions{});
    auto r2 = w2.write(*dag2, outB, PageWriteOptions{});
    if (!r1.has_value() || !r2.has_value()) {
        std::fprintf(stderr,
            "[page_writer_test] case 4 FAIL: write failed\n");
        return false;
    }
    auto bytesA = readFile(outA);
    auto bytesB = readFile(outB);
    std::error_code ec;
    fs::remove(outA, ec);
    fs::remove(outB, ec);
    if (bytesA.size() != bytesB.size()) {
        std::fprintf(stderr,
            "[page_writer_test] case 4 FAIL: file sizes differ a=%zu b=%zu\n",
            bytesA.size(), bytesB.size());
        return false;
    }
    if (std::memcmp(bytesA.data(), bytesB.data(), bytesA.size()) != 0) {
        std::fprintf(stderr,
            "[page_writer_test] case 4 FAIL: bytes differ (size=%zu)\n",
            bytesA.size());
        return false;
    }
    std::printf(
        "[page_writer_test] case 4 PASS: two-run determinism, file=%zu B\n",
        bytesA.size());
    return true;
}

} // namespace

int main(int argc, char** argv) {
    const fs::path asset = locateDamagedHelmet(argc > 0 ? argv[0] : nullptr);
    if (asset.empty()) {
        std::fprintf(stderr,
            "[page_writer_test] FAIL: could not locate assets/DamagedHelmet.glb\n");
        return 1;
    }

    bool ok = true;
    ok &= testRoundTrip(asset);
    ok &= testCorruptMagic(asset);
    ok &= testCorruptVersion(asset);
    ok &= testDeterminism(asset);

    if (!ok) {
        std::fprintf(stderr, "[page_writer_test] FAILED\n");
        return 1;
    }
    std::printf("[page_writer_test] All cases passed.\n");
    return 0;
}
