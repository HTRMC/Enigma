// Unit test for the offline `enigma-mpbake` cluster-builder stage (M1b.2).
// Three cases:
//   1) Happy path: DamagedHelmet.glb → GltfIngest → ClusterBuilder. Verify
//      cluster invariants + determinism (build twice, hash, compare).
//   2) Empty input: an empty `IngestedMesh` → expect `EmptyInput` error.
//   3) Index out of range: a synthesized mesh with an index beyond
//      `positions.size()` → expect `IndexOutOfRange` error.
//
// Plain main, printf output, exit 0 on pass — mirrors gltf_ingest_test.cpp.
// Built under /W4 /WX.

#include "mpbake/ClusterBuilder.h"
#include "mpbake/GltfIngest.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

namespace fs = std::filesystem;

using enigma::mpbake::ClusterBuildError;
using enigma::mpbake::ClusterBuildErrorKind;
using enigma::mpbake::ClusterBuildOptions;
using enigma::mpbake::ClusterBuilder;
using enigma::mpbake::ClusterData;
using enigma::mpbake::GltfIngest;
using enigma::mpbake::IngestedMesh;
using enigma::mpbake::clusterBuildErrorKindString;

namespace {

// Walk upward from argv[0]'s directory looking for `assets/DamagedHelmet.glb`,
// then try cwd as a fallback. Returns empty path if not found.
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

// FNV-1a 64-bit over raw bytes — used to compare cluster outputs across
// two independent build runs without manually diffing every field. Good
// enough for a determinism check (we also compare cluster count + per-
// cluster vertex/tri counts + boundsSphere).
std::uint64_t fnv1a64(const void* data, std::size_t size) {
    const std::uint8_t* p = static_cast<const std::uint8_t*>(data);
    std::uint64_t h = 0xcbf29ce484222325ull;
    for (std::size_t i = 0; i < size; ++i) {
        h ^= static_cast<std::uint64_t>(p[i]);
        h *= 0x100000001b3ull;
    }
    return h;
}

// Hash every ClusterData field that should be stable across runs.
std::uint64_t hashClusters(const std::vector<ClusterData>& cs) {
    std::uint64_t h = 0xcbf29ce484222325ull;
    auto mix = [&](const void* p, std::size_t n) {
        const std::uint64_t sub = fnv1a64(p, n);
        h ^= sub;
        h *= 0x100000001b3ull;
    };
    const std::size_t count = cs.size();
    mix(&count, sizeof(count));
    for (const auto& c : cs) {
        const std::size_t vc = c.positions.size();
        const std::size_t tc = c.triangles.size() / 3u;
        mix(&vc, sizeof(vc));
        mix(&tc, sizeof(tc));
        if (!c.positions.empty()) {
            mix(c.positions.data(), c.positions.size() * sizeof(glm::vec3));
            mix(c.normals.data(),   c.normals.size()   * sizeof(glm::vec3));
            mix(c.uvs.data(),       c.uvs.size()       * sizeof(glm::vec2));
        }
        if (!c.triangles.empty()) {
            mix(c.triangles.data(), c.triangles.size());
        }
        mix(&c.boundsSphere, sizeof(c.boundsSphere));
        mix(&c.coneApex,     sizeof(c.coneApex));
        mix(&c.coneAxis,     sizeof(c.coneAxis));
        mix(&c.coneCutoff,   sizeof(c.coneCutoff));
    }
    return h;
}

bool testHappyPath(const fs::path& assetPath) {
    GltfIngest ingest;
    auto ingestResult = ingest.load(assetPath);
    if (!ingestResult.has_value()) {
        std::fprintf(stderr,
            "[cluster_builder_test] case 1 FAIL: ingest failed on '%s'\n",
            assetPath.string().c_str());
        return false;
    }
    const IngestedMesh& mesh = *ingestResult;

    ClusterBuilder builder;
    ClusterBuildOptions opts{};   // defaults: 128/128/0.5

    auto r = builder.build(mesh, opts);
    if (!r.has_value()) {
        std::fprintf(stderr,
            "[cluster_builder_test] case 1 FAIL: cluster build failed: %s: %s\n",
            clusterBuildErrorKindString(r.error().kind),
            r.error().detail.c_str());
        return false;
    }
    const std::vector<ClusterData>& clusters = *r;

    if (clusters.empty()) {
        std::fprintf(stderr,
            "[cluster_builder_test] case 1 FAIL: zero clusters produced\n");
        return false;
    }

    std::size_t totalTris = 0;
    for (std::size_t i = 0; i < clusters.size(); ++i) {
        const auto& c = clusters[i];
        const std::size_t vc = c.positions.size();
        const std::size_t tc = c.triangles.size() / 3u;

        if (vc == 0 || vc > opts.targetMaxVertices) {
            std::fprintf(stderr,
                "[cluster_builder_test] case 1 FAIL: cluster %zu vertex_count=%zu out of (0, %u]\n",
                i, vc, opts.targetMaxVertices);
            return false;
        }
        if (tc == 0 || tc > opts.targetMaxTriangles) {
            std::fprintf(stderr,
                "[cluster_builder_test] case 1 FAIL: cluster %zu triangle_count=%zu out of (0, %u]\n",
                i, tc, opts.targetMaxTriangles);
            return false;
        }
        if ((c.triangles.size() % 3u) != 0u) {
            std::fprintf(stderr,
                "[cluster_builder_test] case 1 FAIL: cluster %zu triangles.size()=%zu not multiple of 3\n",
                i, c.triangles.size());
            return false;
        }
        if (c.normals.size() != vc || c.uvs.size() != vc) {
            std::fprintf(stderr,
                "[cluster_builder_test] case 1 FAIL: cluster %zu stream sizes diverged (p=%zu n=%zu uv=%zu)\n",
                i, vc, c.normals.size(), c.uvs.size());
            return false;
        }
        for (std::size_t b = 0; b < c.triangles.size(); ++b) {
            if (static_cast<std::size_t>(c.triangles[b]) >= vc) {
                std::fprintf(stderr,
                    "[cluster_builder_test] case 1 FAIL: cluster %zu triangle index %u >= vc %zu\n",
                    i, c.triangles[b], vc);
                return false;
            }
        }
        if (!(c.boundsSphere.w > 0.0f)) {
            std::fprintf(stderr,
                "[cluster_builder_test] case 1 FAIL: cluster %zu bounds radius=%f not > 0\n",
                i, c.boundsSphere.w);
            return false;
        }
        totalTris += tc;
    }

    const std::size_t sourceTris = mesh.indices.size() / 3u;
    if (totalTris != sourceTris) {
        std::fprintf(stderr,
            "[cluster_builder_test] case 1 FAIL: sum-of-cluster-tris=%zu != source tris=%zu\n",
            totalTris, sourceTris);
        return false;
    }

    // --- Determinism: build again and compare ---
    ClusterBuilder builder2;
    auto r2 = builder2.build(mesh, opts);
    if (!r2.has_value()) {
        std::fprintf(stderr,
            "[cluster_builder_test] case 1 FAIL: second build failed\n");
        return false;
    }
    const std::vector<ClusterData>& clusters2 = *r2;
    if (clusters2.size() != clusters.size()) {
        std::fprintf(stderr,
            "[cluster_builder_test] case 1 FAIL: non-deterministic cluster count %zu vs %zu\n",
            clusters.size(), clusters2.size());
        return false;
    }
    const std::uint64_t h1 = hashClusters(clusters);
    const std::uint64_t h2 = hashClusters(clusters2);
    if (h1 != h2) {
        std::fprintf(stderr,
            "[cluster_builder_test] case 1 FAIL: non-deterministic hash 0x%016llx vs 0x%016llx\n",
            static_cast<unsigned long long>(h1),
            static_cast<unsigned long long>(h2));
        return false;
    }

    std::printf(
        "[cluster_builder_test] case 1 PASS: DamagedHelmet.glb -> %zu clusters, "
        "%zu tris total (src=%zu), hash=0x%016llx\n",
        clusters.size(), totalTris, sourceTris,
        static_cast<unsigned long long>(h1));
    return true;
}

bool testEmptyInput() {
    IngestedMesh empty{};   // all vectors default-constructed
    ClusterBuilder builder;
    ClusterBuildOptions opts{};
    auto r = builder.build(empty, opts);
    if (r.has_value()) {
        std::fprintf(stderr,
            "[cluster_builder_test] case 2 FAIL: expected EmptyInput, got success (%zu clusters)\n",
            r->size());
        return false;
    }
    if (r.error().kind != ClusterBuildErrorKind::EmptyInput) {
        std::fprintf(stderr,
            "[cluster_builder_test] case 2 FAIL: expected EmptyInput, got %s\n",
            clusterBuildErrorKindString(r.error().kind));
        return false;
    }
    std::printf(
        "[cluster_builder_test] case 2 PASS: EmptyInput surfaced as expected.\n");
    return true;
}

bool testIndexOutOfRange() {
    // Single position, but the triangle references three indices — the last
    // of which (999) is out of range. The ClusterBuilder pre-flight should
    // catch this before handing to meshopt.
    IngestedMesh bad{};
    bad.positions.push_back(glm::vec3{ 0.0f });
    bad.normals.push_back(glm::vec3{ 0.0f, 1.0f, 0.0f });
    bad.uvs.push_back(glm::vec2{ 0.0f });
    bad.indices = { 0u, 0u, 999u };

    ClusterBuilder builder;
    ClusterBuildOptions opts{};
    auto r = builder.build(bad, opts);
    if (r.has_value()) {
        std::fprintf(stderr,
            "[cluster_builder_test] case 3 FAIL: expected IndexOutOfRange, got success\n");
        return false;
    }
    if (r.error().kind != ClusterBuildErrorKind::IndexOutOfRange) {
        std::fprintf(stderr,
            "[cluster_builder_test] case 3 FAIL: expected IndexOutOfRange, got %s: %s\n",
            clusterBuildErrorKindString(r.error().kind),
            r.error().detail.c_str());
        return false;
    }
    std::printf(
        "[cluster_builder_test] case 3 PASS: IndexOutOfRange surfaced as expected.\n");
    return true;
}

} // namespace

int main(int argc, char** argv) {
    const fs::path asset = locateDamagedHelmet(argc > 0 ? argv[0] : nullptr);
    if (asset.empty()) {
        std::fprintf(stderr,
            "[cluster_builder_test] FAIL: could not locate assets/DamagedHelmet.glb\n");
        return 1;
    }

    bool allPassed = true;
    allPassed &= testHappyPath(asset);
    allPassed &= testEmptyInput();
    allPassed &= testIndexOutOfRange();

    if (!allPassed) {
        std::fprintf(stderr, "[cluster_builder_test] FAILED\n");
        return 1;
    }
    std::printf("[cluster_builder_test] All cases passed.\n");
    return 0;
}
