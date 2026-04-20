// Unit test for the offline `enigma-mpbake` DagBuilder stage (M1b.4).
// Five cases:
//   1) Happy path: DamagedHelmet.glb -> ingest -> cluster -> DAG. Verify:
//        - leafCount == cluster_builder_test's expected leaf count
//        - rootCount <= rootClusterThreshold (4)
//        - maxLodLevel >= 2
//        - every non-root, non-orphan DagNode has parentGroupId != UINT32_MAX
//          (orphans are nodes whose group's simplify/re-cluster produced
//          no output and so have no higher-level parent; they are treated
//          as roots for the child-count invariant below.)
//        - sum of all childCount == count of non-orphan non-root nodes
//          (each such node must be slotted as a child of exactly one parent).
//   2) Empty input: empty span -> EmptyInput error.
//   3) OptionsOutOfRange: targetGroupSize == 0 -> OptionsOutOfRange error.
//   4) Determinism: build twice, hash clusters + nodes, confirm identical.
//   5) Cross-group boundary continuity: for every pair (A, B) of clusters
//      at the same LOD level that share a common parent group at the next
//      level, the welded-position boundary vertices present in both A and
//      B must have identical positions on both sides. Catches M3-era
//      regressions if the DAG's simplify ever stops being symmetric across
//      group boundaries (would tear at mixed-LOD cuts at runtime).
//
// Plain main, printf output, exit 0 on pass — mirrors prior stage tests.
// Built under /W4 /WX.

#include "mpbake/ClusterBuilder.h"
#include "mpbake/DagBuilder.h"
#include "mpbake/GltfIngest.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
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
using enigma::mpbake::DagBuildError;
using enigma::mpbake::DagBuildErrorKind;
using enigma::mpbake::DagBuildOptions;
using enigma::mpbake::DagBuilder;
using enigma::mpbake::DagNode;
using enigma::mpbake::DagResult;
using enigma::mpbake::GltfIngest;
using enigma::mpbake::IngestedMesh;
using enigma::mpbake::dagBuildErrorKindString;

namespace {

// Walk upward from argv[0]'s directory looking for `assets/DamagedHelmet.glb`.
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

// FNV-1a 64-bit — deterministic hash. Matches prior stage test style.
std::uint64_t fnv1a64(const void* data, std::size_t size) {
    const std::uint8_t* p = static_cast<const std::uint8_t*>(data);
    std::uint64_t h = 0xcbf29ce484222325ull;
    for (std::size_t i = 0; i < size; ++i) {
        h ^= static_cast<std::uint64_t>(p[i]);
        h *= 0x100000001b3ull;
    }
    return h;
}

// Hash the full DagResult: cluster geometry + node metadata (lodLevel,
// parentGroupId, childCount, boundsSphere, maxError).
std::uint64_t hashDagResult(const DagResult& r) {
    std::uint64_t h = 0xcbf29ce484222325ull;
    auto mix = [&](const void* p, std::size_t n) {
        const std::uint64_t sub = fnv1a64(p, n);
        h ^= sub;
        h *= 0x100000001b3ull;
    };

    const std::size_t nc = r.clusters.size();
    const std::size_t nn = r.nodes.size();
    mix(&nc, sizeof(nc));
    mix(&nn, sizeof(nn));
    mix(&r.leafCount,   sizeof(r.leafCount));
    mix(&r.rootCount,   sizeof(r.rootCount));
    mix(&r.maxLodLevel, sizeof(r.maxLodLevel));

    for (const auto& c : r.clusters) {
        const std::size_t vc = c.positions.size();
        const std::size_t tc = c.triangles.size() / 3u;
        mix(&vc, sizeof(vc));
        mix(&tc, sizeof(tc));
        if (!c.positions.empty()) {
            mix(c.positions.data(), c.positions.size() * sizeof(glm::vec3));
        }
        if (!c.triangles.empty()) {
            mix(c.triangles.data(), c.triangles.size());
        }
        mix(&c.dagLodLevel, sizeof(c.dagLodLevel));
    }

    for (const auto& n : r.nodes) {
        mix(&n.lodLevel,       sizeof(n.lodLevel));
        mix(&n.parentGroupId,  sizeof(n.parentGroupId));
        mix(&n.childCount,     sizeof(n.childCount));
        mix(&n.firstChildNode, sizeof(n.firstChildNode));
        mix(&n.boundsSphere,   sizeof(n.boundsSphere));
        mix(&n.maxError,       sizeof(n.maxError));
    }
    return h;
}

// Build leaf clusters from DamagedHelmet.
std::vector<ClusterData> loadAndCluster(const fs::path& assetPath) {
    GltfIngest ingest;
    auto ingestResult = ingest.load(assetPath);
    if (!ingestResult.has_value()) {
        std::fprintf(stderr, "[dag_builder_test] ingest failed on '%s'\n",
            assetPath.string().c_str());
        return {};
    }
    ClusterBuilder builder;
    ClusterBuildOptions opts{};   // 128/128/0.5
    auto r = builder.build(*ingestResult, opts);
    if (!r.has_value()) {
        std::fprintf(stderr, "[dag_builder_test] cluster build failed\n");
        return {};
    }
    return std::move(*r);
}

bool testHappyPath(const fs::path& assetPath) {
    const std::vector<ClusterData> leafClusters = loadAndCluster(assetPath);
    if (leafClusters.empty()) {
        std::fprintf(stderr, "[dag_builder_test] case 1 FAIL: leaf cluster set empty\n");
        return false;
    }

    DagBuilder builder;
    DagBuildOptions opts{};
    // Defaults: targetGroupSize=8, simplifyRatio=0.5, maxErrorPerLevel=0.02,
    //           rootClusterThreshold=4, maxLodLevels=24.

    auto result = builder.build(
        std::span<const ClusterData>(leafClusters), opts);
    if (!result.has_value()) {
        const auto& err = result.error();
        std::fprintf(stderr,
            "[dag_builder_test] case 1 FAIL: DagBuilder errored (level %u): %s: %s\n",
            err.lodLevel,
            dagBuildErrorKindString(err.kind),
            err.detail.c_str());
        return false;
    }
    const DagResult& dag = *result;

    // clusters and nodes must be parallel arrays of the same size.
    if (dag.clusters.size() != dag.nodes.size()) {
        std::fprintf(stderr,
            "[dag_builder_test] case 1 FAIL: clusters.size()=%zu != nodes.size()=%zu\n",
            dag.clusters.size(), dag.nodes.size());
        return false;
    }

    // leafCount must match the input cluster count (leaf LOD == 0).
    const std::uint32_t expectedLeafCount =
        static_cast<std::uint32_t>(leafClusters.size());
    if (dag.leafCount != expectedLeafCount) {
        std::fprintf(stderr,
            "[dag_builder_test] case 1 FAIL: leafCount=%u != expected %u\n",
            dag.leafCount, expectedLeafCount);
        return false;
    }

    // rootCount must be <= rootClusterThreshold (4).
    if (dag.rootCount > opts.rootClusterThreshold) {
        std::fprintf(stderr,
            "[dag_builder_test] case 1 FAIL: rootCount=%u > threshold=%u\n",
            dag.rootCount, opts.rootClusterThreshold);
        return false;
    }
    if (dag.rootCount == 0u) {
        std::fprintf(stderr,
            "[dag_builder_test] case 1 FAIL: rootCount=0\n");
        return false;
    }

    // Some DAG depth achieved (DamagedHelmet has ~132 leaves -> >= 2 levels).
    if (dag.maxLodLevel < 2u) {
        std::fprintf(stderr,
            "[dag_builder_test] case 1 FAIL: maxLodLevel=%u < 2\n",
            dag.maxLodLevel);
        return false;
    }

    // Every non-root, non-orphan node must have parentGroupId != UINT32_MAX.
    // "root" means lodLevel == maxLodLevel. "orphan" means a non-root node
    // whose group's simplify or re-cluster stage produced zero clusters at
    // the next LOD level — those nodes legitimately have no higher-level
    // parent and are counted alongside roots for the child-count invariant
    // below. (This can happen when a group's simplify collapses completely
    // or when the locked border leaves no collapsible interior edges.)
    std::uint32_t sumChildCount = 0u;
    std::uint32_t nonOrphanNonRoot = 0u;
    std::uint32_t orphanCount = 0u;
    for (const auto& n : dag.nodes) {
        const bool isRoot   = (n.lodLevel == dag.maxLodLevel);
        const bool isOrphan = (!isRoot && n.parentGroupId == UINT32_MAX);
        if (isOrphan) ++orphanCount;
        if (!isRoot && !isOrphan) ++nonOrphanNonRoot;
        sumChildCount += n.childCount;
    }

    // Child-slot invariant: every non-orphan non-root node must be slotted
    // as a child of exactly one parent group, so the total number of
    // child-slot entries equals the count of non-orphan non-root nodes.
    // Orphans and roots contribute nothing to childCount.
    if (sumChildCount != nonOrphanNonRoot) {
        std::fprintf(stderr,
            "[dag_builder_test] case 1 FAIL: sum(childCount)=%u != "
            "nonOrphanNonRoot=%u (nodes=%zu rootCount=%u orphans=%u)\n",
            sumChildCount, nonOrphanNonRoot,
            dag.nodes.size(), dag.rootCount, orphanCount);
        return false;
    }

    std::printf(
        "[dag_builder_test] case 1 PASS: leafCount=%u rootCount=%u "
        "orphans=%u maxLodLevel=%u totalClusters=%zu\n",
        dag.leafCount, dag.rootCount, orphanCount,
        dag.maxLodLevel, dag.clusters.size());
    return true;
}

bool testEmptyInput() {
    DagBuilder builder;
    DagBuildOptions opts{};
    auto r = builder.build(std::span<const ClusterData>{}, opts);
    if (r.has_value()) {
        std::fprintf(stderr,
            "[dag_builder_test] case 2 FAIL: expected EmptyInput, got success\n");
        return false;
    }
    if (r.error().kind != DagBuildErrorKind::EmptyInput) {
        std::fprintf(stderr,
            "[dag_builder_test] case 2 FAIL: expected EmptyInput, got %s: %s\n",
            dagBuildErrorKindString(r.error().kind),
            r.error().detail.c_str());
        return false;
    }
    std::printf("[dag_builder_test] case 2 PASS: EmptyInput surfaced as expected.\n");
    return true;
}

bool testOptionsOutOfRange(const fs::path& assetPath) {
    const std::vector<ClusterData> leafClusters = loadAndCluster(assetPath);
    if (leafClusters.empty()) {
        std::fprintf(stderr,
            "[dag_builder_test] case 3 FAIL: leaf cluster set empty\n");
        return false;
    }
    DagBuilder builder;
    DagBuildOptions opts{};
    opts.targetGroupSize = 0u;   // invalid

    auto r = builder.build(
        std::span<const ClusterData>(leafClusters), opts);
    if (r.has_value()) {
        std::fprintf(stderr,
            "[dag_builder_test] case 3 FAIL: expected OptionsOutOfRange, "
            "got success\n");
        return false;
    }
    if (r.error().kind != DagBuildErrorKind::OptionsOutOfRange) {
        std::fprintf(stderr,
            "[dag_builder_test] case 3 FAIL: expected OptionsOutOfRange, "
            "got %s: %s\n",
            dagBuildErrorKindString(r.error().kind),
            r.error().detail.c_str());
        return false;
    }
    std::printf(
        "[dag_builder_test] case 3 PASS: OptionsOutOfRange surfaced as expected.\n");
    return true;
}

// Case 5: cross-group boundary continuity.
//
// For every same-level pair (A, B) of clusters whose nodes share a common
// parent group at the next level (or more precisely: whose parent clusters,
// via `parentGroupId`, trace to the same group id), we assert that any
// welded-position vertex present in BOTH A and B occupies the same world
// coordinate on both sides. This is a property of the bake: the DAG's
// per-group simplify must be symmetric at its borders so neighbouring
// clusters agree on shared edges, otherwise mixed-LOD cuts crack at runtime.
//
// Welding is done via an int32 position grid at the same epsilon used by
// the DAG's adjacency build. If a grid cell appears in both clusters we
// require the originating float positions to be identical (bit-equal) —
// they were welded from the same input vertex during the same simplify.
bool testCrossGroupContinuity(const fs::path& assetPath) {
    const std::vector<ClusterData> leafClusters = loadAndCluster(assetPath);
    if (leafClusters.empty()) {
        std::fprintf(stderr,
            "[dag_builder_test] case 5 FAIL: leaf cluster set empty\n");
        return false;
    }

    DagBuilder builder;
    DagBuildOptions opts{};
    auto result = builder.build(
        std::span<const ClusterData>(leafClusters), opts);
    if (!result.has_value()) {
        const auto& err = result.error();
        std::fprintf(stderr,
            "[dag_builder_test] case 5 FAIL: DagBuilder errored (level %u): "
            "%s: %s\n",
            err.lodLevel,
            dagBuildErrorKindString(err.kind),
            err.detail.c_str());
        return false;
    }
    const DagResult& dag = *result;

    // Group same-level clusters by their `parentGroupId`. Two clusters
    // with the same non-MAX parent share a group. For each group, collect
    // the members' welded-cell -> world-position map and ensure overlapping
    // cells have identical positions across all members.
    //
    // Welding uses the same integer-grid quantizer as the DAG's adjacency
    // helper, at the default adjacency epsilon.
    const float weldEpsilon = opts.adjacencyWeldEpsilon;
    const float invEps = 1.0f / weldEpsilon;
    auto quantize = [invEps](float c) -> std::int32_t {
        if (!std::isfinite(c)) return 0;
        const double scaled = static_cast<double>(c) * static_cast<double>(invEps);
        constexpr double kMax = static_cast<double>(INT32_MAX);
        constexpr double kMin = static_cast<double>(INT32_MIN);
        if (scaled >= kMax) return INT32_MAX;
        if (scaled <= kMin) return INT32_MIN;
        return static_cast<std::int32_t>(std::llround(scaled));
    };

    // Key (lodLevel, parentGroupId) -> list of cluster indices in that group.
    std::map<std::pair<std::uint32_t, std::uint32_t>,
             std::vector<std::uint32_t>> sameLevelGroups;
    for (std::uint32_t i = 0; i < dag.nodes.size(); ++i) {
        const auto& n = dag.nodes[i];
        if (n.parentGroupId == UINT32_MAX) continue;  // orphan or root
        sameLevelGroups[{n.lodLevel, n.parentGroupId}].push_back(i);
    }

    std::size_t pairsChecked    = 0;
    std::size_t sharedCellsSeen = 0;
    for (const auto& [key, members] : sameLevelGroups) {
        if (members.size() < 2u) continue;

        // Collect each member's welded-cell -> float-position map.
        std::vector<std::map<std::array<std::int32_t, 3>, glm::vec3>> perMemberCells;
        perMemberCells.reserve(members.size());
        for (std::uint32_t nodeIdx : members) {
            const std::uint32_t cid = dag.nodes[nodeIdx].clusterId;
            if (cid >= dag.clusters.size()) {
                std::fprintf(stderr,
                    "[dag_builder_test] case 5 FAIL: clusterId=%u out of range "
                    "(clusters=%zu)\n",
                    cid, dag.clusters.size());
                return false;
            }
            const ClusterData& c = dag.clusters[cid];
            std::map<std::array<std::int32_t, 3>, glm::vec3> cells;
            for (const glm::vec3& p : c.positions) {
                const std::array<std::int32_t, 3> k{
                    quantize(p.x), quantize(p.y), quantize(p.z),
                };
                cells.emplace(k, p);  // first-position-wins on duplicates
            }
            perMemberCells.push_back(std::move(cells));
        }

        // Pairwise: any cell present in both A and B must have the same
        // float position on both sides.
        for (std::size_t i = 0; i < perMemberCells.size(); ++i) {
            for (std::size_t j = i + 1u; j < perMemberCells.size(); ++j) {
                ++pairsChecked;
                const auto& ca = perMemberCells[i];
                const auto& cb = perMemberCells[j];
                // Walk the smaller map for speed.
                const auto& small = (ca.size() <= cb.size()) ? ca : cb;
                const auto& big   = (ca.size() <= cb.size()) ? cb : ca;
                for (const auto& [k, pa] : small) {
                    auto it = big.find(k);
                    if (it == big.end()) continue;
                    ++sharedCellsSeen;
                    const glm::vec3& pb = it->second;
                    // Strict equality: the bake path welds via the same
                    // quantizer so shared cells must round-trip to equal
                    // float bits. If simplify is asymmetric this will diverge.
                    if (pa.x != pb.x || pa.y != pb.y || pa.z != pb.z) {
                        std::fprintf(stderr,
                            "[dag_builder_test] case 5 FAIL: cross-group "
                            "boundary mismatch at lodLevel=%u pgid=%u "
                            "cell=(%d,%d,%d) a=(%.9g,%.9g,%.9g) "
                            "b=(%.9g,%.9g,%.9g)\n",
                            key.first, key.second, k[0], k[1], k[2],
                            static_cast<double>(pa.x), static_cast<double>(pa.y),
                            static_cast<double>(pa.z),
                            static_cast<double>(pb.x), static_cast<double>(pb.y),
                            static_cast<double>(pb.z));
                        return false;
                    }
                }
            }
        }
    }

    std::printf(
        "[dag_builder_test] case 5 PASS: cross-group continuity — "
        "pairsChecked=%zu sharedCellsSeen=%zu\n",
        pairsChecked, sharedCellsSeen);
    return true;
}

bool testDeterminism(const fs::path& assetPath) {
    const std::vector<ClusterData> leafClusters = loadAndCluster(assetPath);
    if (leafClusters.empty()) {
        std::fprintf(stderr,
            "[dag_builder_test] case 4 FAIL: leaf cluster set empty\n");
        return false;
    }
    DagBuildOptions opts{};

    DagBuilder b1, b2;
    auto r1 = b1.build(std::span<const ClusterData>(leafClusters), opts);
    auto r2 = b2.build(std::span<const ClusterData>(leafClusters), opts);

    if (!r1.has_value() || !r2.has_value()) {
        std::fprintf(stderr,
            "[dag_builder_test] case 4 FAIL: one of the two builds failed\n");
        return false;
    }
    const std::uint64_t h1 = hashDagResult(*r1);
    const std::uint64_t h2 = hashDagResult(*r2);
    if (h1 != h2) {
        std::fprintf(stderr,
            "[dag_builder_test] case 4 FAIL: non-deterministic "
            "hash 0x%016llx vs 0x%016llx\n",
            static_cast<unsigned long long>(h1),
            static_cast<unsigned long long>(h2));
        return false;
    }
    std::printf(
        "[dag_builder_test] case 4 PASS: two-run determinism hash=0x%016llx\n",
        static_cast<unsigned long long>(h1));
    return true;
}

} // namespace

int main(int argc, char** argv) {
    const fs::path asset = locateDamagedHelmet(argc > 0 ? argv[0] : nullptr);
    if (asset.empty()) {
        std::fprintf(stderr,
            "[dag_builder_test] FAIL: could not locate assets/DamagedHelmet.glb\n");
        return 1;
    }

    bool allPassed = true;
    allPassed &= testHappyPath(asset);
    allPassed &= testEmptyInput();
    allPassed &= testOptionsOutOfRange(asset);
    allPassed &= testDeterminism(asset);
    allPassed &= testCrossGroupContinuity(asset);

    if (!allPassed) {
        std::fprintf(stderr, "[dag_builder_test] FAILED\n");
        return 1;
    }
    std::printf("[dag_builder_test] All cases passed.\n");
    return 0;
}
