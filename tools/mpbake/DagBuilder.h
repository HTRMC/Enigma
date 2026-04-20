#pragma once

// DagBuilder.h
// =============
// Stage 4 of `enigma-mpbake` (M1b.4): the iterative loop that builds the
// cluster DAG bottom-up via Fork-A1 recipe from the plan (§3.M1):
//
//   1. Start with leaf clusters (output of ClusterBuilder on the full mesh).
//   2. Build the inter-cluster adjacency graph: two clusters are adjacent iff
//      they share >= 2 welded-position vertices (i.e. share at least one edge).
//      Note: "share >= 2 welded vertices" is a strict SUPERSET of "share at
//      least one edge" — on pathological topology (two clusters that touch
//      at two isolated vertices but not along a common edge) we would flag
//      them as adjacent even though METIS has nothing to cut. This is
//      accepted deliberately: a proper edge-match would require a full
//      per-cluster edge table (O(tri * 3) space per cluster, re-hashed per
//      level), whereas the vertex-count heuristic is one pass over the
//      welded-cell table. False positives only increase METIS's partition
//      search space, not the correctness of the partition.
//   3. Partition clusters into groups of ~`targetGroupSize` via METIS k-way
//      on that adjacency graph.
//   4. For each group: Simplify -> re-Cluster. Emit the new clusters as the
//      NEXT DAG level's inputs.
//   5. Record parent/child DAG links: input clusters of a group all point to
//      the same group id; the group's produced clusters list those inputs as
//      children.
//   6. Repeat (2)-(5) until the level's cluster count drops to
//      `rootClusterThreshold` or the depth hits `maxLodLevels`.
//
// Contract
// --------
// `DagBuilder::build` is stateless + deterministic: same input span + same
// options -> byte-identical `DagResult`. The unit test builds twice and
// hashes. Determinism leans on std::map (ordered) for the adjacency weld
// table, METIS_OPTION_SEED for partition reproducibility, and the inner
// ClusterBuilder / simplify() primitives (both deterministic).
//
// The in-memory `DagNode` here is NOT the on-disk `MpDagNode` from
// `asset/MpAssetFormat.h` — that packing happens later in PageWriter (M1c).

#include "ClusterBuilder.h"
#include "Simplify.h"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

#include <glm/vec4.hpp>

namespace enigma::mpbake {

// Tunables for the DAG construction loop. Defaults match plan §3.M1.
struct DagBuildOptions {
    // Target group size for METIS partitioning. Nanite defaults to ~8.
    // Smaller groups => deeper DAG, better LOD granularity, higher overhead.
    std::uint32_t targetGroupSize = 8u;

    // Simplify aggression per level. 0.5 == halve triangles per level.
    float simplifyRatio = 0.5f;

    // Maximum absolute world-space error allowed PER LEVEL (accumulates).
    float maxErrorPerLevel = 0.02f;

    // Stop iteration when residual cluster count at a level drops to this.
    std::uint32_t rootClusterThreshold = 4u;

    // Hard cap on DAG depth (safety valve against pathological assets).
    std::uint32_t maxLodLevels = 24u;

    // Spatial-weld epsilon for adjacency detection (world units).
    float adjacencyWeldEpsilon = 1e-5f;

    // Deterministic METIS RNG seed. Feeds METIS_OPTION_SEED.
    std::int32_t deterministicSeed = 0xBEEF;

    // Primitive-level knobs propagated into per-group simplify / re-cluster.
    SimplifyOptions     simplifyOpts{};
    ClusterBuildOptions clusterOpts{};
};

// One DAG node = one cluster + its place in the DAG.
//
//  - `clusterId` indexes into `DagResult::clusters` (same array index:
//    `nodes[i].clusterId == i` by construction, but we carry the field
//    explicitly so future reorderings stay easy to reason about).
//  - `parentGroupId` is the index of the group at level `lodLevel + 1`
//    that spawned this cluster; UINT32_MAX at the DAG root. ALSO
//    UINT32_MAX for "orphaned" nodes: clusters whose group's per-level
//    simplify() or re-cluster() produced zero output clusters (degenerate
//    group). Orphans are treated as roots for invariant purposes even
//    though their `lodLevel` is below `maxLodLevel`; they simply have no
//    higher-level parent.
//  - `firstChildNode` / `childCount` name a contiguous range into
//    `DagResult::nodes` holding this cluster's group-merged children.
//    `childCount == 0` for leaves.
//  - `boundsSphere` is (center.xyz, radius) for runtime frustum cull.
//  - `maxError` is the absolute world-space error accumulated monotonically
//    from leaves up through this cluster's bottom-up chain. Specifically
//    it is `max(childMaxError, simplify.achievedError)` at construction
//    time, so a parent's error envelope strictly contains all its
//    children's envelopes. The runtime LOD selector compares this
//    against a screen-space-error threshold.
struct DagNode {
    std::uint32_t clusterId       = 0u;
    std::uint32_t parentGroupId   = UINT32_MAX;
    std::uint32_t firstChildNode  = UINT32_MAX;
    std::uint32_t childCount      = 0u;
    std::uint32_t lodLevel        = 0u;
    glm::vec4     boundsSphere    { 0.0f, 0.0f, 0.0f, 0.0f };
    float         maxError        = 0.0f;
};

// Flat DAG output. `clusters` and `nodes` are parallel arrays, one entry
// per cluster across ALL LOD levels. Leaves are at `lodLevel == 0`; roots
// are at `lodLevel == maxLodLevel`.
struct DagResult {
    std::vector<ClusterData> clusters;
    std::vector<DagNode>     nodes;
    std::uint32_t            rootCount   = 0u;
    std::uint32_t            leafCount   = 0u;
    std::uint32_t            maxLodLevel = 0u;
};

// Classifier for every failure `DagBuilder::build` may surface.
enum class DagBuildErrorKind {
    EmptyInput,                  // leaf cluster span is empty
    AdjacencyGraphBuildFailed,   // CSR adjacency construction tripped an invariant
    MetisPartitionFailed,        // METIS_PartGraphKway returned non-OK
    SimplifyFailed,              // wrapped: inner simplify stage errored
    ClusterRebuildFailed,        // wrapped: inner ClusterBuilder errored
    MaxDepthExceeded,            // DAG hit maxLodLevels without converging to root
    OptionsOutOfRange,           // targetGroupSize==0, etc.
};

// Rich error payload. `lodLevel` is the level at which the error occurred
// (0 for leaf-level failures or option errors).
struct DagBuildError {
    DagBuildErrorKind kind;
    std::string       detail;
    std::uint32_t     lodLevel = 0u;
};

// Stable string for `DagBuildErrorKind`. Guaranteed never null.
const char* dagBuildErrorKindString(DagBuildErrorKind kind) noexcept;

// DagBuilder
// ----------
// Stateless. Construct + `build()`; deterministic given fixed input + opts.
class DagBuilder {
public:
    DagBuilder()  = default;
    ~DagBuilder() = default;

    // Build the DAG from leaf-level clusters. See file header for contract.
    std::expected<DagResult, DagBuildError>
    build(std::span<const ClusterData> leafClusters,
          const DagBuildOptions& opts);
};

} // namespace enigma::mpbake
