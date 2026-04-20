// DagBuilder.cpp
// ==============
// M1b.4 implementation. See DagBuilder.h for the contract. This TU runs
// the Fork-A1 bottom-up DAG build:
//
//   1. Build inter-cluster adjacency: two clusters are adjacent iff they
//      share >= 2 welded-position vertices (i.e. share at least one edge
//      for well-formed input).
//   2. Partition the adjacency graph into groups via METIS k-way.
//   3. For each group: merge cluster triangles into a simplify() call,
//      wrap the simplified mesh back into an IngestedMesh, re-cluster via
//      ClusterBuilder.
//   4. Append the level's groups + newly-produced clusters to the flat
//      DagResult. Wire parent/child indices.
//   5. Loop using the newly-produced clusters as the next level's input
//      until the cluster count converges to `rootClusterThreshold` or the
//      depth hits `maxLodLevels`.
//
// Hardening follows M1b.2 / M1b.3 patterns: knob validation up front,
// overflow guards on CSR sizing, NaN/Inf sanitization on accumulated error,
// std::map (ordered) for determinism.

#include "DagBuilder.h"

#include "ClusterBuilder.h"
#include "GltfIngest.h"
#include "Simplify.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

#ifdef _MSC_VER
#  pragma warning(push)
#  pragma warning(disable: 4005)  // metis.h redefines INT32_MIN/MAX/INT64_MIN/MAX already in stdint.h
#endif
#include <metis.h>
#ifdef _MSC_VER
#  pragma warning(pop)
#endif

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace enigma::mpbake {

const char* dagBuildErrorKindString(DagBuildErrorKind kind) noexcept {
    switch (kind) {
        case DagBuildErrorKind::EmptyInput:                return "EmptyInput";
        case DagBuildErrorKind::AdjacencyGraphBuildFailed: return "AdjacencyGraphBuildFailed";
        case DagBuildErrorKind::MetisPartitionFailed:      return "MetisPartitionFailed";
        case DagBuildErrorKind::SimplifyFailed:            return "SimplifyFailed";
        case DagBuildErrorKind::ClusterRebuildFailed:      return "ClusterRebuildFailed";
        case DagBuildErrorKind::MaxDepthExceeded:          return "MaxDepthExceeded";
        case DagBuildErrorKind::OptionsOutOfRange:         return "OptionsOutOfRange";
    }
    return "Unknown";
}

namespace {

// Re-exported from Simplify.h so the adjacency and the lock-mask builders
// use byte-identical quantization. Local aliases keep the existing call
// sites in this TU compiling unchanged.
inline std::int32_t quantize_coord(float c, float invEpsilon) noexcept {
    return enigma::mpbake::simplify_quantize_coord(c, invEpsilon);
}

using WeldKey = enigma::mpbake::SimplifyWeldKey;

// Adjacency graph in METIS CSR form. `xadj` has size nvtxs+1, `adjncy` is
// the flattened per-vertex neighbour lists. Both are int32_t per METIS's
// IDXTYPEWIDTH=32 setting (see vendor/metis CMake).
struct AdjacencyCSR {
    std::vector<idx_t> xadj;
    std::vector<idx_t> adjncy;
};

} // namespace

// Forward-declared helper: build the cluster-adjacency graph for a slice
// [offset, offset+count) of a ClusterData container. Returns CSR on success.
// Failure is signaled via `outDetail` and a return of false so the caller
// can propagate with context.
//
// We take the container by const-ref + offset/count instead of a span so
// that the caller is free to mutate (e.g. push_back onto) the same vector
// between the time of construction and destruction of this helper's locals.
// A `std::span` computed from `vector::data()` would dangle on realloc;
// carrying (vector&, offset, count) instead lets us index into the vector
// directly and is invalidation-safe as long as the caller holds the vector
// reference stable across this call (which is trivially true — the call is
// synchronous and `build_adjacency` does not reenter DagBuilder::build).
static bool build_adjacency(const std::vector<ClusterData>& container,
                            std::size_t offset,
                            std::size_t count,
                            float weldEpsilon,
                            AdjacencyCSR& outCsr,
                            std::string& outDetail) {
    // --- Pass 1: bucket every cluster's vertices by welded grid cell. ---
    const float invEps = 1.0f / weldEpsilon;

    // position cell -> set of cluster indices that touch that cell.
    // std::map for determinism; std::set for per-cell dedup (a cluster
    // may reference the same welded cell via multiple local vertices).
    std::map<WeldKey, std::set<std::uint32_t>> cellToClusters;

    // Saturation sentinels: quantize_coord() clamps to INT32_MAX/MIN on
    // overflow or Inf input. Two extreme-coord vertices that both saturate
    // would false-positive into the same weld cell and fake an adjacency,
    // so we skip any vertex whose quantized key hits either limit.
    constexpr std::int32_t kSatHi = std::numeric_limits<std::int32_t>::max();
    constexpr std::int32_t kSatLo = std::numeric_limits<std::int32_t>::min();

    for (std::size_t ci = 0; ci < count; ++ci) {
        const ClusterData& c = container[offset + ci];
        for (std::size_t v = 0; v < c.positions.size(); ++v) {
            const glm::vec3 p = c.positions[v];
            const WeldKey key{
                quantize_coord(p.x, invEps),
                quantize_coord(p.y, invEps),
                quantize_coord(p.z, invEps),
            };
            const bool saturated =
                (key[0] == kSatHi || key[0] == kSatLo ||
                 key[1] == kSatHi || key[1] == kSatLo ||
                 key[2] == kSatHi || key[2] == kSatLo);
            if (saturated) continue;
            cellToClusters[key].insert(static_cast<std::uint32_t>(ci));
        }
    }

    // --- Pass 2: count shared cells per (cluster_i, cluster_j) unordered
    // pair. Two clusters are adjacent iff they share >= 2 welded vertices
    // (i.e. share at least one edge). We use an ordered map keyed by
    // (low, high) for determinism. ---
    std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t> sharedCount;

    for (const auto& [key, cset] : cellToClusters) {
        (void)key;
        if (cset.size() < 2u) continue;
        // Pairwise enumeration. For Nanite-scale inputs (N ~ 132 leaves) the
        // max per-cell fanout is bounded by mesh valence (typ. < 16), so
        // the O(k^2) over small k stays cheap in practice.
        std::vector<std::uint32_t> members(cset.begin(), cset.end());
        for (std::size_t i = 0; i < members.size(); ++i) {
            for (std::size_t j = i + 1u; j < members.size(); ++j) {
                const std::uint32_t a = members[i];
                const std::uint32_t b = members[j];
                ++sharedCount[{a, b}];
            }
        }
    }

    // --- Pass 3: build the symmetric neighbour list. Only keep pairs with
    // sharedCount >= 2 (edge sharing, not just vertex touch). ---
    const std::size_t nvtxs = count;
    std::vector<std::vector<std::uint32_t>> neighbours(nvtxs);
    for (const auto& [pair, count2] : sharedCount) {
        if (count2 >= 2u) {
            neighbours[pair.first].push_back(pair.second);
            neighbours[pair.second].push_back(pair.first);
        }
    }

    // --- Flatten into METIS CSR. xadj is size nvtxs+1; adjncy holds the
    // concatenated per-vertex lists. idx_t == int32_t here. ---
    std::size_t totalEdges = 0;
    for (const auto& v : neighbours) totalEdges += v.size();

    // Overflow guard: METIS's idx_t is int32_t under our IDXTYPEWIDTH=32,
    // so every offset must fit in an int32_t.
    constexpr std::size_t kMaxIdx = static_cast<std::size_t>(
        std::numeric_limits<std::int32_t>::max());
    if (nvtxs > kMaxIdx || totalEdges > kMaxIdx) {
        outDetail = "adjacency graph exceeds int32 range: nvtxs="
            + std::to_string(nvtxs) + " totalEdges=" + std::to_string(totalEdges);
        return false;
    }

    outCsr.xadj.assign(nvtxs + 1u, 0);
    outCsr.adjncy.clear();
    outCsr.adjncy.reserve(totalEdges);
    for (std::size_t ci = 0; ci < nvtxs; ++ci) {
        outCsr.xadj[ci] = static_cast<idx_t>(outCsr.adjncy.size());
        // `neighbours[ci]` is populated in pair-insertion order, which is
        // ordered-map traversal order -> deterministic across runs.
        for (std::uint32_t nb : neighbours[ci]) {
            outCsr.adjncy.push_back(static_cast<idx_t>(nb));
        }
    }
    outCsr.xadj[nvtxs] = static_cast<idx_t>(outCsr.adjncy.size());
    return true;
}

// Partition the adjacency graph into `nparts` groups via METIS k-way.
// Fills `outParts` (size == nvtxs) with the group id for each vertex.
// Returns true on METIS_OK, false otherwise with a detail string.
static bool metis_partition(AdjacencyCSR& csr,
                            idx_t nparts,
                            std::int32_t seed,
                            std::vector<idx_t>& outParts,
                            std::string& outDetail) {
    idx_t nvtxs = static_cast<idx_t>(csr.xadj.size() - 1u);
    if (nvtxs <= 0) {
        outDetail = "metis_partition given empty graph (nvtxs="
            + std::to_string(nvtxs) + ")";
        return false;
    }
    idx_t ncon = 1;  // single balancing constraint

    // METIS options block. METIS_SetDefaultOptions initializes to -1s,
    // which the library reads as "use defaults". We override only the
    // seed for reproducibility.
    std::array<idx_t, METIS_NOPTIONS> options{};
    int setRc = METIS_SetDefaultOptions(options.data());
    if (setRc != METIS_OK) {
        outDetail = "METIS_SetDefaultOptions returned "
            + std::to_string(setRc);
        return false;
    }
    options[METIS_OPTION_SEED]      = static_cast<idx_t>(seed);
    options[METIS_OPTION_NUMBERING] = 0;  // 0-based indexing (C convention)

    outParts.assign(static_cast<std::size_t>(nvtxs), 0);
    idx_t edgecut = 0;

    // METIS's entry points take non-const pointers for historical reasons;
    // the implementation does not modify them across normal runs but we
    // pass our own mutable copies to stay safe.
    int rc = METIS_PartGraphKway(
        &nvtxs,
        &ncon,
        csr.xadj.data(),
        csr.adjncy.data(),
        /*vwgt=*/nullptr,
        /*vsize=*/nullptr,
        /*adjwgt=*/nullptr,
        &nparts,
        /*tpwgts=*/nullptr,
        /*ubvec=*/nullptr,
        options.data(),
        &edgecut,
        outParts.data());
    if (rc != METIS_OK) {
        outDetail = "METIS_PartGraphKway rc=" + std::to_string(rc)
            + " nvtxs=" + std::to_string(nvtxs)
            + " nparts=" + std::to_string(nparts);
        return false;
    }
    (void)edgecut;  // informational only; ignored
    return true;
}

// Convert a SimplifiedGroup into an IngestedMesh so it can be fed back
// through ClusterBuilder at the next level. Zero-copy semantics would be
// nicer but simplify() already owns its buffers; we move them in.
static IngestedMesh simplified_to_mesh(SimplifiedGroup&& sg) {
    IngestedMesh m;
    m.positions = std::move(sg.positions);
    m.normals   = std::move(sg.normals);
    m.uvs       = std::move(sg.uvs);
    m.indices   = std::move(sg.indices);
    return m;
}

std::expected<DagResult, DagBuildError>
DagBuilder::build(std::span<const ClusterData> leafClusters,
                  const DagBuildOptions& opts) {
    using Err = DagBuildError;

    static_assert(sizeof(std::size_t) >= 8, "DagBuilder requires 64-bit size_t");
    static_assert(sizeof(idx_t) == 4,
        "DagBuilder expects METIS IDXTYPEWIDTH=32 (idx_t = int32_t)");

    // --- Pre-flight: knob validation. Matches M1b.2 / M1b.3 style. ---
    if (opts.targetGroupSize == 0u) {
        return std::unexpected(Err{
            DagBuildErrorKind::OptionsOutOfRange,
            std::string{"targetGroupSize must be > 0"},
            0u,
        });
    }
    if (opts.rootClusterThreshold == 0u) {
        return std::unexpected(Err{
            DagBuildErrorKind::OptionsOutOfRange,
            std::string{"rootClusterThreshold must be > 0"},
            0u,
        });
    }
    if (opts.maxLodLevels == 0u) {
        return std::unexpected(Err{
            DagBuildErrorKind::OptionsOutOfRange,
            std::string{"maxLodLevels must be > 0"},
            0u,
        });
    }
    if (!(opts.adjacencyWeldEpsilon > 0.0f) ||
        !std::isfinite(opts.adjacencyWeldEpsilon)) {
        return std::unexpected(Err{
            DagBuildErrorKind::OptionsOutOfRange,
            std::string{"adjacencyWeldEpsilon must be finite and > 0"},
            0u,
        });
    }
    if (!(opts.maxErrorPerLevel >= 0.0f) ||
        !std::isfinite(opts.maxErrorPerLevel)) {
        return std::unexpected(Err{
            DagBuildErrorKind::OptionsOutOfRange,
            std::string{"maxErrorPerLevel must be finite and >= 0"},
            0u,
        });
    }
    if (!(opts.simplifyRatio > 0.0f) || !(opts.simplifyRatio <= 1.0f)
        || !std::isfinite(opts.simplifyRatio)) {
        return std::unexpected(Err{
            DagBuildErrorKind::OptionsOutOfRange,
            std::string{"simplifyRatio must be in (0, 1]"},
            0u,
        });
    }

    // --- Pre-flight: empty input is a hard error. ---
    if (leafClusters.empty()) {
        return std::unexpected(Err{
            DagBuildErrorKind::EmptyInput,
            std::string{"DagBuilder::build given empty cluster span"},
            0u,
        });
    }

    // --- Seed `out` with the leaf level. nodes/clusters are parallel; we
    // carry `clusterId == index-into-clusters` by construction.
    //
    // Reserve aggressively up front: the DAG is a geometric reduction
    // (leaf N + N/2 + N/4 + ... ~= 2N in the worst case). We reserve 2*N
    // for both clusters and nodes so that no push_back reallocates during
    // the build loop. This is defense-in-depth — the adjacency helper
    // and per-level slice logic use index-based access into the vector
    // rather than pointers/spans, so a realloc would not actually corrupt
    // anything, but skipping reallocs keeps the hot path cache-friendly
    // and avoids subtle future hazards. ---
    DagResult out;
    out.clusters.reserve(leafClusters.size() * 2u);
    out.nodes.reserve(leafClusters.size() * 2u);
    for (std::size_t i = 0; i < leafClusters.size(); ++i) {
        ClusterData c = leafClusters[i];  // copy — DagResult owns its data
        c.dagLodLevel = 0u;
        DagNode n;
        n.clusterId      = static_cast<std::uint32_t>(i);
        n.parentGroupId  = UINT32_MAX;
        n.firstChildNode = UINT32_MAX;
        n.childCount     = 0u;
        n.lodLevel       = 0u;
        n.boundsSphere   = c.boundsSphere;
        n.maxError       = c.maxSimplificationError;  // zero for leaves
        out.nodes.push_back(n);
        out.clusters.push_back(std::move(c));
    }
    out.leafCount = static_cast<std::uint32_t>(leafClusters.size());

    // --- Iterative bottom-up loop. ---
    // `levelStart` tracks where the CURRENT level's clusters live in
    // out.nodes[]. On each iteration we compute groups over that range,
    // then emit new clusters appended to out.nodes with lodLevel+1.
    std::uint32_t levelStart       = 0u;
    std::uint32_t levelCount       = out.leafCount;
    std::uint32_t lodLevel         = 0u;
    // Track previous iteration's cluster count to detect lack of progress
    // before we drain maxLodLevels iterations on a pathological input.
    std::uint32_t prevLevelCount   = std::numeric_limits<std::uint32_t>::max();

    // Guard: if the initial leaf count is already <= threshold, the leaves
    // ARE the roots. Nothing to do.
    if (levelCount <= opts.rootClusterThreshold) {
        out.rootCount   = levelCount;
        out.maxLodLevel = 0u;
        return out;
    }

    while (levelCount > opts.rootClusterThreshold) {
        if (lodLevel + 1u >= opts.maxLodLevels) {
            return std::unexpected(Err{
                DagBuildErrorKind::MaxDepthExceeded,
                std::string{"DAG did not converge to root within maxLodLevels="}
                    + std::to_string(opts.maxLodLevels)
                    + " (residual clusters=" + std::to_string(levelCount) + ")",
                lodLevel,
            });
        }

        // --- No-progress guard: if the previous level produced a count
        // that is >= the current level's count, the DAG is not converging
        // and further iterations can only waste time or deadlock on a
        // fully-locked input. Bail immediately with a specific detail so
        // callers can distinguish this from the generic max-depth case. ---
        if (prevLevelCount != std::numeric_limits<std::uint32_t>::max() &&
            levelCount >= prevLevelCount &&
            levelCount > opts.rootClusterThreshold) {
            return std::unexpected(Err{
                DagBuildErrorKind::MaxDepthExceeded,
                std::string{"DAG no reduction progress at level "}
                    + std::to_string(lodLevel)
                    + " (prev=" + std::to_string(prevLevelCount)
                    + " cur="  + std::to_string(levelCount)
                    + " residual>threshold=" + std::to_string(opts.rootClusterThreshold)
                    + ")",
                lodLevel,
            });
        }

        // --- Build adjacency for this level. We pass the container + slice
        // offset/count rather than a span: during this level's loop body we
        // will push_back new clusters onto `out.clusters`, which could
        // invalidate any raw pointer held across those mutations. ---
        AdjacencyCSR csr;
        {
            std::string detail;
            if (!build_adjacency(out.clusters,
                                 static_cast<std::size_t>(levelStart),
                                 static_cast<std::size_t>(levelCount),
                                 opts.adjacencyWeldEpsilon, csr, detail)) {
                return std::unexpected(Err{
                    DagBuildErrorKind::AdjacencyGraphBuildFailed,
                    std::move(detail),
                    lodLevel,
                });
            }
        }

        // --- Decide group count. Ceil-divide but floor at 1. ---
        // Casts are safe: `levelCount` <= int32 max (validated via CSR
        // overflow guard above), targetGroupSize > 0 (validated up front).
        std::uint32_t nparts = (levelCount + opts.targetGroupSize - 1u)
            / opts.targetGroupSize;
        if (nparts < 1u) nparts = 1u;
        // Clamp to max(levelCount-1, 1) — METIS requires nparts <= nvtxs.
        // Single-vertex graphs get special-cased below anyway.
        if (nparts > levelCount) nparts = levelCount;

        // --- METIS (skip if nparts == 1 or nvtxs == 1: trivial partition). ---
        std::vector<idx_t> parts(levelCount, 0);
        if (nparts > 1u && levelCount > 1u) {
            std::string detail;
            if (!metis_partition(csr, static_cast<idx_t>(nparts),
                                 opts.deterministicSeed, parts, detail)) {
                return std::unexpected(Err{
                    DagBuildErrorKind::MetisPartitionFailed,
                    std::move(detail),
                    lodLevel,
                });
            }
        }

        // --- Bucket cluster indices by group id. std::map for determinism:
        // METIS numbers groups 0..nparts-1 but some may be empty. We iterate
        // the ordered map so `newGroupIndex` follows the numeric group id. ---
        std::map<idx_t, std::vector<std::uint32_t>> groups;
        for (std::size_t i = 0; i < levelCount; ++i) {
            groups[parts[i]].push_back(static_cast<std::uint32_t>(i));
        }

        // --- Build the global "vertex -> set of METIS groups that touch
        // it" map for THIS level. A welded vertex referenced by clusters
        // in >=2 different groups is a group-external boundary: any such
        // vertex must be locked during per-group simplify so the
        // neighbour group's simplify (at the same level) agrees on
        // geometry, letting runtime mixed-LOD cuts stay crack-free.
        //
        // Quantization uses the same `adjacencyWeldEpsilon` and the shared
        // `simplify_quantize_coord` helper so the keys here line up
        // bit-exactly with the keys emitted inside `weld_group()` below.
        // Saturated (Inf/out-of-range) vertices are excluded just like
        // the adjacency builder does — those can't legitimately map to
        // a lockable welded slot. ---
        std::map<WeldKey, std::set<idx_t>> cellToGroups;
        {
            const float invEps = 1.0f / opts.adjacencyWeldEpsilon;
            constexpr std::int32_t kSatHi = std::numeric_limits<std::int32_t>::max();
            constexpr std::int32_t kSatLo = std::numeric_limits<std::int32_t>::min();
            for (std::size_t ci = 0; ci < levelCount; ++ci) {
                const ClusterData& c = out.clusters[levelStart + ci];
                const idx_t gid = parts[ci];
                for (const glm::vec3& p : c.positions) {
                    const WeldKey key{
                        quantize_coord(p.x, invEps),
                        quantize_coord(p.y, invEps),
                        quantize_coord(p.z, invEps),
                    };
                    const bool saturated =
                        (key[0] == kSatHi || key[0] == kSatLo ||
                         key[1] == kSatHi || key[1] == kSatLo ||
                         key[2] == kSatHi || key[2] == kSatLo);
                    if (saturated) continue;
                    cellToGroups[key].insert(gid);
                }
            }
        }

        // --- Per-group simplify + re-cluster. Emits new clusters at the
        // next LOD level, wires parent/child pointers. ---
        const std::uint32_t nextLodLevel = lodLevel + 1u;
        // Overflow guard: we index cluster IDs via uint32_t in DagNode.
        // On a 64-bit build `out.clusters.size()` is size_t, so bail cleanly
        // if the growing DAG ever exceeds the 32-bit id space.
        constexpr std::size_t kU32Max =
            static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max());
        {
            const std::size_t rawNewLevelStart = out.clusters.size();
            if (rawNewLevelStart > kU32Max) {
                return std::unexpected(Err{
                    DagBuildErrorKind::AdjacencyGraphBuildFailed,
                    std::string{"total cluster count exceeds uint32 range ("}
                        + std::to_string(rawNewLevelStart) + ")",
                    lodLevel,
                });
            }
        }
        std::uint32_t newLevelStart = static_cast<std::uint32_t>(out.clusters.size());
        std::uint32_t newLevelCount = 0u;

        // We allocate a provisional group id per group BEFORE emitting new
        // clusters so the input clusters in this group can record their
        // parent group id. Our "group id" here encodes as the index of the
        // FIRST new cluster emitted by the group; children of a cluster X
        // at level L are the subset of new clusters at level L+1 whose
        // `parentGroupId` is the first-new-cluster-index of the group that
        // claimed X. This collapses "group id" and "first child node" into
        // one value, which is exactly what the on-disk MpDagNode format
        // wants later (M1c). For groups that emit no new clusters we use
        // UINT32_MAX — degenerate group, children still point upward but
        // there's no higher-level cluster to blame.

        // Track: for each input cluster in this level (by index within
        // the current-level slice), which group claimed it.
        std::vector<idx_t> inputClusterGroupId(levelCount,
            static_cast<idx_t>(-1));
        for (const auto& [gid, members] : groups) {
            for (std::uint32_t m : members) {
                if (static_cast<std::size_t>(m) < levelCount) {
                    inputClusterGroupId[m] = gid;
                }
            }
        }

        // Map from group id -> first-new-cluster-index (aka parentGroupId
        // for every child of this group). UINT32_MAX until we emit.
        std::map<idx_t, std::uint32_t> groupFirstNewCluster;

        for (const auto& [gid, members] : groups) {
            if (members.empty()) {
                groupFirstNewCluster[gid] = UINT32_MAX;
                continue;
            }
            // Pack group member ClusterData into a contiguous span.
            std::vector<ClusterData> groupClusters;
            groupClusters.reserve(members.size());
            for (std::uint32_t m : members) {
                // `m` is a local level index; out.clusters[levelStart + m]
                // is the corresponding cluster in the flat store.
                groupClusters.push_back(out.clusters[levelStart + m]);
            }

            // Compute group input tri count for an informed auto-target.
            std::size_t groupInputTris = 0u;
            for (const auto& gc : groupClusters) {
                groupInputTris += gc.triangles.size() / 3u;
            }

            // Build simplify options for this group.
            //
            // lockBorder=false alone would let the simplifier collapse the
            // group's external boundary, cracking adjacent groups at
            // mixed-LOD cuts at runtime. lockBorder=true in turn locks
            // every "open" edge (every edge of a closed shell counts when
            // the group is small enough to not wrap around itself), which
            // on DamagedHelmet stalls simplification outright.
            //
            // The correct Nanite-style behaviour is to lock ONLY the
            // vertices that are shared with OTHER METIS groups at this
            // level — group-external boundaries — leaving group-internal
            // boundary vertices free to collapse. We compute such a mask
            // below via `weld_group()` + the `cellToGroups` table, then
            // feed it to simplify() which routes through
            // `meshopt_simplifyWithAttributes` whenever a mask is present.
            //
            // useAbsoluteError=false + maxError=1.0: let the targetIndexCount
            // drive the reduction (relative error = 100% of AABB = uncapped).
            // The caller's opts.maxErrorPerLevel is recorded separately as
            // the DAG's accumulated-error metadata, not as a simplify budget.
            SimplifyOptions simOpts = opts.simplifyOpts;
            simOpts.maxError        = 1.0f;   // relative, effectively uncapped
            simOpts.weldEpsilon     = opts.adjacencyWeldEpsilon;
            simOpts.lockBorder      = false;  // mask-driven lock replaces flag
            simOpts.useAbsoluteError = false;
            {
                // targetIndexCount = simplifyRatio * groupInputTris * 3,
                // rounded down to a multiple of 3, with a minimum of 3.
                const double target = static_cast<double>(groupInputTris)
                    * static_cast<double>(opts.simplifyRatio)
                    * 3.0;
                std::size_t targetIdx = static_cast<std::size_t>(target);
                targetIdx = (targetIdx / 3u) * 3u;
                if (targetIdx < 3u) targetIdx = 3u;
                simOpts.targetIndexCount = targetIdx;
            }

            // --- Compute the group-external boundary lock mask. Weld the
            // group's positions (using the SAME quantizer as simplify())
            // so welded indices align byte-for-byte with simplify()'s
            // internal table, then mark every welded vertex whose grid
            // cell is referenced by >=2 METIS groups as locked.
            //
            // On a malformed group `weld_group()` surfaces the same error
            // kind simplify() would produce — propagate with SimplifyFailed
            // context so downstream reporting stays uniform. ---
            std::vector<std::uint8_t> lockMask;
            {
                auto welded = weld_group(
                    std::span<const ClusterData>(groupClusters),
                    opts.adjacencyWeldEpsilon);
                if (!welded.has_value()) {
                    return std::unexpected(Err{
                        DagBuildErrorKind::SimplifyFailed,
                        std::string{"level "} + std::to_string(lodLevel)
                            + " group gid=" + std::to_string(gid)
                            + ": weld_group failed: "
                            + simplifyErrorKindString(welded.error().kind)
                            + ": " + welded.error().detail,
                        lodLevel,
                    });
                }
                lockMask.assign(welded->positions.size(), 0u);
                for (const auto& [key, weldedIdx] : welded->keyToWeldedIndex) {
                    auto it = cellToGroups.find(key);
                    if (it == cellToGroups.end()) continue;
                    if (it->second.size() >= 2u) {
                        if (static_cast<std::size_t>(weldedIdx) < lockMask.size()) {
                            lockMask[weldedIdx] = 1u;
                        }
                    }
                }
                simOpts.lockMask = std::move(lockMask);
            }

            auto simResult = simplify(
                std::span<const ClusterData>(groupClusters),
                simOpts);
            if (!simResult.has_value()) {
                return std::unexpected(Err{
                    DagBuildErrorKind::SimplifyFailed,
                    std::string{"level "} + std::to_string(lodLevel)
                        + " group gid=" + std::to_string(gid)
                        + " ("   + std::to_string(members.size())
                        + " clusters, " + std::to_string(groupInputTris)
                        + " tris): " + simplifyErrorKindString(simResult.error().kind)
                        + ": " + simResult.error().detail,
                    lodLevel,
                });
            }

            // Capture the absolute world-space error that simplify actually
            // delivered BEFORE we move the SimplifiedGroup's buffers into
            // the re-cluster input. This feeds the new clusters'
            // `maxSimplificationError` below.
            const float groupAchievedError = simResult->achievedError;

            // Re-cluster the simplified group. Propagate clusterOpts as-is;
            // the level-to-level knobs (128/128/0.5) stay constant.
            IngestedMesh simMesh = simplified_to_mesh(std::move(*simResult));
            // If the simplified mesh is degenerate (e.g. everything collapsed
            // to a single triangle with a locked border that killed all
            // interior edges), ClusterBuilder would error with EmptyInput.
            // That's a legitimate no-op group — skip, leaving these input
            // clusters parented to UINT32_MAX.
            if (simMesh.indices.size() < 3u || simMesh.positions.empty()) {
                groupFirstNewCluster[gid] = UINT32_MAX;
                continue;
            }

            ClusterBuilder rebuilder;
            auto newClusters = rebuilder.build(simMesh, opts.clusterOpts);
            if (!newClusters.has_value()) {
                // EmptyInput from ClusterBuilder after simplify is a
                // tolerable no-op (tiny residual geometry); non-empty
                // errors are hard failures.
                if (newClusters.error().kind == ClusterBuildErrorKind::EmptyInput) {
                    groupFirstNewCluster[gid] = UINT32_MAX;
                    continue;
                }
                return std::unexpected(Err{
                    DagBuildErrorKind::ClusterRebuildFailed,
                    std::string{"level "} + std::to_string(nextLodLevel)
                        + " group gid=" + std::to_string(gid)
                        + ": " + clusterBuildErrorKindString(newClusters.error().kind)
                        + ": " + newClusters.error().detail,
                    lodLevel,
                });
            }

            // Compute the max child error for this group (max over member
            // maxSimplificationError). This becomes the "error to reach
            // this level of detail" for the new clusters.
            float childMaxErr = 0.0f;
            for (std::uint32_t m : members) {
                const float e = out.clusters[levelStart + m].maxSimplificationError;
                if (std::isfinite(e) && e > childMaxErr) childMaxErr = e;
            }

            // `groupAchievedError` (captured above, before simMesh move)
            // feeds the new clusters' monotonic-max error below.

            // --- Record first-new-cluster index for this group. ---
            // Overflow guard on uint32 cluster id space (same bound as
            // newLevelStart above — belt + suspenders inside the group loop).
            {
                const std::size_t rawFirstNewIdx = out.clusters.size();
                if (rawFirstNewIdx > kU32Max) {
                    return std::unexpected(Err{
                        DagBuildErrorKind::AdjacencyGraphBuildFailed,
                        std::string{"total cluster count exceeds uint32 range ("}
                            + std::to_string(rawFirstNewIdx) + ")",
                        lodLevel,
                    });
                }
            }
            const std::uint32_t firstNewIdx =
                static_cast<std::uint32_t>(out.clusters.size());
            groupFirstNewCluster[gid] = firstNewIdx;

            // Pick the group's representative material: use the first
            // member's material index. Groups generally correspond to a
            // spatially-coherent slice of the surface, so the first
            // cluster's material is a good enough stand-in for the whole
            // simplified parent (identical to ClusterBuilder's
            // first-triangle-wins policy at the leaf level).
            const std::uint32_t groupMaterialIdx =
                out.clusters[levelStart + members.front()].materialIndex;

            // --- Append new clusters + nodes. ---
            for (auto& nc : *newClusters) {
                nc.dagLodLevel = nextLodLevel;
                nc.materialIndex = groupMaterialIdx;
                // Propagate the ACTUAL achieved error from simplify(),
                // accumulated monotonically up the DAG. The max() ensures a
                // parent cluster's error envelope strictly contains all its
                // children's envelopes (required by runtime LOD selection).
                // `opts.maxErrorPerLevel` is the caller's per-level BUDGET
                // intent; using it as the runtime-visible error would lie
                // about what simplify actually delivered.
                const float achieved =
                    std::isfinite(groupAchievedError) ? groupAchievedError : 0.0f;
                nc.maxSimplificationError = std::max(childMaxErr, achieved);

                // Overflow guard on cluster id assignment.
                {
                    const std::size_t rawClusterId = out.clusters.size();
                    if (rawClusterId > kU32Max) {
                        return std::unexpected(Err{
                            DagBuildErrorKind::AdjacencyGraphBuildFailed,
                            std::string{"total cluster count exceeds uint32 range ("}
                                + std::to_string(rawClusterId) + ")",
                            lodLevel,
                        });
                    }
                }
                DagNode nnode;
                nnode.clusterId      = static_cast<std::uint32_t>(out.clusters.size());
                nnode.parentGroupId  = UINT32_MAX;  // filled next iteration when this level becomes children
                nnode.firstChildNode = UINT32_MAX;
                nnode.childCount     = 0u;
                nnode.lodLevel       = nextLodLevel;
                nnode.boundsSphere   = nc.boundsSphere;
                nnode.maxError       = nc.maxSimplificationError;
                out.nodes.push_back(nnode);
                out.clusters.push_back(std::move(nc));
                ++newLevelCount;
            }
        }

        // --- Now wire the two sides of the parent/child links:
        //
        //   * Every CURRENT-level cluster gets `parentGroupId` =
        //     groupFirstNewCluster[inputClusterGroupId[i]].
        //
        //   * Every NEW-level cluster gets firstChildNode/childCount over
        //     the contiguous range of current-level indices that share
        //     its `parentGroupId`. To keep the children contiguous we
        //     RE-ORDER the current level's slice of out.clusters/out.nodes
        //     below by parent group id. The input order inside a group is
        //     already deterministic (ordered-map iteration of groups +
        //     stable within-group order from curView scan).
        // ---

        // Compute a stable permutation of the current level's nodes by
        // (parentGroupId, original-index). Groups that produced no new
        // cluster (groupFirstNewCluster == UINT32_MAX) get sorted last;
        // their parents stay UINT32_MAX.
        std::vector<std::uint32_t> perm(levelCount);
        for (std::uint32_t i = 0; i < levelCount; ++i) perm[i] = i;
        // Compute each original index's parent-group-id key.
        std::vector<std::uint32_t> parentGid(levelCount, UINT32_MAX);
        for (std::uint32_t i = 0; i < levelCount; ++i) {
            const idx_t gid = inputClusterGroupId[i];
            auto it = groupFirstNewCluster.find(gid);
            if (it != groupFirstNewCluster.end()) {
                parentGid[i] = it->second;
            }
        }
        // Stable sort: UINT32_MAX (orphans) go last, lower gids first.
        std::stable_sort(perm.begin(), perm.end(),
            [&parentGid](std::uint32_t a, std::uint32_t b) {
                return parentGid[a] < parentGid[b];
            });

        // Apply permutation in place on the current level's slice of
        // out.clusters and out.nodes. Build a fresh reordered vector and
        // swap in.
        std::vector<ClusterData> reorderedClusters;
        std::vector<DagNode>     reorderedNodes;
        reorderedClusters.reserve(levelCount);
        reorderedNodes.reserve(levelCount);
        for (std::uint32_t p : perm) {
            reorderedClusters.push_back(
                std::move(out.clusters[levelStart + p]));
            reorderedNodes.push_back(out.nodes[levelStart + p]);
        }
        for (std::uint32_t i = 0; i < levelCount; ++i) {
            out.clusters[levelStart + i] = std::move(reorderedClusters[i]);
            out.nodes[levelStart + i]    = reorderedNodes[i];
            out.nodes[levelStart + i].clusterId = levelStart + i;
            // Patch parentGroupId for each current-level node now that
            // their positions are final.
            const std::uint32_t origIdx = perm[i];
            out.nodes[levelStart + i].parentGroupId = parentGid[origIdx];
        }

        // --- Populate firstChildNode / childCount on the new-level nodes. ---
        //
        // Invariant: we re-ordered the current level's slice of
        // out.clusters / out.nodes BEFORE writing these parent->child
        // pointers. Writing `firstChildNode = levelStart + i` AFTER the
        // reorder is what keeps `clusterId == flat-index` aligned: every
        // node's position in `out.nodes` matches its `clusterId`, and the
        // contiguous child-range recorded on a new-level node points at
        // the reordered (now stable) slice. Any attempt to patch these
        // pointers before the reorder would capture pre-sort indices and
        // break the parallel-array guarantee downstream consumers rely on.
        // We sweep the newly permuted current level; runs of equal
        // parentGroupId (that is not UINT32_MAX) belong to a single new
        // cluster (the one at `parentGroupId`). For multi-cluster groups
        // (those that emitted >1 new cluster via the re-cluster step), all
        // new clusters of that group share the same set of children —
        // there's no way to disambiguate which simplified cluster a
        // particular source cluster belongs to. We therefore record the
        // child range on the FIRST new cluster of each group and leave
        // subsequent same-group new clusters childless (childCount=0,
        // firstChildNode=UINT32_MAX). This matches Nanite's DAG
        // interpretation: the DAG is a *logical* grouping, children are
        // associated with the group, not individual cluster splits.
        {
            std::uint32_t i = 0;
            while (i < levelCount) {
                const std::uint32_t pgid = out.nodes[levelStart + i].parentGroupId;
                if (pgid == UINT32_MAX) { ++i; continue; }
                std::uint32_t j = i;
                while (j < levelCount &&
                       out.nodes[levelStart + j].parentGroupId == pgid) {
                    ++j;
                }
                // Children are [levelStart + i, levelStart + j). Record on
                // the new-level node at index `pgid`. Tighten the bounds
                // check to the newly-emitted level slice: legitimate
                // parentGroupIds point to cluster ids in
                // [newLevelStart, newLevelStart + newLevelCount). Anything
                // else (including stale/corrupt ids that accidentally
                // falls inside out.nodes.size() at large) is silently
                // dropped to keep write-back tight.
                if (pgid >= newLevelStart &&
                    pgid <  newLevelStart + newLevelCount) {
                    out.nodes[pgid].firstChildNode = levelStart + i;
                    out.nodes[pgid].childCount     = j - i;
                }
                i = j;
            }
        }

        // --- Advance to next level. `prevLevelCount` retains the OLD
        // `levelCount` so the no-progress check at the top of the next
        // iteration can compare current vs. immediately-preceding. ---
        prevLevelCount = levelCount;
        levelStart     = newLevelStart;
        levelCount     = newLevelCount;
        lodLevel       = nextLodLevel;

        // Degenerate: if the level produced zero new clusters (every group
        // was a no-op), we cannot make further progress. Treat the
        // residual current-level as the roots and bail. This only happens
        // on truly disconnected inputs where no simplify group survived.
        if (levelCount == 0u) {
            // Walk back one level — the "previous" level is now the roots.
            // We set maxLodLevel to `lodLevel - 1` since the nextLodLevel
            // level is empty and contains no clusters.
            // Find the previous level's slice: it starts at `oldStart`.
            // But the loop already re-entered with the new (empty) level;
            // we need the previously-advanced state. Simpler: re-derive.
            // We know out.clusters.size() == old levelStart at this point
            // because nothing was appended. The loop condition
            // `levelCount > rootClusterThreshold` now fails (0 < any),
            // so break. The old clusters remain untouched as roots.
            break;
        }
    }

    // --- Finalize counts. Roots are the final level (levelCount clusters
    // starting at levelStart). Their lodLevel gives us maxLodLevel. ---
    if (levelCount == 0u) {
        // Degenerate: earlier-break path. Use nodes.size() and work
        // backwards to count the deepest non-empty level.
        if (out.nodes.empty()) {
            return std::unexpected(Err{
                DagBuildErrorKind::EmptyInput,
                std::string{"DAG build produced zero nodes"},
                lodLevel,
            });
        }
        const std::uint32_t deepestLod = out.nodes.back().lodLevel;
        out.maxLodLevel = deepestLod;
        std::uint32_t rc = 0u;
        for (const auto& n : out.nodes) {
            if (n.lodLevel == deepestLod) ++rc;
        }
        out.rootCount = rc;
    } else {
        out.maxLodLevel = lodLevel;
        out.rootCount   = levelCount;
    }

    return out;
}

} // namespace enigma::mpbake
