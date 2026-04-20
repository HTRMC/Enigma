#pragma once

// ClusterBuilder.h
// =================
// Stage 2 of `enigma-mpbake` (M1b.2): take the consolidated `IngestedMesh`
// emitted by `GltfIngest` and cluster it into fixed-size meshlets via
// `meshopt_buildMeshlets`. Each cluster carries its own local vertex table
// (positions/normals/uvs) plus 8-bit triangle indices, ready for:
//   - `Simplify` (M1b.3) — edge-collapse within cluster groups,
//   - `DagBuilder` (M1b.4) — METIS k-way grouping into DAG nodes,
//   - `PageWriter` (M1b.5) — per-DAG-group compressed streaming pages,
//   - the runtime cull + rasterize path at M3.
//
// Contract
// --------
// `ClusterBuilder::build` is stateless and deterministic: the same
// `IngestedMesh` + the same `ClusterBuildOptions` always produce a
// byte-identical `std::vector<ClusterData>`. The determinism test in
// `tests/cluster_builder_test.cpp` builds twice and hashes the output.
//
// The runtime-side `src/renderer/Meshlet.h` is a DIFFERENT type (64/124
// GPU-side layout); do NOT alias it here. The offline DAG uses a 128/128
// target per plan §3.M1.

#include "GltfIngest.h"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <vector>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace enigma::mpbake {

// Knobs fed into `meshopt_buildMeshlets`. Defaults match plan §3.M1
// (128 tris / 128 verts). `coneWeight` balances vertex locality against
// normal-cone coherence — 0.5 matches the meshopt reference sample.
//
// meshopt v0.21 enforces `targetMaxVertices <= 255` and
// `targetMaxTriangles <= 512 && divisible by 4`. `build()` validates
// these and returns `EmptyInput` if out-of-range values are supplied.
struct ClusterBuildOptions {
    std::uint32_t targetMaxVertices  = 128u;
    std::uint32_t targetMaxTriangles = 128u;
    float         coneWeight         = 0.5f;
};

// One cluster ~= one meshlet in meshopt parlance. Local vertex stream +
// 8-bit triangle indices (into positions[]) + bounds for runtime cull.
//
// `boundsSphere` is `glm::vec4` (center.xyz + radius) — DIFFERENT from
// the on-disk `MpDagNode::boundsSphere` (float[4]); in-memory uses glm
// freely. The f32[4] packing happens later in PageWriter.
//
// Simplification bookkeeping (`maxSimplificationError`, `dagLodLevel`) is
// zero-initialized here and filled by M1b.3 Simplify / M1b.4 DagBuilder.
struct ClusterData {
    // Local vertex table. `positions.size() == normals.size() == uvs.size()`.
    std::vector<glm::vec3>    positions;
    std::vector<glm::vec3>    normals;
    std::vector<glm::vec2>    uvs;

    // 3 * tri_count local indices; each byte in range [0, positions.size()).
    std::vector<std::uint8_t> triangles;

    // Bounding sphere for runtime frustum / HiZ cull (center.xyz + radius).
    glm::vec4 boundsSphere{ 0.0f, 0.0f, 0.0f, 0.0f };

    // Normal cone for backface-cone cull (meshopt convention;
    // see `meshopt_computeMeshletBounds` docs).
    glm::vec3 coneApex{ 0.0f };
    glm::vec3 coneAxis{ 0.0f };
    float     coneCutoff{ 1.0f };    // cos(halfAngle); 1.0 = always-culled placeholder

    // Filled by `Simplify` (M1b.3) and `DagBuilder` (M1b.4).
    float         maxSimplificationError = 0.0f;
    std::uint32_t dagLodLevel            = 0u;

    // Index into the scene material buffer. Default 0 when no source
    // material is known (e.g. first glTF primitive's default material, or
    // groups formed from mixed-material inputs — first-child-wins).
    std::uint32_t materialIndex          = 0u;
};

// Classifier for every failure `ClusterBuilder::build` may surface.
enum class ClusterBuildErrorKind {
    EmptyInput,           // IngestedMesh has zero triangles
    IndexOutOfRange,      // index >= positions.size()
    MeshoptBuildFailed,   // meshopt_buildMeshlets returned 0 on non-empty input
    ClusterOverflow,      // a produced meshlet exceeds opts.targetMax* limits
};

// Rich error payload. `triangleOffset` is the input-index offset of the
// first triangle that triggered `IndexOutOfRange`; otherwise 0.
struct ClusterBuildError {
    ClusterBuildErrorKind kind;
    std::string           detail;
    std::size_t           triangleOffset = 0u;
};

// Stable string for `ClusterBuildErrorKind`. Guaranteed never null.
const char* clusterBuildErrorKindString(ClusterBuildErrorKind kind) noexcept;

// ClusterBuilder
// --------------
// Stateless. Construct + `build()`. Deterministic given fixed input + opts.
class ClusterBuilder {
public:
    ClusterBuilder()  = default;
    ~ClusterBuilder() = default;

    // Cluster the ingested mesh into fixed-size meshlets. Success returns
    // one `ClusterData` per meshlet in meshopt's output order (which is
    // itself deterministic given identical inputs + cone_weight).
    std::expected<std::vector<ClusterData>, ClusterBuildError>
    build(const IngestedMesh& in, const ClusterBuildOptions& opts);
};

} // namespace enigma::mpbake
