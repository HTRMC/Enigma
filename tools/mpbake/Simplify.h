#pragma once

// Simplify.h
// ===========
// Stage 3 of `enigma-mpbake` (M1b.3): the inner step of DagBuilder's
// group → simplify → re-cluster loop. Given a *group* of adjacent clusters,
// merge their triangles into a single weld-deduped soup, then run
// `meshopt_simplify` with `meshopt_SimplifyLockBorder` so the boundary
// between this group and its neighbours stays bit-identical across LOD
// levels (otherwise the DAG's re-clustering cracks).
//
// Scope
// -----
// M1b.3 is the simplify primitive only. DagBuilder (M1b.4) will drive real
// groups. The unit test exercises a synthetic "first N clusters as a group"
// which doesn't guarantee topological adjacency — that's acceptable for
// verifying the primitive in isolation.
//
// Determinism
// -----------
// - `std::map` (ordered) for weld dedup. No `std::unordered_*` anywhere.
// - `meshopt_simplify` is deterministic on identical inputs.
// - Input is iterated in the order the caller passes it via `std::span`.
// - Result hashes bit-identically across two independent calls; the unit
//   test asserts this.

#include "ClusterBuilder.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <map>
#include <span>
#include <string>
#include <vector>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

namespace enigma::mpbake {

// Knobs for one simplify invocation. Defaults match plan §3.M1:
// `maxError = 0.02` world-space units, `targetIndexCount = 0` (auto-half),
// `weldEpsilon = 1e-5` (tight position grid).
//
// `targetIndexCount == 0` means "auto": half the *triangle* count times 3
// (i.e. halve the index buffer). `targetIndexCount > 2^30` is rejected as
// `OptionsOutOfRange` — meshopt's internal allocations would overflow well
// before that bound.
struct SimplifyOptions {
    std::size_t targetIndexCount = 0u;   // 0 means "auto: input_tris / 2 * 3"
    float       maxError         = 0.02f;
    float       weldEpsilon      = 1e-5f;
    // When true (default), passes meshopt_SimplifyLockBorder so the group
    // boundary stays crack-free across adjacent DAG nodes. Set to false
    // inside DagBuilder where inner simplification can collapse freely.
    //
    // When `lockMask` is non-empty this flag is IGNORED — the explicit
    // per-vertex mask takes precedence and the meshopt_SimplifyLockBorder
    // bit is not set.
    bool        lockBorder       = true;
    // When true (default), passes meshopt_SimplifyErrorAbsolute so maxError
    // is interpreted in world-space units. When false, maxError is relative
    // to the mesh's AABB diagonal (meshopt default). DagBuilder sets this
    // to false and uses maxError=1.0 so the index-count target is the
    // primary driver of reduction rather than an error budget.
    bool        useAbsoluteError = true;
    // Optional per-vertex lock mask: 1 = locked (must not move), 0 = free.
    // Size MUST equal the total welded vertex count of the group (see
    // `weld_group()` below). Empty (default) => no per-vertex lock is
    // applied and `lockBorder` governs via meshopt_SimplifyLockBorder.
    // When non-empty, `lockBorder` is ignored and the mask drives locking
    // via `meshopt_simplifyWithAttributes`.
    //
    // Used by DagBuilder to lock only group-external boundaries (vertices
    // shared with other METIS groups at the same level) so mixed-LOD cuts
    // at runtime don't crack, while leaving group-interior boundary edges
    // free to collapse.
    std::vector<std::uint8_t> lockMask;
};

// Output of one simplify invocation. Welded, deduped positions / normals /
// uvs plus a compacted triangle list. `achievedError` is the absolute
// world-space error: meshopt_simplify always returns a *relative* result_error
// even when meshopt_SimplifyErrorAbsolute is set (that flag only affects the
// INPUT target_error interpretation). We convert to absolute by multiplying
// by meshopt_simplifyScale per meshopt v0.21 docs. Telemetry fields carry
// input vs output triangle counts and degenerate-normal diagnostics for
// logging.
struct SimplifiedGroup {
    std::vector<glm::vec3>    positions;
    std::vector<glm::vec3>    normals;
    std::vector<glm::vec2>    uvs;
    std::vector<std::uint32_t> indices;       // triangle list, 3N entries
    float                     achievedError      = 0.0f;
    std::size_t               inputTriangleCount  = 0u;
    std::size_t               outputTriangleCount = 0u;
    std::uint32_t             degenerateNormalCount = 0u; // welded vertices whose normal sum was near-zero (fell back to +Z)
};

// Classifier for every failure `simplify()` may surface.
enum class SimplifyErrorKind {
    EmptyGroup,             // input span is empty
    DegenerateAfterWeld,    // every triangle collapsed to zero-area after weld
    MeshoptSimplifyFailed,  // meshopt_simplify returned 0 on non-empty input
    OptionsOutOfRange,      // weldEpsilon <= 0, targetIndexCount > 2^30, etc.
};

// Rich error payload. `detail` identifies the specific knob or the triangle
// count at the time of failure.
struct SimplifyError {
    SimplifyErrorKind kind;
    std::string       detail;
};

// Stable string for `SimplifyErrorKind`. Guaranteed never null.
const char* simplifyErrorKindString(SimplifyErrorKind kind) noexcept;

// Simplify a group of clusters. See file header for contract. The return
// type is value-on-success / typed-error-on-failure.
//
// LIMITATION: The weld key is position-only. UV seams and normal creases at
// shared positions are silently collapsed (normals/UVs averaged). Attribute
// awareness via meshopt_simplifyWithAttributes is partially wired: when the
// caller supplies `SimplifyOptions::lockMask`, we already route through
// `meshopt_simplifyWithAttributes` with `attribute_count = 0` for the
// per-vertex lock. A future pass can extend the attribute weights to
// encode UV/normal discontinuities without disturbing the lock-mask
// contract.
std::expected<SimplifiedGroup, SimplifyError>
simplify(std::span<const ClusterData> group, const SimplifyOptions& opts);

// Public helper: weld-key type for the position-grid quantizer used by
// Simplify and DagBuilder. Exposed so callers can pre-compute a welded
// vertex id for any input position and align with simplify()'s internal
// welded indexing when building a `SimplifyOptions::lockMask`.
using SimplifyWeldKey = std::array<std::int32_t, 3>;

// Quantize a single float coordinate into a grid cell keyed by
// `1 / weldEpsilon`. Shared with DagBuilder so both sides use byte-
// identical quantization (required for lockMask alignment).
std::int32_t simplify_quantize_coord(float c, float invEpsilon) noexcept;

// Quantize a world-space position into a welded grid key.
SimplifyWeldKey simplify_weld_key(const glm::vec3& p, float invEpsilon) noexcept;

// Output of `weld_group()`. `positions`/`normals`/`uvs` are the welded
// (deduped) vertex streams in the exact order simplify() would produce
// internally. `keyToWeldedIndex` maps from quantized position grid cell
// to welded vertex index — callers building a lock mask can quantize a
// raw position and look up its welded index here.
struct WeldedGroup {
    std::vector<glm::vec3>     positions;
    std::vector<glm::vec3>     normals;
    std::vector<glm::vec2>     uvs;
    std::vector<std::uint32_t> indices;   // triangle list over welded verts
    std::map<SimplifyWeldKey, std::uint32_t> keyToWeldedIndex;
};

// Perform just the merge-and-weld pass over a group of clusters, returning
// the welded streams + the position-cell to welded-index map. Used by
// DagBuilder to compute a group-external boundary lock mask whose indices
// align with simplify()'s internal welded vertex table.
//
// Deterministic and byte-identical to simplify()'s internal weld for the
// same inputs (same `std::map` iteration order, same quantizer, same
// first-hit normal/uv accumulation). If the return is an error, the group
// is malformed (stream size mismatch, non-multiple-of-3 triangles, etc.)
// just as simplify() would report via OptionsOutOfRange.
std::expected<WeldedGroup, SimplifyError>
weld_group(std::span<const ClusterData> group, float weldEpsilon);

} // namespace enigma::mpbake
