// Simplify.cpp
// ============
// M1b.3 implementation. See Simplify.h for the contract. This TU:
//   1. Merges every cluster in the input group into a single triangle soup.
//   2. Welds positions using a quantized integer grid keyed by
//      `weldEpsilon` — ordered `std::map` (determinism!), normals/uvs
//      averaged across welded duplicates. The weld pass is factored into
//      `weld_group_impl` so DagBuilder can reuse exactly the same welded
//      vertex ordering when pre-computing per-vertex lock masks.
//   3. Runs either `meshopt_simplify` or `meshopt_simplifyWithAttributes`.
//      Switches to the "WithAttributes" entrypoint (with attribute_count=0
//      and a non-null vertex_lock) when the caller supplies an explicit
//      per-vertex lock mask. Otherwise falls back to the binary
//      `meshopt_SimplifyLockBorder` flag for border preservation.
//   4. Compacts the output via a single index-buffer pass so we can keep
//      the position / normal / uv streams in lockstep.
//
// Hardening follows M1b.2 patterns: overflow guards on scratch allocations,
// srcIdx bounds checks, NaN/Inf sanitization on the error metric, knob
// validation up front.

#include "Simplify.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <meshoptimizer.h>

#include <glm/geometric.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

namespace enigma::mpbake {

const char* simplifyErrorKindString(SimplifyErrorKind kind) noexcept {
    switch (kind) {
        case SimplifyErrorKind::EmptyGroup:             return "EmptyGroup";
        case SimplifyErrorKind::DegenerateAfterWeld:    return "DegenerateAfterWeld";
        case SimplifyErrorKind::MeshoptSimplifyFailed:  return "MeshoptSimplifyFailed";
        case SimplifyErrorKind::OptionsOutOfRange:      return "OptionsOutOfRange";
    }
    return "Unknown";
}

// Public API: quantize_coord matches the internal file-local helper used by
// Simplify's merge-and-weld pass. Exposed so DagBuilder can align its
// lock-mask computation with simplify()'s welded-vertex indexing.
std::int32_t simplify_quantize_coord(float c, float invEpsilon) noexcept {
    if (!std::isfinite(c)) {
        return 0;
    }
    const double scaled = static_cast<double>(c) * static_cast<double>(invEpsilon);
    if (scaled >=  static_cast<double>(std::numeric_limits<std::int32_t>::max())) {
        return  std::numeric_limits<std::int32_t>::max();
    }
    if (scaled <=  static_cast<double>(std::numeric_limits<std::int32_t>::min())) {
        return  std::numeric_limits<std::int32_t>::min();
    }
    return static_cast<std::int32_t>(std::llround(scaled));
}

SimplifyWeldKey simplify_weld_key(const glm::vec3& p, float invEpsilon) noexcept {
    return SimplifyWeldKey{
        simplify_quantize_coord(p.x, invEpsilon),
        simplify_quantize_coord(p.y, invEpsilon),
        simplify_quantize_coord(p.z, invEpsilon),
    };
}

namespace {

using WeldKey = SimplifyWeldKey;

// File-local alias matching the original file-local quantizer name so the
// rest of this TU reads unchanged after the rename-to-public.
inline std::int32_t quantize_coord(float c, float invEpsilon) noexcept {
    return simplify_quantize_coord(c, invEpsilon);
}

// Result of the internal weld pass. Extends the public WeldedGroup with the
// per-source-vertex remap that simplify() needs when rewriting cluster
// triangle indices through the welded table. `weldCounts` and
// `degenerateNormalCount` let simplify() finalize averaged normal/uv
// streams exactly once.
struct WeldedGroupInternal {
    std::vector<glm::vec3>     mergedPositions;
    std::vector<glm::vec3>     mergedNormals;
    std::vector<glm::vec2>     mergedUvs;
    std::vector<std::uint32_t> weldCounts;
    std::vector<std::uint32_t> clusterStart;    // size == group.size() + 1
    std::vector<std::uint32_t> remap;           // size == totalVerts
    std::map<WeldKey, std::uint32_t> weldMap;
    std::size_t                totalTris = 0u;
};

// Shared weld-and-merge pass used by both simplify() and weld_group(). The
// only difference between the two public entry points is what the caller
// does with the welded data afterwards.
std::expected<WeldedGroupInternal, SimplifyError>
weld_group_impl(std::span<const ClusterData> group, float weldEpsilon) {
    using Err = SimplifyError;

    if (!(weldEpsilon > 0.0f) || !std::isfinite(weldEpsilon)) {
        return std::unexpected(Err{
            SimplifyErrorKind::OptionsOutOfRange,
            std::string{"weldEpsilon="} + std::to_string(weldEpsilon)
                + " must be finite and > 0",
        });
    }
    if (group.empty()) {
        return std::unexpected(Err{
            SimplifyErrorKind::EmptyGroup,
            std::string{"weld_group() given an empty cluster group"},
        });
    }

    std::size_t totalVerts = 0;
    std::size_t totalTris  = 0;
    for (const ClusterData& c : group) {
        if ((c.triangles.size() % 3u) != 0u) {
            return std::unexpected(Err{
                SimplifyErrorKind::OptionsOutOfRange,
                std::string{"cluster in group has triangles.size()="}
                    + std::to_string(c.triangles.size())
                    + " not a multiple of 3",
            });
        }
        if (c.normals.size() != c.positions.size() ||
            c.uvs.size()     != c.positions.size()) {
            return std::unexpected(Err{
                SimplifyErrorKind::OptionsOutOfRange,
                std::string{"cluster stream size mismatch: positions="}
                    + std::to_string(c.positions.size()) + ", normals="
                    + std::to_string(c.normals.size())   + ", uvs="
                    + std::to_string(c.uvs.size()),
            });
        }
        totalVerts += c.positions.size();
        totalTris  += c.triangles.size() / 3u;
    }
    if (totalTris == 0u || totalVerts == 0u) {
        return std::unexpected(Err{
            SimplifyErrorKind::EmptyGroup,
            std::string{"weld_group() group contains zero triangles"},
        });
    }

    constexpr std::size_t kMaxAlloc = std::size_t{1} << 32;
    if (totalTris > kMaxAlloc / (3u * sizeof(std::uint32_t))) {
        return std::unexpected(Err{
            SimplifyErrorKind::OptionsOutOfRange,
            std::string{"totalTris="} + std::to_string(totalTris)
                + " would overflow merged index buffer",
        });
    }
    if (totalVerts > kMaxAlloc / sizeof(glm::vec3)) {
        return std::unexpected(Err{
            SimplifyErrorKind::OptionsOutOfRange,
            std::string{"totalVerts="} + std::to_string(totalVerts)
                + " would overflow merged position buffer",
        });
    }

    const float invEpsilon = 1.0f / weldEpsilon;

    WeldedGroupInternal w;
    w.mergedPositions.reserve(totalVerts);
    w.mergedNormals.reserve(totalVerts);
    w.mergedUvs.reserve(totalVerts);
    w.weldCounts.reserve(totalVerts);
    w.clusterStart.reserve(group.size() + 1u);
    w.clusterStart.push_back(0u);
    for (const ClusterData& c : group) {
        const std::size_t next =
            static_cast<std::size_t>(w.clusterStart.back()) + c.positions.size();
        if (next > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
            return std::unexpected(Err{
                SimplifyErrorKind::OptionsOutOfRange,
                std::string{"cluster group vertex count exceeds uint32 range"},
            });
        }
        w.clusterStart.push_back(static_cast<std::uint32_t>(next));
    }

    w.remap.assign(totalVerts, 0u);

    for (std::size_t ci = 0; ci < group.size(); ++ci) {
        const ClusterData& c = group[ci];
        const std::uint32_t base = w.clusterStart[ci];
        for (std::size_t v = 0; v < c.positions.size(); ++v) {
            const glm::vec3 p = c.positions[v];
            const WeldKey key{
                quantize_coord(p.x, invEpsilon),
                quantize_coord(p.y, invEpsilon),
                quantize_coord(p.z, invEpsilon),
            };
            if (w.mergedPositions.size() >= static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
                return std::unexpected(Err{SimplifyErrorKind::OptionsOutOfRange,
                    "distinct welded vertex count exceeds uint32 range"});
            }
            auto [it, inserted] = w.weldMap.try_emplace(key,
                static_cast<std::uint32_t>(w.mergedPositions.size()));
            if (inserted) {
                w.mergedPositions.push_back(p);
                w.mergedNormals.push_back(c.normals[v]);
                w.mergedUvs.push_back(c.uvs[v]);
                w.weldCounts.push_back(1u);
            } else {
                const std::uint32_t gi = it->second;
                w.mergedNormals[gi] += c.normals[v];
                w.mergedUvs[gi]     += c.uvs[v];
                ++w.weldCounts[gi];
            }
            w.remap[static_cast<std::size_t>(base) + v] = it->second;
        }
    }

    w.totalTris = totalTris;
    return w;
}

} // namespace

std::expected<WeldedGroup, SimplifyError>
weld_group(std::span<const ClusterData> group, float weldEpsilon) {
    auto r = weld_group_impl(group, weldEpsilon);
    if (!r.has_value()) return std::unexpected(r.error());
    WeldedGroupInternal& w = *r;

    // Finalize averaged streams identically to simplify()'s path.
    for (std::size_t i = 0; i < w.mergedPositions.size(); ++i) {
        const float inv = w.weldCounts[i] > 0u
            ? 1.0f / static_cast<float>(w.weldCounts[i]) : 0.0f;
        w.mergedUvs[i] *= inv;

        const float nLen = glm::length(w.mergedNormals[i]);
        if (nLen > 1e-6f && std::isfinite(nLen)) {
            w.mergedNormals[i] /= nLen;
        } else {
            w.mergedNormals[i] = glm::vec3{ 0.0f, 0.0f, 1.0f };
        }
    }

    // Build a welded-index triangle list over the merged vertices.
    std::vector<std::uint32_t> mergedIndices;
    mergedIndices.reserve(w.totalTris * 3u);
    for (std::size_t ci = 0; ci < group.size(); ++ci) {
        const ClusterData& c = group[ci];
        const std::uint32_t base = w.clusterStart[ci];
        const std::size_t vc = c.positions.size();
        for (std::size_t b = 0; b < c.triangles.size(); ++b) {
            const std::uint8_t localIdx = c.triangles[b];
            if (static_cast<std::size_t>(localIdx) >= vc) {
                return std::unexpected(SimplifyError{
                    SimplifyErrorKind::OptionsOutOfRange,
                    std::string{"cluster "} + std::to_string(ci)
                        + " local triangle index " + std::to_string(localIdx)
                        + " >= cluster vertex count " + std::to_string(vc),
                });
            }
            mergedIndices.push_back(
                w.remap[static_cast<std::size_t>(base) + localIdx]);
        }
    }

    WeldedGroup out;
    out.positions        = std::move(w.mergedPositions);
    out.normals          = std::move(w.mergedNormals);
    out.uvs              = std::move(w.mergedUvs);
    out.indices          = std::move(mergedIndices);
    out.keyToWeldedIndex = std::move(w.weldMap);
    return out;
}

std::expected<SimplifiedGroup, SimplifyError>
simplify(std::span<const ClusterData> group, const SimplifyOptions& opts) {
    using Err = SimplifyError;

    static_assert(sizeof(std::size_t) >= 8, "Simplify requires 64-bit size_t");

    // --- Pre-flight: knob validation matching ClusterBuilder's style. ---
    // weldEpsilon + empty-group + per-cluster stream consistency get
    // re-checked inside `weld_group_impl`; we validate the simplify-only
    // knobs here so the error surface matches the prior M1b.3 contract.
    constexpr std::size_t kMaxTargetIndexCount = std::size_t{1} << 30;
    if (opts.targetIndexCount > kMaxTargetIndexCount) {
        return std::unexpected(Err{
            SimplifyErrorKind::OptionsOutOfRange,
            std::string{"targetIndexCount="} + std::to_string(opts.targetIndexCount)
                + " exceeds 2^30 cap",
        });
    }
    if (!(opts.maxError >= 0.0f) || !std::isfinite(opts.maxError)) {
        return std::unexpected(Err{
            SimplifyErrorKind::OptionsOutOfRange,
            std::string{"maxError="} + std::to_string(opts.maxError)
                + " must be finite and >= 0",
        });
    }

    // --- Merge + weld via the shared helper. This matches the welded
    // layout exposed through `weld_group()` exactly, so a caller-supplied
    // `lockMask` indexed by welded-vertex id lines up with the table
    // meshopt will actually see. ---
    auto welded = weld_group_impl(group, opts.weldEpsilon);
    if (!welded.has_value()) {
        return std::unexpected(welded.error());
    }
    std::vector<glm::vec3>&     mergedPositions = welded->mergedPositions;
    std::vector<glm::vec3>&     mergedNormals   = welded->mergedNormals;
    std::vector<glm::vec2>&     mergedUvs       = welded->mergedUvs;
    std::vector<std::uint32_t>& weldCounts      = welded->weldCounts;
    std::vector<std::uint32_t>& remap           = welded->remap;
    std::vector<std::uint32_t>& clusterStart    = welded->clusterStart;
    const std::size_t totalTris                 = welded->totalTris;

    // Finalize averaged streams. Normals get normalized; UVs divided by
    // the weld count. A zero-length normal falls back to +Z so downstream
    // shading has a defined axis. Threshold raised to 1e-6f to catch
    // practical near-zero cases (e.g. opposing normals that cancel).
    std::uint32_t degenNormalCount = 0u;
    for (std::size_t i = 0; i < mergedPositions.size(); ++i) {
        const float inv = weldCounts[i] > 0u
            ? 1.0f / static_cast<float>(weldCounts[i]) : 0.0f;
        mergedUvs[i] *= inv;

        const float nLen = glm::length(mergedNormals[i]);
        if (nLen > 1e-6f && std::isfinite(nLen)) {
            mergedNormals[i] /= nLen;
        } else {
            mergedNormals[i] = glm::vec3{ 0.0f, 0.0f, 1.0f };
            ++degenNormalCount;
        }
    }

    // --- Build the merged triangle-list index buffer using the remap. ---
    std::vector<std::uint32_t> mergedIndices;
    mergedIndices.reserve(totalTris * 3u);
    for (std::size_t ci = 0; ci < group.size(); ++ci) {
        const ClusterData& c = group[ci];
        const std::uint32_t base = clusterStart[ci];
        const std::size_t vc = c.positions.size();
        for (std::size_t b = 0; b < c.triangles.size(); ++b) {
            const std::uint8_t localIdx = c.triangles[b];
            if (static_cast<std::size_t>(localIdx) >= vc) {
                return std::unexpected(Err{
                    SimplifyErrorKind::OptionsOutOfRange,
                    std::string{"cluster "} + std::to_string(ci)
                        + " local triangle index " + std::to_string(localIdx)
                        + " >= cluster vertex count " + std::to_string(vc),
                });
            }
            mergedIndices.push_back(
                remap[static_cast<std::size_t>(base) + localIdx]);
        }
    }

    // --- Validate the optional per-vertex lock mask. Must align with the
    // welded vertex table — one byte per welded vertex, in the same
    // iteration order as `weld_group()`'s output streams. ---
    if (!opts.lockMask.empty() && opts.lockMask.size() != mergedPositions.size()) {
        return std::unexpected(Err{
            SimplifyErrorKind::OptionsOutOfRange,
            std::string{"lockMask size="} + std::to_string(opts.lockMask.size())
                + " does not match welded vertex count "
                + std::to_string(mergedPositions.size()),
        });
    }

    // --- Choose target index count. "Auto" == halve the triangle count
    // (rounded to a multiple of 3). Cap the caller value at the input
    // index count (meshopt treats a target >= input as a no-op). ---
    const std::size_t mergedTris = mergedIndices.size() / 3u;
    std::size_t targetIndices;
    if (opts.targetIndexCount == 0u) {
        targetIndices = (mergedTris / 2u) * 3u;
        if (targetIndices < 3u) {
            targetIndices = 3u;  // keep at least one triangle
        }
    } else {
        targetIndices = opts.targetIndexCount;
    }
    if (targetIndices > mergedIndices.size()) {
        targetIndices = mergedIndices.size();
    }

    // --- meshopt_simplify (optionally with attributes for lock mask). ---
    //
    // When the caller supplies a `lockMask` it explicitly encodes which
    // welded vertices must not move (typically group-external boundaries
    // shared with neighbouring METIS groups at the same LOD level). In
    // that case we route through `meshopt_simplifyWithAttributes` with
    // `attribute_count == 0` and a non-null `vertex_lock`. The mesh-
    // simplifier then respects the mask verbatim, ignoring any notion of
    // "open boundary" it would otherwise derive from meshopt_SimplifyLockBorder.
    // We therefore clear the LockBorder bit when a mask is present so the
    // two mechanisms don't compete.
    std::vector<std::uint32_t> simplifiedIndices(mergedIndices.size());
    float achievedError = 0.0f;
    const bool haveLockMask = !opts.lockMask.empty();
    const unsigned int simplifyOptions =
        ((opts.lockBorder && !haveLockMask) ? meshopt_SimplifyLockBorder    : 0u)
        | (opts.useAbsoluteError             ? meshopt_SimplifyErrorAbsolute : 0u);
    std::size_t resultCount = 0u;
    if (haveLockMask) {
        resultCount = meshopt_simplifyWithAttributes(
            simplifiedIndices.data(),
            mergedIndices.data(), mergedIndices.size(),
            reinterpret_cast<const float*>(mergedPositions.data()),
            mergedPositions.size(), sizeof(glm::vec3),
            /*vertex_attributes=*/ nullptr,
            /*vertex_attributes_stride=*/ 0u,
            /*attribute_weights=*/ nullptr,
            /*attribute_count=*/ 0u,
            opts.lockMask.data(),
            targetIndices,
            opts.maxError,
            simplifyOptions,
            &achievedError);
    } else {
        resultCount = meshopt_simplify(
            simplifiedIndices.data(),
            mergedIndices.data(), mergedIndices.size(),
            reinterpret_cast<const float*>(mergedPositions.data()),
            mergedPositions.size(), sizeof(glm::vec3),
            targetIndices,
            opts.maxError,
            simplifyOptions,
            &achievedError);
    }

    // Fix B: resultCount==0 on non-empty input is a legitimate result when
    // LockBorder prevents all reduction (fully-boundary group). Return the
    // merged input unchanged with zero achieved error. Only treat as an error
    // when the input itself was somehow empty (defense-in-depth).
    if (resultCount == 0u && mergedIndices.size() >= 3u) {
        SimplifiedGroup outGroup;
        outGroup.positions             = std::move(mergedPositions);
        outGroup.normals               = std::move(mergedNormals);
        outGroup.uvs                   = std::move(mergedUvs);
        outGroup.indices               = std::move(mergedIndices);
        outGroup.achievedError         = 0.0f;
        outGroup.inputTriangleCount    = mergedTris;
        outGroup.outputTriangleCount   = mergedTris;
        outGroup.degenerateNormalCount = degenNormalCount;
        return outGroup;
    }
    if (resultCount == 0u) {
        return std::unexpected(Err{SimplifyErrorKind::MeshoptSimplifyFailed,
            "meshopt_simplify returned 0 indices on input that became empty after welding"});
    }

    // Fix H: sanity-check that meshopt didn't return more indices than the
    // allocated buffer can hold (should never happen, but defence-in-depth).
    if (resultCount > simplifiedIndices.size()) {
        return std::unexpected(Err{SimplifyErrorKind::MeshoptSimplifyFailed,
            std::string{"meshopt_simplify returned resultCount="} + std::to_string(resultCount)
                + " > allocated buffer size " + std::to_string(simplifiedIndices.size())});
    }
    simplifiedIndices.resize(resultCount);

    // Fix A + E: meshopt_simplify's result_error is ALWAYS relative (the
    // meshopt_SimplifyErrorAbsolute flag only affects INPUT target_error).
    // To get absolute world-space units we must multiply by
    // meshopt_simplifyScale per meshopt v0.21 docs.
    // Sanitize raw first (can be NaN on pathological input).
    if (!std::isfinite(achievedError) || achievedError < 0.0f) {
        achievedError = 0.0f;
    }
    // Multiply by scale to convert relative → absolute world-space units.
    {
        const float scale = meshopt_simplifyScale(
            reinterpret_cast<const float*>(mergedPositions.data()),
            mergedPositions.size(), sizeof(glm::vec3));
        if (std::isfinite(scale) && scale > 0.0f) {
            achievedError *= scale;
        }
    }
    // Re-sanitize the product in case scale was pathological.
    if (!std::isfinite(achievedError) || achievedError < 0.0f) {
        achievedError = 0.0f;
    }

    // --- Compact: drop vertices no longer referenced by the simplified
    // index buffer. We roll our own compaction (rather than call
    // `meshopt_optimizeVertexFetch`) so we can propagate the same
    // old→new remap to the normals and UVs streams. Single pass: walk
    // the index buffer, first-hit claims a new slot, subsequent hits
    // reuse it. Deterministic: visitation order follows the index
    // buffer which is itself a fixed function of the input. ---
    std::vector<std::uint32_t> oldToNew(mergedPositions.size(),
        std::numeric_limits<std::uint32_t>::max());
    std::vector<glm::vec3> finalPositions;
    std::vector<glm::vec3> finalNormals;
    std::vector<glm::vec2> finalUvs;
    finalPositions.reserve(mergedPositions.size());
    finalNormals.reserve(mergedPositions.size());
    finalUvs.reserve(mergedPositions.size());

    for (std::size_t i = 0; i < simplifiedIndices.size(); ++i) {
        const std::uint32_t oldIdx = simplifiedIndices[i];
        if (static_cast<std::size_t>(oldIdx) >= mergedPositions.size()) {
            return std::unexpected(Err{
                SimplifyErrorKind::MeshoptSimplifyFailed,
                std::string{"simplify emitted out-of-range index "}
                    + std::to_string(oldIdx)
                    + " >= merged vertex count "
                    + std::to_string(mergedPositions.size()),
            });
        }
        std::uint32_t newIdx = oldToNew[oldIdx];
        if (newIdx == std::numeric_limits<std::uint32_t>::max()) {
            newIdx = static_cast<std::uint32_t>(finalPositions.size());
            oldToNew[oldIdx] = newIdx;
            finalPositions.push_back(mergedPositions[oldIdx]);
            finalNormals.push_back(mergedNormals[oldIdx]);
            finalUvs.push_back(mergedUvs[oldIdx]);
        }
        simplifiedIndices[i] = newIdx;
    }

    if (finalPositions.empty()) {
        return std::unexpected(Err{
            SimplifyErrorKind::MeshoptSimplifyFailed,
            std::string{"compaction produced zero vertices from "}
                + std::to_string(resultCount) + " simplified indices",
        });
    }

    SimplifiedGroup outGroup;
    outGroup.positions             = std::move(finalPositions);
    outGroup.normals               = std::move(finalNormals);
    outGroup.uvs                   = std::move(finalUvs);
    outGroup.indices               = std::move(simplifiedIndices);
    outGroup.achievedError         = achievedError;
    outGroup.inputTriangleCount    = mergedTris;
    outGroup.outputTriangleCount   = outGroup.indices.size() / 3u;
    outGroup.degenerateNormalCount = degenNormalCount;
    return outGroup;
}

} // namespace enigma::mpbake
