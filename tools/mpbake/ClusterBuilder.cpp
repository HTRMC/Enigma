// ClusterBuilder.cpp
// ==================
// M1b.2 implementation. See ClusterBuilder.h for the contract. This TU
// runs `meshopt_buildMeshlets` on the ingested triangle soup, optimizes
// each meshlet's internal vertex/triangle order via
// `meshopt_optimizeMeshlet`, and copies the per-vertex streams into
// per-cluster local tables alongside bounding-sphere + normal-cone data
// produced by `meshopt_computeClusterBounds`.
//
// Determinism: meshopt_buildMeshlets is deterministic given identical
// `indices`, `vertex_positions`, and `cone_weight`. We touch NO hash
// containers and do not parallelize. The unit test builds twice and
// hashes the result to confirm.

#include "ClusterBuilder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <meshoptimizer.h>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace enigma::mpbake {

const char* clusterBuildErrorKindString(ClusterBuildErrorKind kind) noexcept {
    switch (kind) {
        case ClusterBuildErrorKind::EmptyInput:         return "EmptyInput";
        case ClusterBuildErrorKind::IndexOutOfRange:    return "IndexOutOfRange";
        case ClusterBuildErrorKind::MeshoptBuildFailed: return "MeshoptBuildFailed";
        case ClusterBuildErrorKind::ClusterOverflow:    return "ClusterOverflow";
    }
    return "Unknown";
}

std::expected<std::vector<ClusterData>, ClusterBuildError>
ClusterBuilder::build(const IngestedMesh& in, const ClusterBuildOptions& opts) {
    using Err = ClusterBuildError;

    static_assert(sizeof(std::size_t) >= 8, "ClusterBuilder requires 64-bit size_t");

    // --- Pre-flight: validate ClusterBuildOptions knobs against meshopt
    // documented constraints before touching any data. ---
    if (opts.targetMaxVertices == 0u || opts.targetMaxVertices > 255u) {
        return std::unexpected(Err{
            ClusterBuildErrorKind::EmptyInput,
            std::string{"targetMaxVertices="} + std::to_string(opts.targetMaxVertices)
                + " out of meshopt range [1, 255]",
            0u,
        });
    }
    if (opts.targetMaxTriangles == 0u || opts.targetMaxTriangles > 512u
        || (opts.targetMaxTriangles % 4u) != 0u) {
        return std::unexpected(Err{
            ClusterBuildErrorKind::EmptyInput,
            std::string{"targetMaxTriangles="} + std::to_string(opts.targetMaxTriangles)
                + " violates meshopt constraint [4..512, multiple of 4]",
            0u,
        });
    }

    // --- Pre-flight: empty input is a hard error (matches plan's
    // "`EmptyInput` if IngestedMesh has 0 triangles"). ---
    if (in.indices.empty() || in.positions.empty()) {
        return std::unexpected(Err{
            ClusterBuildErrorKind::EmptyInput,
            std::string{ "IngestedMesh has zero triangles or zero vertices" },
            0u,
        });
    }
    if ((in.indices.size() % 3u) != 0u) {
        // GltfIngest guarantees multiple-of-three, but defend anyway so
        // ad-hoc callers (tests, fuzzers) get a clean error, not a UB path.
        return std::unexpected(Err{
            ClusterBuildErrorKind::EmptyInput,
            std::string{ "IngestedMesh index count is not a multiple of 3" },
            0u,
        });
    }

    // --- Pre-flight: every index must point into positions[]. This is
    // defense-in-depth after ingest; failures here point at a corrupted
    // upstream, not a meshopt bug. Scan triangle-wise so we can report a
    // meaningful `triangleOffset`. ---
    const std::size_t vertCount = in.positions.size();
    for (std::size_t t = 0; t < in.indices.size(); t += 3u) {
        const std::uint32_t a = in.indices[t + 0];
        const std::uint32_t b = in.indices[t + 1];
        const std::uint32_t c = in.indices[t + 2];
        if (static_cast<std::size_t>(a) >= vertCount ||
            static_cast<std::size_t>(b) >= vertCount ||
            static_cast<std::size_t>(c) >= vertCount) {
            std::string detail = "triangle at index ";
            detail += std::to_string(t);
            detail += " references vertex out of range (positions.size()=";
            detail += std::to_string(vertCount);
            detail += ")";
            return std::unexpected(Err{
                ClusterBuildErrorKind::IndexOutOfRange,
                std::move(detail),
                t,
            });
        }
    }

    const std::size_t kMaxVerts = static_cast<std::size_t>(opts.targetMaxVertices);
    const std::size_t kMaxTris  = static_cast<std::size_t>(opts.targetMaxTriangles);

    // --- Size the meshopt scratch buffers to the worst-case bound. ---
    const std::size_t maxMeshlets =
        meshopt_buildMeshletsBound(in.indices.size(), kMaxVerts, kMaxTris);

    // security: guard against crafted input driving the bound to SIZE_MAX
    constexpr std::size_t kMaxAlloc = std::size_t{1} << 32;  // 4 GiB hard cap per buffer
    if (kMaxVerts > 0 && maxMeshlets > kMaxAlloc / kMaxVerts) {
        return std::unexpected(Err{
            ClusterBuildErrorKind::MeshoptBuildFailed,
            std::string{"meshopt_buildMeshletsBound produced implausibly large count: "} + std::to_string(maxMeshlets),
            0u,
        });
    }
    if (kMaxTris > 0 && maxMeshlets > kMaxAlloc / (kMaxTris * 3u)) {
        return std::unexpected(Err{
            ClusterBuildErrorKind::MeshoptBuildFailed,
            std::string{"moTris allocation would overflow: maxMeshlets="} + std::to_string(maxMeshlets),
            0u,
        });
    }

    std::vector<meshopt_Meshlet> moMeshlets(maxMeshlets);
    std::vector<unsigned int>    moVerts(maxMeshlets * kMaxVerts);
    std::vector<unsigned char>   moTris(maxMeshlets * kMaxTris * 3u);

    const std::size_t meshletCount = meshopt_buildMeshlets(
        moMeshlets.data(),
        moVerts.data(),
        moTris.data(),
        in.indices.data(),
        in.indices.size(),
        reinterpret_cast<const float*>(in.positions.data()),
        vertCount,
        sizeof(glm::vec3),
        kMaxVerts,
        kMaxTris,
        opts.coneWeight);

    if (meshletCount == 0u) {
        std::string detail = "meshopt_buildMeshlets returned 0 meshlets on ";
        detail += std::to_string(in.indices.size() / 3u);
        detail += " triangles / ";
        detail += std::to_string(vertCount);
        detail += " vertices";
        return std::unexpected(Err{
            ClusterBuildErrorKind::MeshoptBuildFailed,
            std::move(detail),
            0u,
        });
    }

    // --- Pre-build a (sorted-vertex-triple) -> source-triangle-index map so
    // each produced cluster can resolve the source material for its first
    // local triangle. Deterministic via std::map. Empty when the caller did
    // not supply per-triangle material indices (back-compat path). ---
    std::map<std::array<std::uint32_t, 3>, std::uint32_t> triKeyToSrcTri;
    const bool haveMaterialIndices =
        (in.triangleMaterialIndex.size() == in.indices.size() / 3u);
    if (haveMaterialIndices) {
        for (std::size_t t = 0; t < in.indices.size(); t += 3u) {
            std::array<std::uint32_t, 3> key{
                in.indices[t + 0],
                in.indices[t + 1],
                in.indices[t + 2],
            };
            std::sort(key.begin(), key.end());
            // First-writer wins (duplicate triangles are a malformed input
            // edge case; we don't error since meshopt also tolerates them).
            triKeyToSrcTri.emplace(key, static_cast<std::uint32_t>(t / 3u));
        }
    }

    // --- Walk meshopt output in index order, optimize in-place, and copy
    // into ClusterData. Order is deterministic across runs. ---
    std::vector<ClusterData> out;
    out.reserve(meshletCount);

    for (std::size_t mi = 0; mi < meshletCount; ++mi) {
        const meshopt_Meshlet& m = moMeshlets[mi];

        // Overflow guard: meshopt should never exceed the knobs we passed
        // in, but assert defensively (cheap, correctness-critical).
        if (m.vertex_count > kMaxVerts || m.triangle_count > kMaxTris) {
            std::string detail = "meshlet ";
            detail += std::to_string(mi);
            detail += " exceeds configured limits (vertex_count=";
            detail += std::to_string(m.vertex_count);
            detail += "/";
            detail += std::to_string(kMaxVerts);
            detail += ", triangle_count=";
            detail += std::to_string(m.triangle_count);
            detail += "/";
            detail += std::to_string(kMaxTris);
            detail += ")";
            return std::unexpected(Err{
                ClusterBuildErrorKind::ClusterOverflow,
                std::move(detail),
                0u,
            });
        }

        // Rearrange within the meshlet for better rasterizer locality.
        // Operates on the meshlet's slice of moVerts / moTris in place.
        // Free (only touches data we already own).
        meshopt_optimizeMeshlet(
            &moVerts[m.vertex_offset],
            &moTris[m.triangle_offset],
            static_cast<std::size_t>(m.triangle_count),
            static_cast<std::size_t>(m.vertex_count));

        ClusterData cluster;
        cluster.positions.resize(m.vertex_count);
        cluster.normals.resize(m.vertex_count);
        cluster.uvs.resize(m.vertex_count);

        // Copy per-vertex streams. `moVerts[vertex_offset + i]` indexes
        // the parent mesh's position/normal/uv arrays.
        for (std::uint32_t v = 0; v < m.vertex_count; ++v) {
            const std::uint32_t srcIdx = moVerts[m.vertex_offset + v];
            if (static_cast<std::size_t>(srcIdx) >= vertCount) {
                return std::unexpected(Err{
                    ClusterBuildErrorKind::IndexOutOfRange,
                    std::string{"meshlet "} + std::to_string(mi) + " vertex remap produced srcIdx="
                        + std::to_string(srcIdx) + " >= vertCount=" + std::to_string(vertCount)
                        + " (meshopt internal state violation)",
                    0u,
                });
            }
            cluster.positions[v] = in.positions[srcIdx];
            cluster.normals[v]   = in.normals[srcIdx];
            cluster.uvs[v]       = in.uvs[srcIdx];
        }

        // Copy 8-bit local triangle indices. Each triangle is 3 bytes
        // indexing cluster.positions.
        const std::size_t triByteCount = static_cast<std::size_t>(m.triangle_count) * 3u;
        cluster.triangles.resize(triByteCount);
        for (std::size_t b = 0; b < triByteCount; ++b) {
            cluster.triangles[b] = moTris[m.triangle_offset + b];
        }

        // Compute bounds. `meshopt_computeMeshletBounds` consumes the
        // meshopt-native (vertex_offset / triangle_offset / count) layout
        // directly — no temporary buffer needed.
        const meshopt_Bounds mob = meshopt_computeMeshletBounds(
            &moVerts[m.vertex_offset],
            &moTris[m.triangle_offset],
            static_cast<std::size_t>(m.triangle_count),
            reinterpret_cast<const float*>(in.positions.data()),
            vertCount,
            sizeof(glm::vec3));

        cluster.boundsSphere = glm::vec4{ mob.center[0], mob.center[1], mob.center[2], mob.radius };
        cluster.coneApex     = glm::vec3{ mob.cone_apex[0], mob.cone_apex[1], mob.cone_apex[2] };
        cluster.coneAxis     = glm::vec3{ mob.cone_axis[0], mob.cone_axis[1], mob.cone_axis[2] };
        cluster.coneCutoff   = mob.cone_cutoff;

        // Sanitize: degenerate clusters can produce NaN/Inf from meshopt internals.
        if (!std::isfinite(cluster.boundsSphere.w) || cluster.boundsSphere.w < 0.0f) {
            cluster.boundsSphere.w = 0.0f;  // always-visible fallback
        }
        if (!std::isfinite(cluster.coneAxis.x) || !std::isfinite(cluster.coneAxis.y) || !std::isfinite(cluster.coneAxis.z)) {
            cluster.coneAxis   = glm::vec3{0.0f, 0.0f, 1.0f};
            cluster.coneCutoff = 1.0f;  // always-visible sentinel
        }

        // --- Material index: resolve the first local triangle back to a
        // global triangle and look up its source material. meshopt output
        // order is deterministic, so this picks a stable representative for
        // each cluster. Clusters straddling primitive boundaries pick the
        // material of their first triangle (good-enough for visual fidelity
        // because meshopt groups spatially-coherent triangles). ---
        if (haveMaterialIndices && m.triangle_count > 0u && m.vertex_count > 0u) {
            const std::uint8_t li0 = moTris[m.triangle_offset + 0];
            const std::uint8_t li1 = moTris[m.triangle_offset + 1];
            const std::uint8_t li2 = moTris[m.triangle_offset + 2];
            if (li0 < m.vertex_count && li1 < m.vertex_count && li2 < m.vertex_count) {
                std::array<std::uint32_t, 3> key{
                    moVerts[m.vertex_offset + li0],
                    moVerts[m.vertex_offset + li1],
                    moVerts[m.vertex_offset + li2],
                };
                std::sort(key.begin(), key.end());
                auto it = triKeyToSrcTri.find(key);
                if (it != triKeyToSrcTri.end() &&
                    static_cast<std::size_t>(it->second) < in.triangleMaterialIndex.size()) {
                    cluster.materialIndex = in.triangleMaterialIndex[it->second];
                }
            }
        }

        // Simplification bookkeeping stays at defaults — M1b.3 fills it.
        out.push_back(std::move(cluster));
    }

    return out;
}

} // namespace enigma::mpbake
