// GltfIngest.cpp
// ==============
// M1b.1 implementation. See GltfIngest.h for the contract; this TU walks
// the glTF scene graph, rejects non-static / non-triangle content, and
// consolidates every primitive into a single world-space vertex stream.

#include "GltfIngest.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <variant>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>

#include <fastgltf/core.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>

namespace enigma::mpbake {

const char* errorKindString(IngestErrorKind kind) noexcept {
    switch (kind) {
        case IngestErrorKind::FileNotFound:             return "FileNotFound";
        case IngestErrorKind::FileReadFailed:           return "FileReadFailed";
        case IngestErrorKind::GltfParseFailed:          return "GltfParseFailed";
        case IngestErrorKind::NoMeshes:                 return "NoMeshes";
        case IngestErrorKind::SkinnedMeshUnsupported:   return "SkinnedMeshUnsupported";
        case IngestErrorKind::UnsupportedPrimitiveMode: return "UnsupportedPrimitiveMode";
        case IngestErrorKind::MissingPositions:         return "MissingPositions";
        case IngestErrorKind::MissingIndices:           return "MissingIndices";
        case IngestErrorKind::ZeroTriangles:            return "ZeroTriangles";
        case IngestErrorKind::DegenerateGeometry:       return "DegenerateGeometry";
        case IngestErrorKind::AccessorTypeMismatch:     return "AccessorTypeMismatch";
        case IngestErrorKind::IndexRangeExceeded:       return "IndexRangeExceeded";
    }
    return "Unknown";
}

namespace {

// Fold a fastgltf Node transform (TRS or 4x4 matrix) into a column-major
// glm::mat4, matching glTF's right-handed, column-vector convention.
glm::mat4 nodeLocalMatrix(const fastgltf::Node& node) {
    return std::visit([](auto&& t) -> glm::mat4 {
        using T = std::decay_t<decltype(t)>;
        if constexpr (std::is_same_v<T, fastgltf::TRS>) {
            const glm::vec3 tr{ t.translation[0], t.translation[1], t.translation[2] };
            // glTF rotation is stored as (x, y, z, w); glm::quat ctor takes (w, x, y, z).
            const glm::quat rot{ t.rotation[3], t.rotation[0], t.rotation[1], t.rotation[2] };
            const glm::vec3 sc{ t.scale[0], t.scale[1], t.scale[2] };
            glm::mat4 m{ 1.0f };
            m = glm::translate(m, tr);
            m *= glm::mat4_cast(rot);
            m = glm::scale(m, sc);
            return m;
        } else {
            // std::array<num, 16> — column-major per glTF spec.
            static_assert(std::is_same_v<T, fastgltf::Node::TransformMatrix>,
                          "Unexpected Node::transform alternative");
            glm::mat4 m{};
            for (std::size_t i = 0; i < 16; ++i) {
                glm::value_ptr(m)[i] = static_cast<float>(t[i]);
            }
            return m;
        }
    }, node.transform);
}

// Walk the scene graph depth-first starting at `rootIdx` with accumulated
// world-space `parentWorld`. For each mesh node visited, invoke
// `visitor(node, world)`. Iteration order is stable (children in declaration
// order). Implemented iteratively with an explicit stack + visited set:
//   - iterative: avoids stack overflow on pathological glTFs with deep
//     hierarchies (attacker-supplied or otherwise),
//   - visited set: defends against malformed inputs whose `children` arrays
//     form cycles (would otherwise infinitely recurse/loop).
template <typename Visitor>
void walkNodes(const fastgltf::Asset& asset,
               std::size_t rootIdx,
               const glm::mat4& parentWorld,
               Visitor&& visitor) {
    struct Frame { std::size_t nodeIdx; glm::mat4 world; };
    std::vector<Frame> stack;
    stack.reserve(64);
    stack.push_back({rootIdx, parentWorld});

    // Cycle detection. Used only for O(1) insert/lookup; iteration order
    // never observed, so unordered_set is safe for determinism.
    std::unordered_set<std::size_t> visited;

    while (!stack.empty()) {
        const std::size_t nodeIdx = stack.back().nodeIdx;
        const glm::mat4   parent  = stack.back().world;
        stack.pop_back();

        if (nodeIdx >= asset.nodes.size()) continue;
        if (!visited.insert(nodeIdx).second) continue;  // cycle: skip

        const auto& node = asset.nodes[nodeIdx];
        const glm::mat4 world = parent * nodeLocalMatrix(node);

        if (node.meshIndex.has_value()) {
            visitor(node, world);
        }

        // Preserve declaration order by pushing children in reverse so the
        // first child is popped first (LIFO). Critical for determinism.
        for (auto it = node.children.rbegin(); it != node.children.rend(); ++it) {
            stack.push_back({*it, world});
        }
    }
}

} // namespace

std::expected<IngestedMesh, IngestError>
GltfIngest::load(const std::filesystem::path& gltfPath) {
    using Err = IngestError;

    // --- Pre-flight: file existence ---
    std::error_code ec;
    if (!std::filesystem::exists(gltfPath, ec) || ec) {
        return std::unexpected(Err{
            IngestErrorKind::FileNotFound,
            std::string{"file does not exist"},
            gltfPath});
    }

    // --- Parse glTF/GLB ---
    fastgltf::GltfDataBuffer data;
    if (!data.loadFromFile(gltfPath)) {
        // File existed at the pre-flight check but the byte-buffer load still
        // failed — distinct from FileNotFound. Enrich the detail with any
        // filesystem-level signal we can cheaply obtain.
        std::string detail = "GltfDataBuffer::loadFromFile failed";
        std::error_code statEc;
        auto status = std::filesystem::status(gltfPath, statEc);
        if (!statEc) {
            if (std::filesystem::is_directory(status)) {
                detail += " (path is a directory)";
            } else {
                std::error_code sizeEc;
                auto sz = std::filesystem::file_size(gltfPath, sizeEc);
                if (!sizeEc) {
                    detail += " (file_size=" + std::to_string(sz) + ")";
                }
            }
        }
        return std::unexpected(Err{
            IngestErrorKind::FileReadFailed,
            std::move(detail),
            gltfPath});
    }

    fastgltf::Parser parser;
    // SECURITY: LoadExternalBuffers is explicitly NOT set. A malicious glTF with
    // a URI like "file:///C:/sensitive" could make fastgltf read arbitrary files.
    // For M1b.1 we support GLB-embedded buffers only; M1b.2+ may reinstate
    // external buffers behind a containment check that validates resolved URIs
    // are descendants of the glTF's parent directory.
    constexpr auto options = fastgltf::Options::LoadGLBBuffers
                           | fastgltf::Options::GenerateMeshIndices;

    auto result = parser.loadGltf(&data, gltfPath.parent_path(), options);
    if (result.error() != fastgltf::Error::None) {
        std::string msg{ fastgltf::getErrorMessage(result.error()) };
        return std::unexpected(Err{
            IngestErrorKind::GltfParseFailed,
            std::move(msg),
            gltfPath});
    }
    const fastgltf::Asset& asset = result.get();  // Safe: result.error() == None confirmed above

    if (asset.meshes.empty()) {
        return std::unexpected(Err{
            IngestErrorKind::NoMeshes,
            std::string{"glTF contains zero meshes"},
            gltfPath});
    }

    // --- Reject skinned meshes (before we do any work) ---
    // Any node that points at a mesh while also carrying a `skinIndex` is
    // animated and out-of-scope for M1 bake. Additionally, some exporters
    // emit JOINTS_0/WEIGHTS_0 primitive attributes on meshes whose nodes
    // have no `skinIndex`; those are also rejected.
    //
    // Deterministic first-found-wins: scan nodes in declaration order, then
    // primitives in declaration order. No unordered container iteration.
    bool        hasSkinnedMesh      = false;
    std::size_t firstSkinnedMeshIdx = 0;
    std::string firstSkinnedReason;

    for (std::size_t nodeIdx = 0; nodeIdx < asset.nodes.size(); ++nodeIdx) {
        const auto& node = asset.nodes[nodeIdx];
        if (node.meshIndex.has_value() && node.skinIndex.has_value()) {
            hasSkinnedMesh       = true;
            firstSkinnedMeshIdx  = *node.meshIndex;
            firstSkinnedReason   = "node " + std::to_string(nodeIdx) + " binds a skin";
            break;
        }
    }

    if (!hasSkinnedMesh) {
        // Sniff primitive attributes for JOINTS_*/WEIGHTS_*. attr.first may be
        // either std::string or std::pmr::string depending on fastgltf's
        // configuration; fold through std::string_view for portability.
        for (std::size_t meshIdx = 0; meshIdx < asset.meshes.size() && !hasSkinnedMesh; ++meshIdx) {
            const auto& mesh = asset.meshes[meshIdx];
            for (std::size_t primIdx = 0; primIdx < mesh.primitives.size() && !hasSkinnedMesh; ++primIdx) {
                const auto& prim = mesh.primitives[primIdx];
                for (const auto& attr : prim.attributes) {
                    const std::string_view name{ attr.first.data(), attr.first.size() };
                    if (name.starts_with("JOINTS_") || name.starts_with("WEIGHTS_")) {
                        hasSkinnedMesh      = true;
                        firstSkinnedMeshIdx = meshIdx;
                        firstSkinnedReason  = "mesh " + std::to_string(meshIdx) +
                            " primitive " + std::to_string(primIdx) +
                            " has skinning attribute \"" + std::string{ name } + "\"";
                        break;
                    }
                }
            }
        }
    }

    if (hasSkinnedMesh) {
        std::string detail = "mesh[";
        detail += std::to_string(firstSkinnedMeshIdx);
        detail += "]";
        if (!asset.meshes[firstSkinnedMeshIdx].name.empty()) {
            detail += " '";
            detail.append(asset.meshes[firstSkinnedMeshIdx].name.data(),
                          asset.meshes[firstSkinnedMeshIdx].name.size());
            detail += "'";
        }
        detail += " is skinned: ";
        detail += firstSkinnedReason;
        return std::unexpected(Err{
            IngestErrorKind::SkinnedMeshUnsupported,
            std::move(detail),
            gltfPath});
    }

    // --- Walk scenes → nodes; consolidate every triangle primitive ---
    IngestedMesh out{};

    // Tracks any failure from inside the visitor lambda (can't early-return
    // through walkNodes otherwise).
    std::expected<void, Err> visitorStatus{};

    auto primitiveFailed = [&](IngestErrorKind kind, std::string detail) {
        visitorStatus = std::unexpected(Err{ kind, std::move(detail), gltfPath });
    };

    auto processPrimitive =
        [&](std::size_t meshIdx, std::size_t primIdx,
            const fastgltf::Primitive& prim, const glm::mat4& world) {
            // Mesh/primitive label used for every error string.
            auto label = [&]() {
                std::string s = "mesh[";
                s += std::to_string(meshIdx);
                s += "]";
                if (!asset.meshes[meshIdx].name.empty()) {
                    s += " '";
                    s.append(asset.meshes[meshIdx].name.data(),
                             asset.meshes[meshIdx].name.size());
                    s += "'";
                }
                s += " primitive[";
                s += std::to_string(primIdx);
                s += "]";
                return s;
            };

            if (prim.type != fastgltf::PrimitiveType::Triangles) {
                primitiveFailed(IngestErrorKind::UnsupportedPrimitiveMode,
                                label() + " has non-TRIANGLES primitive mode");
                return;
            }

            auto posIt = prim.findAttribute("POSITION");
            if (posIt == prim.attributes.end()) {
                primitiveFailed(IngestErrorKind::MissingPositions,
                                label() + " has no POSITION attribute");
                return;
            }
            const fastgltf::Accessor& posAccessor = asset.accessors[posIt->second];
            if (posAccessor.type != fastgltf::AccessorType::Vec3) {
                primitiveFailed(IngestErrorKind::AccessorTypeMismatch,
                                label() + " POSITION accessor is not VEC3");
                return;
            }

            if (!prim.indicesAccessor.has_value()) {
                primitiveFailed(IngestErrorKind::MissingIndices,
                                label() + " has no indices accessor");
                return;
            }
            const fastgltf::Accessor& idxAccessor =
                asset.accessors[*prim.indicesAccessor];
            if (idxAccessor.type != fastgltf::AccessorType::Scalar) {
                primitiveFailed(IngestErrorKind::AccessorTypeMismatch,
                                label() + " indices accessor is not SCALAR");
                return;
            }

            const std::size_t vertCount = posAccessor.count;
            const std::size_t idxCount  = idxAccessor.count;
            if (idxCount % 3u != 0u) {
                primitiveFailed(IngestErrorKind::AccessorTypeMismatch,
                                label() + " index count is not a multiple of 3");
                return;
            }

            // --- Load positions (world-space) ---
            const std::size_t vBase = out.positions.size();

            // uint32 index-range guard: the consolidated index buffer is
            // std::uint32_t, so vBase + idx must fit. Reject primitives that
            // would push us past the limit instead of silently wrapping.
            constexpr std::size_t kMaxU32Verts =
                static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max());
            if (vBase > kMaxU32Verts ||
                vertCount > kMaxU32Verts ||
                (kMaxU32Verts - vBase) < vertCount) {
                primitiveFailed(IngestErrorKind::IndexRangeExceeded,
                    label() + " would exceed uint32 index range after consolidation ("
                    + std::to_string(vBase) + " existing + "
                    + std::to_string(vertCount) + " new verts)");
                return;
            }
            out.positions.resize(vBase + vertCount);
            out.normals.resize(vBase + vertCount, glm::vec3{ 0.0f });
            out.uvs.resize(vBase + vertCount, glm::vec2{ 0.0f });

            // Precompute the normal transform (inverse-transpose of upper 3x3).
            // For non-uniformly scaled nodes this is required; for identity/
            // uniform scale it's a no-op.
            const glm::mat3 normalMat = glm::transpose(
                glm::inverse(glm::mat3{ world }));

            fastgltf::iterateAccessorWithIndex<glm::vec3>(
                asset, posAccessor,
                [&](glm::vec3 p, std::size_t i) {
                    const glm::vec4 wp = world * glm::vec4{ p, 1.0f };
                    out.positions[vBase + i] = glm::vec3{ wp };
                });

            // --- Normals (generate if absent) ---
            bool haveNormals = false;
            if (auto it = prim.findAttribute("NORMAL");
                it != prim.attributes.end()) {
                const fastgltf::Accessor& nAcc = asset.accessors[it->second];
                if (nAcc.type == fastgltf::AccessorType::Vec3 &&
                    nAcc.count == vertCount) {
                    fastgltf::iterateAccessorWithIndex<glm::vec3>(
                        asset, nAcc,
                        [&](glm::vec3 n, std::size_t i) {
                            glm::vec3 wn = normalMat * n;
                            const float len = glm::length(wn);
                            out.normals[vBase + i] =
                                (len > 1e-20f) ? (wn / len) : glm::vec3{ 0, 1, 0 };
                        });
                    haveNormals = true;
                }
            }

            // --- UVs (zero-fill if absent) ---
            if (auto it = prim.findAttribute("TEXCOORD_0");
                it != prim.attributes.end()) {
                const fastgltf::Accessor& uvAcc = asset.accessors[it->second];
                if (uvAcc.type == fastgltf::AccessorType::Vec2 &&
                    uvAcc.count == vertCount) {
                    fastgltf::iterateAccessorWithIndex<glm::vec2>(
                        asset, uvAcc,
                        [&](glm::vec2 uv, std::size_t i) {
                            out.uvs[vBase + i] = uv;
                        });
                }
            }

            // --- Indices (offset into consolidated vertex stream) ---
            const std::size_t iBase = out.indices.size();
            out.indices.resize(iBase + idxCount);
            fastgltf::iterateAccessorWithIndex<std::uint32_t>(
                asset, idxAccessor,
                [&](std::uint32_t idx, std::size_t i) {
                    out.indices[iBase + i] =
                        static_cast<std::uint32_t>(vBase) + idx;
                });

            // --- Per-triangle material index for this primitive's slice. ---
            // Default to 0 when the primitive has no material (glTF spec
            // allows this; the renderer uses material[0] as a fallback).
            const std::uint32_t matIdx = prim.materialIndex.has_value()
                ? static_cast<std::uint32_t>(*prim.materialIndex)
                : 0u;
            const std::size_t tBase = iBase / 3u;
            const std::size_t tCount = idxCount / 3u;
            out.triangleMaterialIndex.resize(tBase + tCount, matIdx);
            for (std::size_t t = 0; t < tCount; ++t) {
                out.triangleMaterialIndex[tBase + t] = matIdx;
            }

            // --- Generate face-area-weighted per-vertex smooth normals when source lacked them ---
            // Stable iteration order: triangles in index-buffer order.
            if (!haveNormals) {
                // Accumulate (cross-product length is already proportional to
                // face area, so this is a face-area-weighted smooth fallback),
                // then normalize. Good enough for M1b.1; M1b.2 may upgrade.
                for (std::size_t t = 0; t < idxCount; t += 3u) {
                    const std::uint32_t ia = out.indices[iBase + t + 0];
                    const std::uint32_t ib = out.indices[iBase + t + 1];
                    const std::uint32_t ic = out.indices[iBase + t + 2];
                    const glm::vec3& a = out.positions[ia];
                    const glm::vec3& b = out.positions[ib];
                    const glm::vec3& c = out.positions[ic];
                    const glm::vec3 n  = glm::cross(b - a, c - a);
                    out.normals[ia] += n;
                    out.normals[ib] += n;
                    out.normals[ic] += n;
                }
                for (std::size_t i = vBase; i < vBase + vertCount; ++i) {
                    const float len = glm::length(out.normals[i]);
                    out.normals[i] =
                        (len > 1e-20f) ? (out.normals[i] / len) : glm::vec3{ 0, 1, 0 };
                }
            }
        };

    // Choose a scene to walk. glTF allows `defaultScene` to be absent — in
    // that case, walk every scene in declaration order. If there are no
    // scenes at all, fall back to iterating every mesh as if placed at the
    // origin (matches the behavior of most importers).
    auto visitNode = [&](const fastgltf::Node& node, const glm::mat4& world) {
        if (visitorStatus.has_value() && node.meshIndex.has_value()) {
            const std::size_t meshIdx = *node.meshIndex;
            const auto& mesh = asset.meshes[meshIdx];
            for (std::size_t p = 0; p < mesh.primitives.size(); ++p) {
                if (!visitorStatus.has_value()) break;
                processPrimitive(meshIdx, p, mesh.primitives[p], world);
            }
        }
    };

    if (!asset.scenes.empty()) {
        const std::size_t sceneIdx = asset.defaultScene.value_or(0);
        const auto& scene =
            (sceneIdx < asset.scenes.size()) ? asset.scenes[sceneIdx]
                                             : asset.scenes.front();
        for (std::size_t nodeIdx : scene.nodeIndices) {
            if (!visitorStatus.has_value()) break;
            walkNodes(asset, nodeIdx, glm::mat4{ 1.0f }, visitNode);
        }
    } else {
        // No scene: treat each mesh as if at the origin, stable order.
        const glm::mat4 ident{ 1.0f };
        for (std::size_t meshIdx = 0; meshIdx < asset.meshes.size(); ++meshIdx) {
            if (!visitorStatus.has_value()) break;
            const auto& mesh = asset.meshes[meshIdx];
            for (std::size_t p = 0; p < mesh.primitives.size(); ++p) {
                if (!visitorStatus.has_value()) break;
                processPrimitive(meshIdx, p, mesh.primitives[p], ident);
            }
        }
    }

    if (!visitorStatus.has_value()) {
        return std::unexpected(std::move(visitorStatus.error()));
    }

    // --- Post-flight validation ---
    if (out.indices.empty() || out.positions.empty()) {
        return std::unexpected(Err{
            IngestErrorKind::ZeroTriangles,
            std::string{"consolidated mesh has no triangles"},
            gltfPath});
    }

    // Degenerate check: any triangle with a zero-area cross product is a
    // source-data defect. Count them so the error line is actionable.
    //
    // Scale-invariant: the cross-product length² (units: length⁴) is compared
    // against the longest edge² squared (also units: length⁴), so the ratio
    // is dimensionless. This works across asset scales from ZBrush (~0.01
    // units) to planetary (~1e6 units) without tuning an absolute epsilon.
    std::size_t degenerateCount = 0;
    constexpr float kDegenerateRelEps = 1e-12f;  // area²/edge⁴ threshold
    for (std::size_t t = 0; t + 2 < out.indices.size(); t += 3u) {
        const glm::vec3& a = out.positions[out.indices[t + 0]];
        const glm::vec3& b = out.positions[out.indices[t + 1]];
        const glm::vec3& c = out.positions[out.indices[t + 2]];
        const glm::vec3 e1 = b - a;
        const glm::vec3 e2 = c - a;
        const glm::vec3 nrm = glm::cross(e1, e2);
        const float crossSq   = glm::dot(nrm, nrm);     // length⁴
        const float e1Sq      = glm::dot(e1, e1);       // length²
        const float e2Sq      = glm::dot(e2, e2);       // length²
        const float maxEdgeSq = std::max(e1Sq, e2Sq);   // length²
        // Zero-length edges → definitely degenerate.
        if (maxEdgeSq <= 0.0f ||
            crossSq < kDegenerateRelEps * maxEdgeSq * maxEdgeSq) {
            ++degenerateCount;
        }
    }
    if (degenerateCount > 0) {
        std::string detail = "found ";
        detail += std::to_string(degenerateCount);
        detail += " degenerate triangle(s) (scale-invariant area² < 1e-12 * maxEdge²²)";
        return std::unexpected(Err{
            IngestErrorKind::DegenerateGeometry,
            std::move(detail),
            gltfPath});
    }

    return out;
}

} // namespace enigma::mpbake
