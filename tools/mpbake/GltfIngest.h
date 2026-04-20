#pragma once

// GltfIngest.h
// =============
// Stage 1 of the offline `enigma-mpbake` pipeline: parse an input glTF/GLB
// asset, walk the node hierarchy, collapse world-space transforms, and emit a
// flat `IngestedMesh` POD for the clustering stage (M1b.2) to consume.
//
// Contract
// --------
// `load()` is stateless and deterministic. Success produces an `IngestedMesh`
// with:
//   - positions, normals, uvs all `.size()` equal to `vertCount`
//   - indices.size() % 3 == 0 and every index < vertCount
// Failure returns a typed `IngestError` with the offending mesh/primitive
// name in `.detail` so baker invocations surface actionable error messages.
//
// M1 scope
// --------
// Only static triangle meshes are supported:
//   - Primitive mode MUST be TRIANGLES (TRIANGLE_STRIP/FAN rejected).
//   - Meshes referenced by any node with a `skinIndex` are rejected.
//   - POSITION and an indices accessor are required.
//   - NORMAL is generated as a face-area-weighted per-vertex smooth fallback
//     when absent; TEXCOORD_0 is zero-filled. The clustering stage may
//     revisit normal generation once M1b.2 lands.
// Morph targets, animations, lights, cameras, and material data are ignored —
// the bake pipeline consumes geometry only.

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

// std::expected is a C++23 library feature — available under /std:c++latest.
#include <expected>

namespace enigma::mpbake {

// POD holding the result of the glTF parse. World-space static geometry
// consolidated across every node instance into a single vertex stream.
// Index buffer is always uint32 for determinism, irrespective of the
// source encoding.
struct IngestedMesh {
    std::vector<glm::vec3>     positions;
    std::vector<glm::vec3>     normals;    // face-area-weighted per-vertex smooth fallback if absent from source
    std::vector<glm::vec2>     uvs;        // zero-filled if TEXCOORD_0 absent
    std::vector<std::uint32_t> indices;    // triangle list; size is a multiple of 3
    // Per-triangle source material index. `size()` equals `indices.size() / 3`.
    // Primitives with no material index resolve to 0 (the scene's first material).
    // Consumed by ClusterBuilder to stamp per-cluster materialIndex.
    std::vector<std::uint32_t> triangleMaterialIndex;
};

// Classifier for every failure `GltfIngest::load` may surface. Callers print
// the `errorKindString(kind)` alongside `detail` for a human-readable line.
enum class IngestErrorKind {
    FileNotFound,
    FileReadFailed,             // path exists but contents couldn't be read
    GltfParseFailed,
    NoMeshes,
    SkinnedMeshUnsupported,
    UnsupportedPrimitiveMode,   // only TRIANGLES supported in M1
    MissingPositions,
    MissingIndices,
    ZeroTriangles,
    DegenerateGeometry,
    AccessorTypeMismatch,
    IndexRangeExceeded,         // consolidated vertex count would overflow uint32
};

// Rich error payload — `detail` names the offending mesh/primitive and
// `path` echoes the input so log lines are self-contained.
struct IngestError {
    IngestErrorKind         kind;
    std::string             detail;
    std::filesystem::path   path;
};

// Stable string for `IngestErrorKind`. Guaranteed never null.
const char* errorKindString(IngestErrorKind kind) noexcept;

// GltfIngest
// ----------
// Stateless loader. Construct + `load()`; success returns a populated
// `IngestedMesh`, failure returns a typed `IngestError`.
class GltfIngest {
public:
    GltfIngest()  = default;
    ~GltfIngest() = default;

    // Parse `gltfPath` (.gltf or .glb), walk all scenes + nodes, and emit
    // a world-space `IngestedMesh`. See header preamble for the contract.
    std::expected<IngestedMesh, IngestError>
    load(const std::filesystem::path& gltfPath);
};

} // namespace enigma::mpbake
