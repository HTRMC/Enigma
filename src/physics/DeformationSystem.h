#pragma once

#include "physics/CrumpleZone.h"
#include "core/Types.h"
#include "core/Math.h"

#include <volk.h>

#include <vector>
#include <functional>

struct VmaAllocation_T;
using VmaAllocation = VmaAllocation_T*;

namespace enigma::gfx {
    class Device;
    class Allocator;
    class BLAS;
}

namespace enigma {

struct MeshPrimitive;

// Impact event: position + direction + force magnitude.
struct ImpactEvent {
    vec3 worldPosition;   // impact point in world space
    vec3 direction;       // direction of force (normalised)
    f32  force;           // N — mapped to displacement via material hardness
};

class DeformationSystem {
public:
    DeformationSystem() = default;
    ~DeformationSystem() = default;

    // Register a primitive for deformation tracking.
    // `originalPositions` is a CPU-side copy of the vertex positions at load time.
    // `zone` describes per-vertex deformability.
    void registerPrimitive(u32 primitiveIndex,
                           std::vector<vec3> originalPositions,
                           CrumpleZone zone);

    // Apply an impact to all registered primitives within `radius` metres of
    // `event.worldPosition`. Displaces vertices according to zone weights and force.
    // Returns the maximum vertex displacement caused (metres).
    f32 applyImpact(const ImpactEvent& event, f32 radius = 1.5f);

    // Upload deformed vertex positions to the GPU vertex buffer for `primitiveIndex`.
    // Call after applyImpact() if maxDisplacement > 0.
    // Only updates the position component of each vertex (leaves normal/uv/tangent intact).
    void uploadDeformedPositions(u32 primitiveIndex,
                                 VkBuffer vertexBuffer,
                                 gfx::Device& device);

    // Returns true if the deformation is large enough to require a BLAS rebuild
    // (> 30% of vertices displaced > maxDisplacement * 0.3).
    // Returns false for minor deformation -> use BLAS refit instead.
    bool requiresBlasRebuild(u32 primitiveIndex) const;

    // Reset all deformation for a primitive (e.g., vehicle respawn).
    void reset(u32 primitiveIndex);

    // Current deformed positions (CPU side) for a primitive.
    const std::vector<vec3>& deformedPositions(u32 primitiveIndex) const;

private:
    struct PrimitiveState {
        u32               index;
        std::vector<vec3> originalPositions;
        std::vector<vec3> deformedPositions;
        CrumpleZone       zone;
        u32               largeDisplacementCount = 0; // for rebuild decision
    };

    std::vector<PrimitiveState> m_primitives;
    PrimitiveState* findPrimitive(u32 index);
    const PrimitiveState* findPrimitive(u32 index) const;
};

} // namespace enigma
