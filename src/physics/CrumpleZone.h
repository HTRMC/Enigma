#pragma once

#include "core/Types.h"
#include "core/Math.h"

#include <string>
#include <string_view>
#include <vector>

namespace enigma {

// Per-vertex deformation weight and displacement limits.
// Loaded from a sidecar JSON or hardcoded per vehicle zone.
struct CrumpleZoneVertex {
    f32  weight;           // 0 = rigid, 1 = fully deformable
    f32  maxDisplacement;  // metres — clamped vertex travel
    f32  hardness;         // resistance to deformation (higher = less displacement per unit force)
};

// Named deformation zone on a mesh (e.g. "front_hood", "door_left").
struct CrumpleZone {
    std::string              name;
    std::vector<CrumpleZoneVertex> vertices; // parallel to mesh vertex buffer
    f32  currentDamage = 0.0f; // 0–1, accumulates over impacts

    // Build a default crumple map for a mesh with `vertexCount` vertices.
    // Zone name drives defaults: "front*" -> high deformability at front face vertices,
    // "rear*" -> high deformability at rear, "door*" -> side deformability.
    // Default (no name match): uniform low deformability.
    static CrumpleZone makeDefault(std::string_view name, u32 vertexCount);
};

} // namespace enigma
