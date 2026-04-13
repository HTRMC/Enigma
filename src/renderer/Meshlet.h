#pragma once

#include "core/Math.h"
#include "core/Types.h"

#include <vector>

namespace enigma {

// Per-meshlet descriptor. Matches the GPU-side struct in visibility_buffer.mesh.hlsl.
// NV recommendations: max 64 vertices, max 124 triangles per meshlet.
struct Meshlet {
    u32 vertex_offset;    // offset into the meshlet's vertex-index array
    u32 triangle_offset;  // offset into the packed triangle array
    u32 vertex_count;     // number of unique vertices in this meshlet (≤ 64)
    u32 triangle_count;   // number of triangles in this meshlet (≤ 124)

    // Bounding sphere for meshlet-level frustum culling (world space after transform).
    // Center is relative to mesh origin; radius is the tightest enclosing sphere.
    vec3 bounding_sphere_center;
    f32  bounding_sphere_radius;

    // Normal cone for back-face culling (in mesh-local space).
    // axis is the average normal direction, angle is the half-angle of the cone.
    vec3 cone_axis;
    f32  cone_cutoff; // cos(half_angle + 90°) — negative means cone test always passes
};

// CPU-side meshlet build result for one mesh.
struct MeshletData {
    std::vector<Meshlet> meshlets;
    std::vector<u32>     meshlet_vertices;   // remapped vertex indices (into original vertex buffer)
    std::vector<u8>      meshlet_triangles;  // packed: 3 u8 indices per triangle (local to meshlet)
};

} // namespace enigma
