#pragma once

#include "core/Math.h"
#include "core/Types.h"

#include <cstddef>

namespace enigma {

struct MeshletData;

class MeshletBuilder {
public:
    // Build meshlets from an indexed triangle mesh.
    // vertices: positions (stride = 3 floats = 12 bytes)
    // indices: triangle list
    // Returns MeshletData with meshlets, remapped vertices, and packed triangles.
    static MeshletData build(
        const float*    vertex_positions,  // xyz per vertex
        size_t          vertex_count,
        const uint32_t* indices,
        size_t          index_count,
        size_t          max_vertices   = 64,
        size_t          max_triangles  = 124
    );

private:
    static void compute_bounding_sphere(
        const float* positions, size_t stride_floats,
        const u32* local_vertices, size_t vertex_count,
        vec3& center, f32& radius);

    static void compute_normal_cone(
        const float* positions, size_t stride_floats,
        const u32* local_vertices, const u8* local_triangles,
        size_t triangle_count,
        vec3& cone_axis, f32& cone_cutoff);
};

} // namespace enigma
