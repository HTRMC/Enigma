#include "renderer/MeshletBuilder.h"

#include "renderer/Meshlet.h"

#include "core/Assert.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace enigma {

// ---------------------------------------------------------------------------
// Ritter's bounding sphere (simple, fast, good-enough for meshlet culling).
// ---------------------------------------------------------------------------

void MeshletBuilder::compute_bounding_sphere(
    const float* positions, size_t stride_floats,
    const u32* local_vertices, size_t vertex_count,
    vec3& center, f32& radius)
{
    ENIGMA_ASSERT(vertex_count > 0);

    // Initial pass: find the two most distant points along x, y, z axes.
    vec3 pmin = vec3(positions[local_vertices[0] * stride_floats + 0],
                     positions[local_vertices[0] * stride_floats + 1],
                     positions[local_vertices[0] * stride_floats + 2]);
    vec3 pmax = pmin;
    u32 imin = 0, imax = 0;

    for (size_t i = 0; i < vertex_count; ++i) {
        const u32 vi = local_vertices[i];
        const vec3 p(positions[vi * stride_floats + 0],
                     positions[vi * stride_floats + 1],
                     positions[vi * stride_floats + 2]);
        if (p.x < pmin.x) pmin.x = p.x;
        if (p.y < pmin.y) pmin.y = p.y;
        if (p.z < pmin.z) pmin.z = p.z;
        if (p.x > pmax.x) pmax.x = p.x;
        if (p.y > pmax.y) pmax.y = p.y;
        if (p.z > pmax.z) pmax.z = p.z;
    }

    // Find the axis with the largest spread and pick the two extremes on it.
    const vec3 spread = pmax - pmin;
    int axis = 0;
    if (spread.y > spread.x) axis = 1;
    if (spread.z > spread[axis]) axis = 2;

    f32 lo = pmax[axis], hi = pmin[axis];
    for (size_t i = 0; i < vertex_count; ++i) {
        const u32 vi = local_vertices[i];
        const f32 v = positions[vi * stride_floats + static_cast<size_t>(axis)];
        if (v < lo) { lo = v; imin = static_cast<u32>(i); }
        if (v > hi) { hi = v; imax = static_cast<u32>(i); }
    }

    auto loadPos = [&](u32 local_idx) -> vec3 {
        const u32 vi = local_vertices[local_idx];
        return vec3(positions[vi * stride_floats + 0],
                    positions[vi * stride_floats + 1],
                    positions[vi * stride_floats + 2]);
    };

    const vec3 a = loadPos(imin);
    const vec3 b = loadPos(imax);
    center = (a + b) * 0.5f;
    radius = glm::length(b - a) * 0.5f;

    // Ritter expansion: grow sphere to include all points.
    for (size_t i = 0; i < vertex_count; ++i) {
        const vec3 p = loadPos(static_cast<u32>(i));
        const vec3 d = p - center;
        const f32 dist = glm::length(d);
        if (dist > radius) {
            const f32 new_radius = (radius + dist) * 0.5f;
            const f32 shift = new_radius - radius;
            center += (d / dist) * shift;
            radius = new_radius;
        }
    }
}

// ---------------------------------------------------------------------------
// Normal cone computation for meshlet back-face culling.
// ---------------------------------------------------------------------------

void MeshletBuilder::compute_normal_cone(
    const float* positions, size_t stride_floats,
    const u32* local_vertices, const u8* local_triangles,
    size_t triangle_count,
    vec3& cone_axis, f32& cone_cutoff)
{
    if (triangle_count == 0) {
        cone_axis = vec3(0.0f, 1.0f, 0.0f);
        cone_cutoff = -1.0f; // always passes
        return;
    }

    auto loadPos = [&](u8 local_idx) -> vec3 {
        const u32 vi = local_vertices[local_idx];
        return vec3(positions[vi * stride_floats + 0],
                    positions[vi * stride_floats + 1],
                    positions[vi * stride_floats + 2]);
    };

    // Accumulate face normals.
    vec3 normal_sum(0.0f);
    for (size_t t = 0; t < triangle_count; ++t) {
        const vec3 v0 = loadPos(local_triangles[t * 3 + 0]);
        const vec3 v1 = loadPos(local_triangles[t * 3 + 1]);
        const vec3 v2 = loadPos(local_triangles[t * 3 + 2]);
        const vec3 n = glm::cross(v1 - v0, v2 - v0);
        normal_sum += n; // weighted by triangle area (unnormalised cross product)
    }

    const f32 len = glm::length(normal_sum);
    if (len < 1e-8f) {
        cone_axis = vec3(0.0f, 1.0f, 0.0f);
        cone_cutoff = -1.0f;
        return;
    }

    cone_axis = normal_sum / len;

    // Find the maximum deviation angle from the average normal.
    f32 max_cos = 1.0f;
    for (size_t t = 0; t < triangle_count; ++t) {
        const vec3 v0 = loadPos(local_triangles[t * 3 + 0]);
        const vec3 v1 = loadPos(local_triangles[t * 3 + 1]);
        const vec3 v2 = loadPos(local_triangles[t * 3 + 2]);
        const vec3 n = glm::cross(v1 - v0, v2 - v0);
        const f32 nlen = glm::length(n);
        if (nlen < 1e-8f) {
            cone_cutoff = -1.0f;
            return;
        }
        const f32 cos_angle = glm::dot(cone_axis, n / nlen);
        if (cos_angle < max_cos) max_cos = cos_angle;
    }

    // cone_cutoff = cos(max_deviation_angle + 90°)
    // = cos(acos(max_cos) + pi/2) = -sin(acos(max_cos)) = -sqrt(1 - max_cos^2)
    const f32 sin_angle = std::sqrt(std::max(0.0f, 1.0f - max_cos * max_cos));
    cone_cutoff = -sin_angle;
}

// ---------------------------------------------------------------------------
// Greedy meshlet builder.
// ---------------------------------------------------------------------------

MeshletData MeshletBuilder::build(
    const float*    vertex_positions,
    size_t          vertex_count,
    const uint32_t* indices,
    size_t          index_count,
    size_t          max_vertices,
    size_t          max_triangles)
{
    ENIGMA_ASSERT(vertex_positions != nullptr);
    ENIGMA_ASSERT(indices != nullptr);
    ENIGMA_ASSERT(index_count % 3 == 0);
    (void)vertex_count; // bounds checking deferred to caller
    ENIGMA_ASSERT(max_vertices > 0 && max_vertices <= 255);
    ENIGMA_ASSERT(max_triangles > 0 && max_triangles <= 255);

    MeshletData result;

    const size_t triangle_count = index_count / 3;
    if (triangle_count == 0) return result;

    // Per-meshlet scratch data.
    std::unordered_map<u32, u8> vertex_map; // global vertex index -> local index
    std::vector<u32> local_verts;
    std::vector<u8>  local_tris;

    vertex_map.reserve(max_vertices * 2);
    local_verts.reserve(max_vertices);
    local_tris.reserve(max_triangles * 3);

    auto flush_meshlet = [&]() {
        if (local_tris.empty()) return;

        Meshlet m{};
        m.vertex_offset   = static_cast<u32>(result.meshlet_vertices.size());
        m.triangle_offset = static_cast<u32>(result.meshlet_triangles.size());
        m.vertex_count    = static_cast<u32>(local_verts.size());
        m.triangle_count  = static_cast<u32>(local_tris.size() / 3);

        compute_bounding_sphere(
            vertex_positions, 3,
            local_verts.data(), local_verts.size(),
            m.bounding_sphere_center, m.bounding_sphere_radius);

        compute_normal_cone(
            vertex_positions, 3,
            local_verts.data(), local_tris.data(),
            m.triangle_count,
            m.cone_axis, m.cone_cutoff);

        result.meshlets.push_back(m);
        result.meshlet_vertices.insert(result.meshlet_vertices.end(),
                                       local_verts.begin(), local_verts.end());
        result.meshlet_triangles.insert(result.meshlet_triangles.end(),
                                        local_tris.begin(), local_tris.end());

        vertex_map.clear();
        local_verts.clear();
        local_tris.clear();
    };

    for (size_t t = 0; t < triangle_count; ++t) {
        const u32 i0 = indices[t * 3 + 0];
        const u32 i1 = indices[t * 3 + 1];
        const u32 i2 = indices[t * 3 + 2];

        // Count how many new vertices this triangle would add.
        size_t new_verts = 0;
        if (vertex_map.find(i0) == vertex_map.end()) ++new_verts;
        if (vertex_map.find(i1) == vertex_map.end()) ++new_verts;
        if (vertex_map.find(i2) == vertex_map.end()) ++new_verts;

        const bool verts_fit = (local_verts.size() + new_verts) <= max_vertices;
        const bool tris_fit  = (local_tris.size() / 3 + 1) <= max_triangles;

        if (!verts_fit || !tris_fit) {
            flush_meshlet();
        }

        auto get_or_add = [&](u32 global_idx) -> u8 {
            auto it = vertex_map.find(global_idx);
            if (it != vertex_map.end()) return it->second;
            const u8 local_idx = static_cast<u8>(local_verts.size());
            vertex_map[global_idx] = local_idx;
            local_verts.push_back(global_idx);
            return local_idx;
        };

        const u8 l0 = get_or_add(i0);
        const u8 l1 = get_or_add(i1);
        const u8 l2 = get_or_add(i2);

        local_tris.push_back(l0);
        local_tris.push_back(l1);
        local_tris.push_back(l2);
    }

    flush_meshlet();

    return result;
}

} // namespace enigma
