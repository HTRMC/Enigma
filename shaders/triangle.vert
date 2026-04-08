#version 460
#extension GL_EXT_nonuniform_qualifier : require

// Global bindless descriptor set declarations.
//
// Layout matches src/gfx/DescriptorAllocator.cpp exactly:
//   binding 0: SAMPLED_IMAGE   (unused at milestone 1)
//   binding 1: STORAGE_IMAGE   (unused at milestone 1)
//   binding 2: STORAGE_BUFFER  (populated — vertex positions live here)
//   binding 3: SAMPLER         (unused at milestone 1)
//
// Only binding 2 is declared here because a vertex shader cannot
// meaningfully read from images without samplers; the other bindings
// will be declared as needed by the shaders that consume them.
layout(set = 0, binding = 2) readonly buffer Positions {
    vec4 pos[];
} g_storageBuffers[];

layout(push_constant) uniform PC {
    uint bufferIndex;
    uint _pad[3];
} pc;

layout(location = 0) out vec3 v_color;

void main() {
    // Fetch vertex position from the bindless SSBO. `nonuniformEXT` is
    // technically over-cautious here (the push constant is uniform
    // across the draw) but it is the authentic idiom and proves the
    // non-uniform-indexing path is live.
    const vec4 p = g_storageBuffers[nonuniformEXT(pc.bufferIndex)].pos[gl_VertexIndex];
    gl_Position = vec4(p.xy, 0.0, 1.0);

    // Per-vertex color derived from the vertex index so the three
    // corners come out red / green / blue.
    const vec3 palette[3] = vec3[3](
        vec3(1.0, 0.0, 0.0),
        vec3(0.0, 1.0, 0.0),
        vec3(0.0, 0.0, 1.0)
    );
    v_color = palette[gl_VertexIndex % 3];
}
