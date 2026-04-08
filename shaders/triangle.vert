#version 460
#extension GL_EXT_nonuniform_qualifier : require

// Global bindless descriptor set declarations.
//
// Layout matches src/gfx/DescriptorAllocator.cpp exactly:
//   binding 0: SAMPLED_IMAGE   (populated — checkerboard texture)
//   binding 1: STORAGE_IMAGE   (unused at milestone 1)
//   binding 2: STORAGE_BUFFER  (populated — per-vertex pos+uv SSBO)
//   binding 3: SAMPLER         (populated — default linear/repeat)
//
// Each storage-buffer entry packs a single vertex as a vec4 where
// .xy = NDC position (math convention, +Y up — flipped on write
// below to match Vulkan's +Y-down NDC) and .zw = UV.
layout(set = 0, binding = 2) readonly buffer Vertices {
    vec4 data[];
} g_storageBuffers[];

layout(push_constant) uniform PC {
    uint bufferIndex;
    uint textureIndex;
    uint samplerIndex;
    uint _pad;
} pc;

layout(location = 0) out vec2 v_uv;

void main() {
    // Fetch vertex from the bindless SSBO. `nonuniformEXT` is
    // technically over-cautious here (the push constant is uniform
    // across the draw) but it is the authentic idiom and proves the
    // non-uniform-indexing path is live.
    const vec4 vertex = g_storageBuffers[nonuniformEXT(pc.bufferIndex)].data[gl_VertexIndex];

    // Vulkan NDC has +Y pointing DOWN the screen. Flip Y on write so
    // the engine stores math-convention coordinates and the screen
    // sees the right-side-up triangle.
    gl_Position = vec4(vertex.x, -vertex.y, 0.0, 1.0);

    // UVs pass through untouched — texture() sampling in the
    // fragment shader handles address mode and filtering.
    v_uv = vertex.zw;
}
