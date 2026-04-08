#version 460
#extension GL_EXT_nonuniform_qualifier : require

// Bindless sampled images and samplers. The Vulkan + descriptor
// indexing idiom declares the two resource classes as separate
// arrays and combines them at use-time via `sampler2D(texture,
// sampler)`. The push constant carries one index into each.
layout(set = 0, binding = 0) uniform texture2D g_sampledImages[];
layout(set = 0, binding = 3) uniform sampler   g_samplers[];

layout(push_constant) uniform PC {
    uint bufferIndex;
    uint textureIndex;
    uint samplerIndex;
    uint _pad;
} pc;

layout(location = 0) in  vec2 v_uv;
layout(location = 0) out vec4 o_color;

void main() {
    const sampler2D combined = sampler2D(
        g_sampledImages[nonuniformEXT(pc.textureIndex)],
        g_samplers[nonuniformEXT(pc.samplerIndex)]);
    o_color = texture(combined, v_uv);
}
