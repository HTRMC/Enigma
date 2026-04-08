// triangle.hlsl
// ================
// Bindless triangle shader (HLSL, one file per pass) — replaces the
// prior `triangle.vert` / `triangle.frag` split. Both `VSMain` and
// `PSMain` live here and share the binding declarations and push
// constant block so the layout can only be declared once.
//
// Layout matches src/gfx/DescriptorAllocator.cpp exactly:
//   binding 0: SAMPLED_IMAGE   (populated — checkerboard texture)
//   binding 1: STORAGE_IMAGE   (unused at this milestone)
//   binding 2: STORAGE_BUFFER  (populated — per-vertex pos+uv SSBO)
//   binding 3: SAMPLER         (populated — linear/repeat)
//
// `[[vk::binding(N, 0)]]` is the authoritative Vulkan binding. The
// companion `register(tN, spaceN)` is a D3D register declaration
// that HLSL requires syntactically but DXC ignores when emitting
// SPIR-V — we pick distinct D3D slots per resource only to keep
// HLSL happy, not because Vulkan uses them.
//
// `NonUniformResourceIndex` is technically over-cautious here (the
// push constants are uniform across the draw) but it's the authentic
// bindless idiom and keeps the non-uniform-indexing path live on the
// SPIR-V backend, matching what the GLSL version was doing with
// `nonuniformEXT`.

[[vk::binding(2, 0)]]
StructuredBuffer<float4> g_vertices[] : register(t0, space1);

[[vk::binding(0, 0)]]
Texture2D g_sampledImages[] : register(t0, space0);

[[vk::binding(3, 0)]]
SamplerState g_samplers[] : register(s0, space0);

// DXC requires `[[vk::push_constant]]` to annotate a global variable
// of struct type — not a cbuffer block. The `pc.X` reference form
// mirrors the GLSL `layout(push_constant) uniform PC { ... } pc;`
// idiom we were using before the migration.
struct PushBlock {
    uint bufferIndex;
    uint textureIndex;
    uint samplerIndex;
    uint _pad;
};

[[vk::push_constant]] PushBlock pc;

struct VSOut {
    float4 pos : SV_Position;
    float2 uv  : TEXCOORD0;
};

VSOut VSMain(uint vid : SV_VertexID) {
    // Each SSBO entry packs a single vertex as a vec4 where
    // .xy = NDC position (math convention, +Y up — flipped on write
    // below to match Vulkan's +Y-down NDC) and .zw = UV coordinates.
    const float4 vertex = g_vertices[NonUniformResourceIndex(pc.bufferIndex)][vid];

    VSOut o;
    o.pos = float4(vertex.x, -vertex.y, 0.0, 1.0);
    o.uv  = vertex.zw;
    return o;
}

float4 PSMain(VSOut vs) : SV_Target {
    // Bindless sampled texture + sampler via the separate-arrays
    // idiom: pick a Texture2D slot and a SamplerState slot by index,
    // then call .Sample(). This maps to the same SPIR-V as the GLSL
    // `texture(sampler2D(tex, samp), uv)` construction.
    Texture2D    tex  = g_sampledImages[NonUniformResourceIndex(pc.textureIndex)];
    SamplerState samp = g_samplers[NonUniformResourceIndex(pc.samplerIndex)];
    return tex.Sample(samp, vs.uv);
}
