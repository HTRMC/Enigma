// debug_wireframe.frag.hlsl — Fragment shader for VK_POLYGON_MODE_LINE wireframe

struct WireBlock {
    [[vk::offset(32)]] float3 wireColor;
    float _pad;
};
[[vk::push_constant]] WireBlock pc;

// VertexOutput must match visibility_buffer.mesh.hlsl
struct VertexOutput {
    float4 pos : SV_Position;
    nointerpolation uint vis_value : TEXCOORD0;
};

float4 PSMain(VertexOutput input) : SV_Target {
    return float4(pc.wireColor, 1.0);
}
