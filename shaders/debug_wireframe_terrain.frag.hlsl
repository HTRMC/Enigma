// debug_wireframe_terrain.frag.hlsl
// ===================================
// Wireframe fragment shader for terrain mesh shader pipeline.
// Push constant layout: VBTerrainPushBlock (40 bytes) at task+mesh stages,
// then wireColor (16 bytes) at offset 40 for the fragment stage.

struct WireBlock {
    [[vk::offset(40)]] float3 wireColor;
    float _pad;
};
[[vk::push_constant]] WireBlock pc;

float4 PSMain() : SV_Target {
    return float4(pc.wireColor, 1.0);
}
