// debug_wireframe_terrain.frag.hlsl
// ===================================
// Wireframe fragment shader for terrain mesh shader pipeline.
// Push constant layout: VBTerrainPushBlock (40 bytes) at task+mesh stages,
// then wireColor (16 bytes) at offset 48 for the fragment stage.
// Offset 48 (not 40) keeps the float4 on a 16-byte boundary — SPIR-V scalar
// block layout forbids vec types straddling a 16-byte boundary.

struct WireBlock {
    [[vk::offset(48)]] float4 wireColor;
};
[[vk::push_constant]] WireBlock pc;

float4 PSMain() : SV_Target {
    return float4(pc.wireColor.rgb, 1.0);
}
