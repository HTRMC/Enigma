// Terrain wireframe fragment shader — outputs a flat wire color.
// Push block layout (32 bytes total):
//   bytes  0-15: TerrainPushBlock fields (unused in FS — owned by VSMain in terrain_clipmap.hlsl)
//   bytes 16-31: wireColor (float3) + pad (float) — read here via [[vk::offset(16)]]
#include "common.hlsl"

struct TerrainWirePC {
    [[vk::offset(16)]] float3 wireColor;
    float _pad;
};
[[vk::push_constant]] TerrainWirePC pc;

float4 PSMain() : SV_Target {
    return float4(pc.wireColor, 1.0f);
}
