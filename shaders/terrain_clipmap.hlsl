// terrain_clipmap.hlsl
// ====================
// GPU-driven clipmap LOD terrain. One vkCmdDraw call per frame; all
// geometry is synthesised in the vertex shader from SV_VertexID and
// SV_InstanceID. No vertex buffers bound, no UVs stored, no XZ
// positions stored.
//
// Output targets (must match G-buffer layout in GBufferFormats.h):
//   SV_Target0  albedo      R8G8B8A8_UNORM
//   SV_Target1  normal      A2B10G10R10_UNORM_PACK32
//   SV_Target2  metalRough  R8G8_UNORM
//   SV_Target3  motionVec   R16G16_SFLOAT
//
// Depth writes to D32_SFLOAT (reverse-Z, far = 0).

#include "common.hlsl"

// --- Bindless resource arrays (binding 2 = storage buffers) ---
[[vk::binding(2, 0)]]
StructuredBuffer<float4> g_buffers[] : register(t0, space1);

// --- Push constants (16 bytes) ---
struct TerrainPC {
    uint  chunkSSBOSlot; // bindless slot for the TerrainChunkDesc array
    uint  cameraSlot;    // bindless slot for the camera SSBO
    uint  quadsPerSide;  // N (32)
    float pad;
};
[[vk::push_constant]] TerrainPC pc;

CameraData loadCamera(uint slot) {
    StructuredBuffer<float4> buf = g_buffers[NonUniformResourceIndex(slot)];
    CameraData cam;
    cam.view         = transpose(float4x4(buf[0],  buf[1],  buf[2],  buf[3]));
    cam.proj         = transpose(float4x4(buf[4],  buf[5],  buf[6],  buf[7]));
    cam.viewProj     = transpose(float4x4(buf[8],  buf[9],  buf[10], buf[11]));
    cam.prevViewProj = transpose(float4x4(buf[12], buf[13], buf[14], buf[15]));
    cam.invViewProj  = transpose(float4x4(buf[16], buf[17], buf[18], buf[19]));
    cam.worldPos     = buf[20];
    return cam;
}

// Procedural height — must match the C++ terrainHeight() in Application.cpp exactly.
float terrainHeight(float wx, float wz) {
    return sin(wx * 0.05) * cos(wz * 0.05) * 2.0
         + sin(wx * 0.13 + 1.1) * sin(wz * 0.09) * 0.8;
}

struct VSOut {
    float4 pos         : SV_Position;
    float3 worldPos    : TEXCOORD0;
    float3 normal      : TEXCOORD1;
    float2 uv          : TEXCOORD2;
    float4 clipPos     : TEXCOORD3;
    float4 prevClipPos : TEXCOORD4;
};

VSOut VSMain(uint vid : SV_VertexID, uint iid : SV_InstanceID) {
    const uint N = pc.quadsPerSide;

    // Map a flat vertex id to a quad + corner within that quad.
    uint quadIdx   = vid / 6;
    uint localVert = vid % 6;
    uint col       = quadIdx % N;
    uint row       = quadIdx / N;

    // Six-vertex quad layout:
    //   0: (col,   row)      dc=0 dr=0
    //   1: (col,   row+1)    dc=0 dr=1
    //   2: (col+1, row)      dc=1 dr=0
    //   3: (col+1, row)      dc=1 dr=0
    //   4: (col,   row+1)    dc=0 dr=1
    //   5: (col+1, row+1)    dc=1 dr=1
    uint dc = (localVert == 2 || localVert == 3 || localVert == 5) ? 1u : 0u;
    uint dr = (localVert == 1 || localVert == 4 || localVert == 5) ? 1u : 0u;

    // Read this chunk's descriptor from the SSBO (one float4 per chunk).
    float4 chunkDesc = g_buffers[NonUniformResourceIndex(pc.chunkSSBOSlot)][iid];
    float2 worldOffset = chunkDesc.xy;
    float  scale       = chunkDesc.z;
    float  sink        = chunkDesc.w;

    float chunkSize = 64.0 * scale;
    float quadSize  = chunkSize / float(N);

    float localX = float(col + dc) * quadSize;
    float localZ = float(row + dr) * quadSize;

    float wx = worldOffset.x + localX;
    float wz = worldOffset.y + localZ;
    float height = terrainHeight(wx, wz) + sink;

    float3 worldPos = float3(wx, height, wz);

    CameraData cam = loadCamera(pc.cameraSlot);
    float4 clipPos     = mul(cam.viewProj,     float4(worldPos, 1.0));
    float4 prevClipPos = mul(cam.prevViewProj, float4(worldPos, 1.0));

    VSOut o;
    o.pos         = clipPos;
    o.worldPos    = worldPos;
    o.normal      = float3(0.0, 1.0, 0.0); // flat normal — full normal compression deferred
    o.uv          = float2(wx, wz) / chunkSize;
    o.clipPos     = clipPos;
    o.prevClipPos = prevClipPos;
    return o;
}

// --- G-buffer output (matches GBufferFormats.h) ---
struct GBufferOut {
    float4 albedo     : SV_Target0; // rgb=baseColor, a=occlusion
    float4 normal     : SV_Target1; // rgb=world normal packed to [0,1]
    float2 metalRough : SV_Target2; // r=metallic, g=roughness (perceptual)
    float2 motionVec  : SV_Target3; // rg=NDC-space velocity
};

GBufferOut PSMain(VSOut vs) {
    // Asphalt-like material parameters.
    float3 baseColor = float3(0.10, 0.10, 0.10);
    const float  metallic  = 0.0;
    const float  roughness = 0.85;
    const float  occlusion = 1.0;

    // Grid lines every 4 m for depth perception.
    // frac(worldPos/gridSpacing) → 0 at line, 1 at midpoint.
    const float gridSpacing = 4.0;
    float2 gridUV = frac(float2(vs.worldPos.x, vs.worldPos.z) / gridSpacing);
    float2 gridDeriv = fwidth(float2(vs.worldPos.x, vs.worldPos.z) / gridSpacing);
    float2 gridLine = smoothstep(float2(0.0, 0.0), gridDeriv * 1.5, gridUV)
                    * smoothstep(float2(0.0, 0.0), gridDeriv * 1.5, 1.0 - gridUV);
    float grid = 1.0 - min(gridLine.x, gridLine.y); // 1 = on line, 0 = off
    baseColor = lerp(baseColor, float3(0.22, 0.22, 0.22), grid * 0.6);

    float3 N = normalize(vs.normal);

    // Motion vector from current/previous clip-space positions.
    float2 currentNDC = vs.clipPos.xy    / vs.clipPos.w;
    float2 prevNDC    = vs.prevClipPos.xy / vs.prevClipPos.w;
    float2 motion     = currentNDC - prevNDC;

    GBufferOut o;
    o.albedo     = float4(baseColor, occlusion);
    o.normal     = float4(N * 0.5 + 0.5, 0.0);
    o.metalRough = float2(metallic, roughness);
    o.motionVec  = motion;
    return o;
}
