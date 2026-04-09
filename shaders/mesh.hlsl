// mesh.hlsl
// =========
// Bindless mesh shader for Enigma milestone 2. Renders glTF meshes with
// camera transforms and base-color texturing. One file per pass — both
// VSMain and PSMain live here.
//
// Vertex data is packed as StructuredBuffer<float4>, 3 entries per vertex:
//   [vid*3+0] = (position.x, position.y, position.z, normal.x)
//   [vid*3+1] = (normal.y, normal.z, uv.x, uv.y)
//   [vid*3+2] = (tangent.x, tangent.y, tangent.z, tangent.w)
//
// Camera data is a single CameraData struct accessed via
// StructuredBuffer<float4> at the camera SSBO slot (13 float4s).

#include "common.hlsl"

// --- Bindless resource arrays (same layout as triangle.hlsl) ----------
[[vk::binding(2, 0)]]
StructuredBuffer<float4> g_buffers[] : register(t0, space1);

[[vk::binding(0, 0)]]
Texture2D g_textures[] : register(t0, space0);

[[vk::binding(3, 0)]]
SamplerState g_samplers[] : register(s0, space0);

// --- Push constants (80 bytes) ----------------------------------------
struct PushBlock {
    float4x4 model;            // 64 bytes
    float4   baseColorFactor;  // 16 bytes
    uint     vertexSlot;       //  4 bytes
    uint     cameraSlot;       //  4 bytes
    uint     textureSlot;      //  4 bytes
    uint     samplerSlot;      //  4 bytes
};                             // Total: 96 bytes

[[vk::push_constant]] PushBlock pc;

// --- Helper: load CameraData from flat float4 buffer ------------------
// glm stores matrices column-major: buf[0..3] are the 4 columns.
// HLSL's float4x4(a,b,c,d) constructor always fills rows, so loading
// columns as rows gives the transpose. Transpose to correct.
CameraData loadCamera(uint slot) {
    StructuredBuffer<float4> buf = g_buffers[NonUniformResourceIndex(slot)];
    CameraData cam;
    cam.view     = transpose(float4x4(buf[0],  buf[1],  buf[2],  buf[3]));
    cam.proj     = transpose(float4x4(buf[4],  buf[5],  buf[6],  buf[7]));
    cam.viewProj = transpose(float4x4(buf[8],  buf[9],  buf[10], buf[11]));
    cam.worldPos = buf[12];
    return cam;
}

// --- Vertex shader ----------------------------------------------------
struct VSOut {
    float4 pos      : SV_Position;
    float3 worldPos : TEXCOORD0;
    float3 normal   : TEXCOORD1;
    float2 uv       : TEXCOORD2;
};

VSOut VSMain(uint vid : SV_VertexID) {
    // Unpack vertex from float4 triplet.
    StructuredBuffer<float4> vbuf = g_buffers[NonUniformResourceIndex(pc.vertexSlot)];
    float4 d0 = vbuf[vid * 3 + 0];
    float4 d1 = vbuf[vid * 3 + 1];
    float4 d2 = vbuf[vid * 3 + 2];

    float3 position = d0.xyz;
    float3 normal   = float3(d0.w, d1.xy);
    float2 uv       = d1.zw;
    // tangent (d2) available for normal mapping in future milestones

    // Transform.
    CameraData cam = loadCamera(pc.cameraSlot);
    float4 worldPos = mul(pc.model, float4(position, 1.0));

    VSOut o;
    o.pos      = mul(cam.viewProj, worldPos);
    o.worldPos = worldPos.xyz;
    o.normal   = normalize(mul((float3x3)pc.model, normal));
    o.uv       = uv;
    return o;
}

// --- Fragment shader --------------------------------------------------
float4 PSMain(VSOut vs) : SV_Target {
    // Sample base color texture and multiply by material factor.
    Texture2D    tex  = g_textures[NonUniformResourceIndex(pc.textureSlot)];
    SamplerState samp = g_samplers[NonUniformResourceIndex(pc.samplerSlot)];
    float4 baseColor  = tex.Sample(samp, vs.uv) * pc.baseColorFactor;

    // Simple directional light (sun-like, from upper-right-front).
    float3 lightDir = normalize(float3(0.5, 1.0, 0.3));
    float3 N = normalize(vs.normal);
    float NdotL = max(dot(N, lightDir), 0.0);

    // Ambient + diffuse.
    float3 ambient = 0.15;
    float3 diffuse = baseColor.rgb * (ambient + NdotL * 0.85);

    return float4(diffuse, baseColor.a);
}
