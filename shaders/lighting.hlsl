// lighting.hlsl
// =============
// Deferred lighting pass. Fullscreen triangle reads the G-buffer textures
// and evaluates the same Cook-Torrance BRDF as mesh.hlsl to produce a
// shaded result on the swapchain color attachment.
//
// G-buffer layout expected:
//   albedoSlot    R8G8B8A8_UNORM     rgb=baseColor, a=occlusion
//   normalSlot    A2B10G10R10_UNORM  rgb=world normal packed to [0,1]
//   metalRoughSlot R8G8_UNORM        r=metallic, g=roughness (perceptual)
//   depthSlot     D32_SFLOAT (sampled as float) depth for world-pos reconstruction
//
// All textures are sampled with a nearest-neighbour sampler (samplerSlot)
// to avoid bilinear bleeding across geometry edges.

#include "common.hlsl"

// --- Bindless resource arrays ---
[[vk::binding(2, 0)]]
StructuredBuffer<float4> g_buffers[] : register(t0, space1);

[[vk::binding(0, 0)]]
Texture2D g_textures[] : register(t0, space0);

[[vk::binding(3, 0)]]
SamplerState g_samplers[] : register(s0, space0);

// --- Push constants (64 bytes) ---
struct PushBlock {
    uint  albedoSlot;
    uint  normalSlot;
    uint  metalRoughSlot;
    uint  depthSlot;
    uint  cameraSlot;
    uint  samplerSlot;
    uint  _pad0;
    uint  _pad1;
    float4 lightDirIntensity; // xyz = direction (pre-normalised), w = intensity
    float4 lightColor;         // xyz = colour, w = unused
};

[[vk::push_constant]] PushBlock pc;

// --- Constants ---
static const float PI = 3.14159265359;

// --- Camera load ---
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

// --- PBR functions (identical to mesh.hlsl) ---

float D_GGX(float NdotH, float alpha) {
    float a2 = alpha * alpha;
    float d  = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-7);
}

float V_SmithGGXCorrelated(float NdotV, float NdotL, float alpha) {
    float a2   = alpha * alpha;
    float GGXV = NdotL * sqrt(max(NdotV * NdotV * (1.0 - a2) + a2, 1e-7));
    float GGXL = NdotV * sqrt(max(NdotL * NdotL * (1.0 - a2) + a2, 1e-7));
    return 0.5 / max(GGXV + GGXL, 1e-7);
}

float3 F_Schlick(float VdotH, float3 F0) {
    return F0 + (1.0 - F0) * pow(saturate(1.0 - VdotH), 5.0);
}

float3 ACES(float3 x) {
    return saturate((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14));
}

// --- Vertex shader: fullscreen triangle ---
struct VSOut {
    float4 pos      : SV_Position;
    float2 texCoord : TEXCOORD0; // [0,1]^2, origin top-left
};

VSOut VSMain(uint vid : SV_VertexID) {
    // Three vertices that form a triangle covering the entire NDC clip space:
    //   vid=0 → (-1,-1)   vid=1 → (3,-1)   vid=2 → (-1,3)
    float2 uv = float2((vid << 1) & 2, vid & 2);
    VSOut o;
    o.pos      = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    o.texCoord = uv; // already in [0,1] for the visible quad
    return o;
}

// --- Pixel shader ---
float4 PSMain(VSOut vs) : SV_Target {
    float2 uv = vs.texCoord;
    SamplerState samp = g_samplers[NonUniformResourceIndex(pc.samplerSlot)];

    // Sample G-buffer.
    float4 albedoSample     = g_textures[NonUniformResourceIndex(pc.albedoSlot    )].Sample(samp, uv);
    float4 normalSample     = g_textures[NonUniformResourceIndex(pc.normalSlot    )].Sample(samp, uv);
    float2 metalRoughSample = g_textures[NonUniformResourceIndex(pc.metalRoughSlot)].Sample(samp, uv).rg;
    float  depth            = g_textures[NonUniformResourceIndex(pc.depthSlot     )].Sample(samp, uv).r;

    // Reverse-Z: far plane = 0. Skip background pixels.
    if (depth == 0.0) {
        return float4(0.02, 0.02, 0.05, 1.0); // match clear colour
    }

    // Reconstruct world position from depth + NDC.
    // uv (0,0)=top-left → NDC (-1,-1); uv (1,1)=bottom-right → NDC (1,1).
    float2 ndc     = uv * 2.0 - 1.0;
    float4 ndcPos  = float4(ndc.x, ndc.y, depth, 1.0);
    CameraData cam = loadCamera(pc.cameraSlot);
    float4 worldPos4 = mul(cam.invViewProj, ndcPos);
    float3 worldPos  = worldPos4.xyz / worldPos4.w;

    // Decode G-buffer.
    float3 baseColor  = albedoSample.rgb;
    float  occlusion  = albedoSample.a;
    float3 N          = normalize(normalSample.rgb * 2.0 - 1.0); // decode [0,1] → [-1,1]
    float  metallic   = metalRoughSample.r;
    float  roughness  = metalRoughSample.g;
    float  alpha      = roughness * roughness; // perceptual → linear

    // Directional sun light.
    float3 lightDir   = normalize(pc.lightDirIntensity.xyz);
    float3 lightColor = pc.lightColor.xyz * pc.lightDirIntensity.w;

    float3 V = normalize(cam.worldPos.xyz - worldPos);
    float3 L = lightDir;
    float3 H = normalize(V + L);

    float NdotL = saturate(dot(N, L));
    float NdotV = saturate(dot(N, V)) + 1e-5;
    float NdotH = saturate(dot(N, H));
    float VdotH = saturate(dot(V, H));

    // Cook-Torrance specular.
    float3 F0       = lerp(float3(0.04, 0.04, 0.04), baseColor, metallic);
    float  D        = D_GGX(NdotH, alpha);
    float  Vis      = V_SmithGGXCorrelated(NdotV, NdotL, alpha);
    float3 F        = F_Schlick(VdotH, F0);
    float3 specular = D * Vis * F;

    // Lambertian diffuse.
    float3 kD      = (1.0 - F) * (1.0 - metallic);
    float3 diffuse = kD * baseColor / PI;

    float3 Lo = (diffuse + specular) * lightColor * NdotL;

    // Ambient (IBL placeholder — constant).
    float3 ambient = 0.03 * baseColor * occlusion;

    float3 color = ambient + Lo;

    return float4(color, 1.0);
}
