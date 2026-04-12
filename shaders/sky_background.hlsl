// sky_background.hlsl
// ===================
// Fullscreen sky background pass. Renders the Hillaire 2020 SkyView LUT into
// pixels that have no geometry (reverse-Z depth == 0.0). Also composites a
// physically-based sun disk using the transmittance LUT.
//
// Depth test is done in-shader via the bindless depth slot: if depth != 0.0
// the pixel has geometry and is discarded. This avoids a separate depth
// attachment and keeps the pass stateless with respect to depth writes.
//
// Outputs physical HDR radiance into the R16G16B16A16_SFLOAT HDR buffer.

#include "common.hlsl"
#include "atmosphere_common.hlsl"

[[vk::binding(2, 0)]] StructuredBuffer<float4> g_buffers[]  : register(t0, space1);
[[vk::binding(0, 0)]] Texture2D               g_textures[] : register(t0, space0);
[[vk::binding(3, 0)]] SamplerState            g_samplers[] : register(s0, space0);

struct PushBlock {
    uint   cameraSlot;
    uint   depthSlot;
    uint   skyViewLutSlot;
    uint   transmittanceLutSlot;
    uint   samplerSlot;
    uint   _pad0;
    uint   _pad1;
    uint   _pad2;
    float4 sunWorldDir;       // xyz = normalised direction FROM surface TO sun, w = intensity
    float4 cameraWorldPosKm;  // xyz = world-space position in km from planet centre
};
[[vk::push_constant]] PushBlock pc;

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

// Convert a world-space view direction to SkyView LUT UV.
// Must exactly invert atmosphere_skyview.hlsl's skyViewUVToDir().
//
// Parameterization (Hillaire 2020 horizon-concentrated):
//   u = azimuth / (2π)        wrapped to [0, 1]
//   v = 0.5 + 0.5 * sign(cosZenith) * sqrt(|cosZenith|)
float2 dirToSkyViewUV(float3 viewDir, float3 camPosKm) {
    // Planet-surface-relative local frame at camera position.
    float3 up       = normalize(camPosKm);
    // Check cross-product length BEFORE normalising to handle near-pole cameras.
    float3 rightRaw = cross(up, float3(0.0f, 0.0f, 1.0f));
    if (length(rightRaw) < 0.01f)
        rightRaw = cross(up, float3(1.0f, 0.0f, 0.0f));
    float3 right   = normalize(rightRaw);
    float3 forward = cross(right, up);

    // Project view direction into local frame.
    // In skyViewUVToDir: localDir = (sinZenith*cos(az))*right + cosZenith*up + (sinZenith*sin(az))*forward
    float localX = dot(viewDir, right);   // sinZenith * cos(az)
    float localY = dot(viewDir, up);      // cosZenith
    float localZ = dot(viewDir, forward); // sinZenith * sin(az)

    float az = atan2(localZ, localX);     // ∈ [-π, π]
    float u  = frac(az / (2.0f * PI) + 1.0f); // wrap to [0, 1]

    // sign(0) is undefined on some drivers → use explicit ternary for the horizon row.
    float cosZenith = localY;
    float signZ = (cosZenith >= 0.0f) ? 1.0f : -1.0f;
    float v = 0.5f + 0.5f * signZ * sqrt(abs(cosZenith));

    return float2(u, v);
}

// ---- Vertex shader: fullscreen triangle ----
struct VSOut {
    float4 pos      : SV_Position;
    float2 texCoord : TEXCOORD0; // [0,1]^2, origin top-left
};

VSOut VSMain(uint vid : SV_VertexID) {
    float2 uv = float2((vid << 1) & 2, vid & 2);
    VSOut o;
    o.pos      = float4(uv * 2.0f - 1.0f, 0.0f, 1.0f);
    o.texCoord = uv;
    return o;
}

// ---- Pixel shader ----
float4 PSMain(VSOut vs) : SV_Target {
    float2 uv   = vs.texCoord;
    SamplerState samp = g_samplers[NonUniformResourceIndex(pc.samplerSlot)];

    // Reverse-Z: sky pixels have depth == 0. Skip anything with geometry.
    float depth = g_textures[NonUniformResourceIndex(pc.depthSlot)].Sample(samp, uv).r;
    if (depth != 0.0f)
        discard;

    // Reconstruct world-space view direction from pixel UV.
    // Use two clip-space points at different depths to form a ray.
    CameraData cam = loadCamera(pc.cameraSlot);
    float2 ndc     = uv * 2.0f - 1.0f;

    // Reverse-Z: near = depth 1, far = depth 0.
    float4 nearH = mul(cam.invViewProj, float4(ndc, 1.0f, 1.0f)); // near plane
    float4 farH  = mul(cam.invViewProj, float4(ndc, 0.0f, 1.0f)); // far plane
    float3 nearWorld = nearH.xyz / nearH.w;
    float3 farWorld  = farH.xyz  / farH.w;
    float3 viewDir   = normalize(farWorld - nearWorld);

    // Ensure camera position is above the planet surface (fallback for origin / unit test).
    float3 camPosKm = pc.cameraWorldPosKm.xyz;
    if (dot(camPosKm, camPosKm) < 1.0f)
        camPosKm = float3(0.0f, R_EARTH + 0.001f, 0.0f);

    // Sample SkyView LUT.
    Texture2D svLut = g_textures[NonUniformResourceIndex(pc.skyViewLutSlot)];
    float2 svUV     = dirToSkyViewUV(viewDir, camPosKm);
    float3 skyColor = svLut.SampleLevel(samp, svUV, 0).rgb;

    // Sun disk — rendered when the ray is within the sun's angular radius.
    float3 sunDir  = normalize(pc.sunWorldDir.xyz);
    float  cosView = dot(viewDir, sunDir);

    // Angular half-diameter of the sun ≈ 0.265°.
    // kCosSunRadius is the hard edge; kCosSunEdge is slightly wider for a smooth limb.
    const float kCosSunRadius = 0.99998f; // cos(0.26°) hard centre
    const float kCosSunEdge   = 0.99990f; // cos(0.41°) soft outer edge

    if (cosView > kCosSunEdge) {
        float  camAlt    = max(0.0f, getAltitude(camPosKm));
        float  cosSunUp  = dot(normalize(camPosKm), sunDir);
        Texture2D tlut   = g_textures[NonUniformResourceIndex(pc.transmittanceLutSlot)];
        float3 Tsun      = sampleTransmittanceLUT(tlut, samp, camAlt, cosSunUp);

        float  diskMask  = smoothstep(kCosSunEdge, kCosSunRadius, cosView);

        // Convert solar irradiance to radiance (L = E / Ω).
        // Ω_sun ≈ 6.8e-5 sr → 1/Ω ≈ 14706
        const float kSunRadianceScale = 14706.0f;
        skyColor += Tsun * pc.sunWorldDir.w * kSunRadianceScale * diskMask;
    }

    return float4(skyColor, 1.0f);
}
