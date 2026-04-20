// debug_micropoly_bounds.hlsl
// ==========================
// M6.2b: per-cluster bounding-sphere wireframe overlay. Fullscreen PS that
// iterates the DAG, projects each cluster's bounding sphere to screen
// space, and tints any pixel within `kOutlineThickness` of a sphere's
// projected-circle outline.
//
// This completes the M6 debug-overlay inventory (Bounds / LOD / HwSwClass
// / Residency). Unlike the vis-image-sampling overlays (M4.6 / M6.1) this
// one does NOT read the 64-bit vis image — it derives everything from
// the DAG SSBO + the camera matrices. Availability still gates on the DAG
// buffer being wired (same `mpBoundsAvail` flag in Renderer.cpp).
//
// Complexity: O(pixels x clusters). Hard-capped to kMaxClustersToIterate
// so the worst-case per-pixel cost stays bounded when large assets push
// the DAG node count into the tens of thousands. Acceptable for debug
// mode; NOT production-viable.
//
// Color: per-cluster hue cycled by index (golden-ratio step through HSV)
// so adjacent clusters are visually distinguishable.

#include "common.hlsl"

// StructuredBuffer<float4> bindless alias — same binding as mp_cluster_cull
// and the other M6.1 overlays.
[[vk::binding(2, 0)]]
StructuredBuffer<float4> g_buffers[] : register(t0, space1);

struct PushBlock {
    uint dagBufferBindless;
    uint dagNodeCount;
    uint cameraSlot;
    uint screenWidth;
    uint screenHeight;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};
[[vk::push_constant]] PushBlock pc;

struct VSOut {
    float4 pos : SV_Position;
    float2 uv  : TEXCOORD0;
};

VSOut VSMain(uint vid : SV_VertexID) {
    float2 uv = float2((vid << 1) & 2, vid & 2);
    VSOut o;
    o.pos = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    o.uv  = uv;
    return o;
}

// --- tuning knobs -----------------------------------------------------------
static const float kOutlineThickness    = 1.5f;      // pixels
// Per-pixel worst-case cap. Previously 4096 — at 1080p that's ~8.5 B ops
// per frame and tripped the Windows 2s TDR on large assets (real-world
// observation: BMW GT3 asset with 20K clusters + 42K DAG nodes device-lost
// within a frame when this overlay was selected). The overlay is a DEBUG
// visualisation; dropping to 256 keeps the worst case at ~530M ops
// (~50ms) which is comfortably under TDR and still gives representative
// coverage — only the FIRST 256 clusters in the DAG get outlines but
// that's fine for "roughly where are the large cluster groups".
static const uint  kMaxClustersToIterate = 256u;     // per-pixel worst-case cap

// --- Camera load (column-major GLM -> HLSL row-major under -Zpc) ------------
// Mirrors loadCamera() in mp_cluster_cull.comp.hlsl. DXC's float4x4(v0..v3)
// ctor takes COLUMNS under -Zpc -spirv; transpose() gives us the
// engine's row-major convention so mul(cam.viewProj, p) is correct.
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

// HSV->RGB (H in [0,1], S, V in [0,1]). Standard formula.
float3 hsvToRgb(float h, float s, float v) {
    float3 k = float3(5.0f, 3.0f, 1.0f);
    float3 p = abs(frac(h.xxx + k / 6.0f) * 6.0f - 3.0f);
    return v * lerp(float3(1.0f, 1.0f, 1.0f), saturate(p - 1.0f), s);
}

float3 clusterColor(uint clusterIdx) {
    // Golden-ratio hue step for good visual separation between adjacent IDs.
    const float hue = frac(float(clusterIdx) * 0.6180339887498949f);
    return hsvToRgb(hue, 0.85f, 1.0f);
}

float4 PSMain(VSOut vs) : SV_Target {
    // Defensive: if the DAG buffer isn't wired the Renderer shouldn't route
    // us here at all, but fall through to black so a wiring bug is visible
    // rather than sampling an invalid bindless slot.
    if (pc.dagBufferBindless == 0xFFFFFFFFu || pc.dagNodeCount == 0u) {
        return float4(0.0f, 0.0f, 0.0f, 1.0f);
    }

    const float2 pixelCoord = vs.pos.xy;
    const float2 screenSize = float2(float(pc.screenWidth),
                                     float(pc.screenHeight));

    CameraData cam = loadCamera(pc.cameraSlot);

    // Projection-to-pixel scale: projRadius (in pixels) for a sphere of
    // world-space radius r at clip-space depth clipW is
    //     projRadius = r * projScale / clipW
    // where projScale = (-cam.proj[1][1]) * (screenHeight / 2). Same derivation
    // as mp_cluster_cull's classifyRasterClass() pxRadius.
    const float projScaleY = -cam.proj[1][1] * 0.5f * screenSize.y;

    // StructuredBuffer<float4> handle — read the DAG node m0 per iteration.
    StructuredBuffer<float4> dag = g_buffers[NonUniformResourceIndex(pc.dagBufferBindless)];

    const uint iterCount = min(pc.dagNodeCount, kMaxClustersToIterate);
    for (uint i = 0u; i < iterCount; ++i) {
        // MpDagNode::m0 = (center.xyz, radius). Layout matches loadDagNode()
        // in mp_cluster_cull.comp.hlsl — 5 float4 per node (M4 widening →
        // m3 for SSE errors; M4-fix widening → m4 for parentCenter).
        float4 m0 = dag[i * 5u + 0u];
        const float3 center = m0.xyz;
        const float  radius = m0.w;

        // Project centre to clip space.
        float4 clip = mul(cam.viewProj, float4(center, 1.0f));
        if (clip.w <= 0.0f) continue; // behind near plane

        // Screen-space centre in pixel coordinates.
        const float invW   = 1.0f / clip.w;
        const float2 ndc   = clip.xy * invW;
        const float2 uvTex = ndc * 0.5f + 0.5f; // [0,1]
        const float2 centerPx = uvTex * screenSize;

        // Projected radius in pixels.
        const float projRadius = radius * projScaleY * invW;
        if (projRadius < 1.0f) continue; // sub-pixel spheres are invisible

        // Distance from current pixel to the projected circle's outline.
        const float dist = length(pixelCoord - centerPx);
        if (abs(dist - projRadius) < kOutlineThickness) {
            return float4(clusterColor(i), 1.0f);
        }
    }

    return float4(0.0f, 0.0f, 0.0f, 1.0f);
}
