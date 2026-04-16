// physics_debug.hlsl
// ==================
// Wireframe overlay for physics body debug visualization.
// Topology: LINE_LIST — two vertices per line, pulled from a bindless SSBO.
//
// LineVertex (16 bytes, one float4 in the SSBO):
//   xyz  = world-space endpoint position
//   w    = packed RGBA8 color (bit-cast via asuint)
//
// Push constants (16 bytes):
//   lineSSBOSlot  — bindless slot for the LineVertex array
//   cameraSlot    — bindless slot for CameraData (same layout as terrain)
//   pad0, pad1    — unused

#include "common.hlsl"

// Bindless storage-buffer array, binding 2, set 0.
[[vk::binding(2, 0)]]
StructuredBuffer<float4> g_buffers[] : register(t0, space1);

struct DebugPC {
    uint lineSSBOSlot;
    uint cameraSlot;
    uint pad0;
    uint pad1;
};
[[vk::push_constant]] DebugPC pc;

// Load camera via the shared bindless camera SSBO pattern.
// HLSL float4x4 is column-major; the SSBO stores row vectors, so transpose.
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

struct VSOut {
    float4 svPos : SV_Position;
    float4 color : COLOR0;
};

VSOut VSMain(uint vertId : SV_VertexID) {
    StructuredBuffer<float4> lineBuf = g_buffers[NonUniformResourceIndex(pc.lineSSBOSlot)];
    float4 v = lineBuf[vertId];

    CameraData cam = loadCamera(pc.cameraSlot);

    VSOut o;
    o.svPos = mul(cam.viewProj, float4(v.xyz, 1.0));

    // Unpack RGBA8 stored in the float's bit pattern.
    uint packed = asuint(v.w);
    o.color = float4(
        float((packed >>  0) & 0xFFu) * (1.0 / 255.0),
        float((packed >>  8) & 0xFFu) * (1.0 / 255.0),
        float((packed >> 16) & 0xFFu) * (1.0 / 255.0),
        float((packed >> 24) & 0xFFu) * (1.0 / 255.0)
    );
    return o;
}

float4 PSMain(VSOut i) : SV_Target0 {
    return i.color;
}
