#ifndef COMMON_HLSL
#define COMMON_HLSL

// GPU-side camera data. Matches enigma::GpuCameraData in Camera.h.
// All members are float4 / float4x4 so std430 layout == C++ natural
// alignment — no scalarBlockLayout required.
//
// Buffer slot indices (each mat4 = 4×float4):
//   view[0..3]  proj[4..7]  viewProj[8..11]  prevViewProj[12..15]
//   invViewProj[16..19]  worldPos[20]
struct CameraData {
    float4x4 view;
    float4x4 proj;
    float4x4 viewProj;
    float4x4 prevViewProj;  // previous-frame viewProj for motion vectors
    float4x4 invViewProj;   // inverse viewProj for depth → world-pos reconstruction
    float4   worldPos;      // xyz = position, w unused
};

#endif // COMMON_HLSL
