#ifndef COMMON_HLSL
#define COMMON_HLSL

// GPU-side camera data. Matches enigma::GpuCameraData in Camera.h.
// All members are float4 / float4x4 so std430 layout == C++ natural
// alignment — no scalarBlockLayout required.
struct CameraData {
    float4x4 view;
    float4x4 proj;
    float4x4 viewProj;
    float4   worldPos; // xyz = position, w unused
};

#endif // COMMON_HLSL
