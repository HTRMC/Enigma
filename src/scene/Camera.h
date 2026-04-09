#pragma once

#include "core/Math.h"
#include "core/Types.h"

namespace enigma {

// GPU-side camera data layout. All members are 16-byte aligned so
// std430 layout matches C++ natural alignment — no scalarBlockLayout
// needed. Uploaded to a bindless SSBO (binding 2) each frame.
struct GpuCameraData {
    mat4 view;
    mat4 proj;
    mat4 viewProj;
    vec4 worldPos; // xyz = position, w unused
};

static_assert(sizeof(GpuCameraData) == 208);

// First-person camera with reverse-Z infinite perspective projection.
// Orientation is quaternion-based (no gimbal lock). Aspect ratio is
// NOT stored — derived from the swapchain extent each frame.
class Camera {
public:
    Camera() = default;
    explicit Camera(vec3 position, f32 fovYDegrees = 60.0f, f32 nearPlane = 0.1f);

    mat4 viewMatrix() const;
    mat4 projMatrix(f32 aspect) const;
    mat4 viewProjMatrix(f32 aspect) const;

    vec3 forward() const;
    vec3 right()   const;
    vec3 up()      const;

    GpuCameraData gpuData(f32 aspect) const;

    vec3 position{0.0f, 0.0f, 0.0f};
    quat orientation{1.0f, 0.0f, 0.0f, 0.0f};
    f32  fovY      = glm::radians(60.0f);
    f32  nearPlane = 0.1f;
    f32  speed     = 5.0f;
    f32  sensitivity = 0.1f;
};

} // namespace enigma
