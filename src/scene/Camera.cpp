#include "scene/Camera.h"

#include <cmath>
#include <glm/gtc/matrix_inverse.hpp>

namespace enigma {

Camera::Camera(vec3 pos, f32 fovYDegrees, f32 near)
    : position(pos)
    , fovY(glm::radians(fovYDegrees))
    , nearPlane(near) {}

mat4 Camera::viewMatrix() const {
    // Quaternion → rotation matrix, then translate by -position in
    // the rotated frame.
    const mat4 rot = glm::mat4_cast(glm::conjugate(orientation));
    const mat4 trans = glm::translate(mat4{1.0f}, -position);
    return rot * trans;
}

mat4 Camera::projMatrix(f32 aspect) const {
    // Reverse-Z infinite perspective (UE5 / Decima pattern).
    //
    // Standard perspective maps [near, far] → [0, 1].
    // Reverse-Z maps [near, ∞] → [1, 0]:
    //   proj[0][0] = f / aspect
    //   proj[1][1] = f            (Vulkan NDC: +Y down, handled by negation)
    //   proj[2][2] = 0
    //   proj[2][3] = -1           (perspective divide)
    //   proj[3][2] = nearPlane    (maps near plane to Z=1)
    //
    // This puts maximum float32 precision at the near plane and
    // gracefully degrades toward zero at infinity — the opposite of
    // standard [0,1] mapping, which wastes precision at distance.
    const f32 f = 1.0f / std::tan(fovY * 0.5f);

    mat4 p{0.0f};
    p[0][0] = f / aspect;
    p[1][1] = -f;           // Vulkan +Y down
    p[2][2] = 0.0f;
    p[2][3] = -1.0f;
    p[3][2] = nearPlane;
    return p;
}

mat4 Camera::viewProjMatrix(f32 aspect) const {
    return projMatrix(aspect) * viewMatrix();
}

vec3 Camera::forward() const {
    return glm::rotate(orientation, vec3{0.0f, 0.0f, -1.0f});
}

vec3 Camera::right() const {
    return glm::rotate(orientation, vec3{1.0f, 0.0f, 0.0f});
}

vec3 Camera::up() const {
    return glm::rotate(orientation, vec3{0.0f, 1.0f, 0.0f});
}

GpuCameraData Camera::gpuData(f32 aspect) const {
    GpuCameraData data{};
    data.view        = viewMatrix();
    data.proj        = projMatrix(aspect);
    data.viewProj    = data.proj * data.view;
    data.invViewProj = glm::inverse(data.viewProj);
    data.worldPos    = vec4{position, 1.0f};
    // prevViewProj is zero-initialized; Renderer patches it each frame.
    return data;
}

} // namespace enigma
