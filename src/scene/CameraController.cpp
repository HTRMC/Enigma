#include "scene/CameraController.h"

#include "core/Math.h"
#include "input/Input.h"
#include "scene/Camera.h"

#include <GLFW/glfw3.h>

#include <algorithm>

namespace enigma {

CameraController::CameraController(Camera& camera, Input& input)
    : m_camera(&camera), m_input(&input) {}

void CameraController::update(f32 deltaTime) {
    const Input& input = *m_input;
    // --- Cursor capture toggle via right mouse button ---
    const bool rmb = input.isMouseButtonDown(GLFW_MOUSE_BUTTON_RIGHT);
    if (rmb != m_input->isCursorCaptured()) {
        m_input->setCursorCaptured(rmb);
    }

    // --- Mouse look (only while captured) ---
    if (input.isCursorCaptured()) {
        const vec2 delta = input.mouseDelta();
        m_yaw   -= delta.x * m_camera->sensitivity;
        m_pitch -= delta.y * m_camera->sensitivity;
        m_pitch  = std::clamp(m_pitch, -89.0f, 89.0f);

        // Rebuild orientation from yaw + pitch (no roll).
        const quat qYaw   = glm::angleAxis(glm::radians(m_yaw),   vec3{0.0f, 1.0f, 0.0f});
        const quat qPitch = glm::angleAxis(glm::radians(m_pitch), vec3{1.0f, 0.0f, 0.0f});
        m_camera->orientation = qYaw * qPitch;
    }

    // --- Keyboard movement ---
    const f32 boost = input.isKeyDown(GLFW_KEY_LEFT_SHIFT) ? 3.0f : 1.0f;
    const f32 moveSpeed = m_camera->speed * boost * deltaTime;

    const vec3 fwd   = m_camera->forward();
    const vec3 right = m_camera->right();
    const vec3 worldUp{0.0f, 1.0f, 0.0f};

    vec3 move{0.0f};
    if (input.isKeyDown(GLFW_KEY_W)) move += fwd;
    if (input.isKeyDown(GLFW_KEY_S)) move -= fwd;
    if (input.isKeyDown(GLFW_KEY_D)) move += right;
    if (input.isKeyDown(GLFW_KEY_A)) move -= right;
    if (input.isKeyDown(GLFW_KEY_E)) move += worldUp;
    if (input.isKeyDown(GLFW_KEY_Q)) move -= worldUp;

    if (glm::dot(move, move) > 0.0f) {
        m_camera->position += glm::normalize(move) * moveSpeed;
    }
}

} // namespace enigma
