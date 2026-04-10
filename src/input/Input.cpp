#include "input/Input.h"

#include "platform/Window.h"

#include <GLFW/glfw3.h>

namespace enigma {

Input::Input(Window& window)
    : m_handle(window.handle()) {
    double x = 0.0;
    double y = 0.0;
    glfwGetCursorPos(m_handle, &x, &y);
    m_lastCursorPos = {static_cast<f32>(x), static_cast<f32>(y)};
}

void Input::update() {
    double x = 0.0;
    double y = 0.0;
    glfwGetCursorPos(m_handle, &x, &y);
    const vec2 current{static_cast<f32>(x), static_cast<f32>(y)};

    if (m_firstUpdate) {
        m_mouseDelta = {0.0f, 0.0f};
        m_firstUpdate = false;
    } else {
        m_mouseDelta = current - m_lastCursorPos;
    }
    m_lastCursorPos = current;
}

bool Input::isKeyDown(int glfwKey) const {
    return glfwGetKey(m_handle, glfwKey) == GLFW_PRESS;
}

bool Input::isMouseButtonDown(int glfwButton) const {
    return glfwGetMouseButton(m_handle, glfwButton) == GLFW_PRESS;
}

void Input::setCursorCaptured(bool captured) {
    m_cursorCaptured = captured;
    glfwSetInputMode(m_handle, GLFW_CURSOR,
                     captured ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
    if (captured) {
        // Reset delta so the first captured frame doesn't jump.
        m_firstUpdate = true;
    }
}

bool Input::isGamepadPresent(int gamepadId) const {
    return glfwJoystickPresent(gamepadId) == GLFW_TRUE;
}

f32 Input::getGamepadAxis(int gamepadId, int axis) const {
    int count = 0;
    const float* axes = glfwGetJoystickAxes(gamepadId, &count);
    if (!axes || axis >= count) return 0.0f;
    return axes[axis];
}

bool Input::isGamepadButtonDown(int gamepadId, int button) const {
    int count = 0;
    const unsigned char* buttons = glfwGetJoystickButtons(gamepadId, &count);
    if (!buttons || button >= count) return false;
    return buttons[button] == GLFW_PRESS;
}

} // namespace enigma
