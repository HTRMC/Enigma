#pragma once

#include "core/Math.h"
#include "core/Types.h"

struct GLFWwindow;

namespace enigma {

class Window;

// Polling-based input abstraction over GLFW. No callbacks are installed
// — all state is sampled once per frame via glfwGetKey / glfwGetCursorPos /
// glfwGetMouseButton so there are no conflicts with Window's own GLFW
// callbacks (framebuffer-size, etc.).
//
// Call `update()` exactly once per frame, after `Window::pollEvents()`.
class Input {
public:
    explicit Input(Window& window);

    // Sample current keyboard, mouse position, and mouse button state.
    // Computes mouse delta from previous frame's cursor position.
    void update();

    bool isKeyDown(int glfwKey) const;
    bool isMouseButtonDown(int glfwButton) const;

    vec2 mouseDelta() const { return m_mouseDelta; }

    void setCursorCaptured(bool captured);
    bool isCursorCaptured() const { return m_cursorCaptured; }

private:
    GLFWwindow* m_handle = nullptr;
    vec2        m_lastCursorPos{0.0f, 0.0f};
    vec2        m_mouseDelta{0.0f, 0.0f};
    bool        m_cursorCaptured = false;
    bool        m_firstUpdate    = true;
};

} // namespace enigma
