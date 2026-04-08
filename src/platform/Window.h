#pragma once

#include "core/Types.h"

struct GLFWwindow;

namespace enigma {

// RAII GLFW window. Move-only. Construction initializes GLFW on first use
// and terminates GLFW when the last live Window is destroyed.
class Window {
public:
    Window(u32 width, u32 height, const char* title);
    ~Window();

    Window(const Window&)            = delete;
    Window& operator=(const Window&) = delete;

    Window(Window&& other) noexcept;
    Window& operator=(Window&& other) noexcept;

    void pollEvents();
    bool shouldClose() const;

    // Framebuffer size in pixels (not screen coordinates). For resize /
    // swapchain sizing.
    struct Extent {
        u32 width;
        u32 height;
    };
    Extent framebufferSize() const;

    // Resize flag set by GLFW's framebuffer-size callback; sticky until
    // consumed via clearResized().
    bool wasResized() const;
    void clearResized();

    // Blocks until at least one event arrives. Used when the window is
    // minimized (0x0) so we don't spin the frame loop.
    void waitEvents() const;

    GLFWwindow* handle() const { return m_handle; }

private:
    GLFWwindow* m_handle   = nullptr;
    bool        m_resized  = false;
};

} // namespace enigma
