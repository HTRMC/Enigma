#include "platform/Window.h"

#include "core/Assert.h"
#include "core/Log.h"

#include <GLFW/glfw3.h>

#include <atomic>
#include <utility>

namespace enigma {

namespace {

// Reference-counted GLFW init: the first Window constructed calls
// glfwInit(); the last one destroyed calls glfwTerminate().
std::atomic<u32> g_glfwRefCount{0};

void glfwErrorCallback(int code, const char* description) {
    ENIGMA_LOG_ERROR("[glfw] error {}: {}", code, description ? description : "(null)");
}

void acquireGlfw() {
    if (g_glfwRefCount.fetch_add(1) == 0) {
        glfwSetErrorCallback(&glfwErrorCallback);
        if (glfwInit() != GLFW_TRUE) {
            ENIGMA_LOG_ERROR("[glfw] glfwInit failed");
            ENIGMA_ASSERT(false);
        }
    }
}

void releaseGlfw() {
    if (g_glfwRefCount.fetch_sub(1) == 1) {
        glfwTerminate();
    }
}

} // namespace

Window::Window(u32 width, u32 height, const char* title) {
    acquireGlfw();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE,  GLFW_TRUE);

    m_handle = glfwCreateWindow(
        static_cast<int>(width),
        static_cast<int>(height),
        title ? title : "Enigma",
        nullptr,
        nullptr);

    if (m_handle == nullptr) {
        ENIGMA_LOG_ERROR("[glfw] glfwCreateWindow failed");
        releaseGlfw();
        ENIGMA_ASSERT(false);
        return;
    }
}

Window::~Window() {
    if (m_handle != nullptr) {
        glfwDestroyWindow(m_handle);
        m_handle = nullptr;
        releaseGlfw();
    }
}

Window::Window(Window&& other) noexcept
    : m_handle(std::exchange(other.m_handle, nullptr))
    , m_resized(std::exchange(other.m_resized, false)) {}

Window& Window::operator=(Window&& other) noexcept {
    if (this != &other) {
        if (m_handle != nullptr) {
            glfwDestroyWindow(m_handle);
            releaseGlfw();
        }
        m_handle  = std::exchange(other.m_handle, nullptr);
        m_resized = std::exchange(other.m_resized, false);
    }
    return *this;
}

// --- Stubs that later steps (15) fill in -----------------------------------

void Window::pollEvents() {
    // Implemented in step 15.
}

bool Window::shouldClose() const {
    // Implemented in step 15.
    return true;
}

Window::Extent Window::framebufferSize() const {
    // Implemented in step 15.
    return {0, 0};
}

bool Window::wasResized() const {
    // Implemented in step 15.
    return false;
}

void Window::clearResized() {
    // Implemented in step 15.
}

void Window::waitEvents() const {
    // Implemented in step 15.
}

} // namespace enigma
