#include "renderer/Renderer.h"

#include "core/Log.h"
#include "gfx/Allocator.h"
#include "gfx/Device.h"
#include "gfx/Instance.h"
#include "gfx/Swapchain.h"
#include "platform/Window.h"

namespace enigma {

Renderer::Renderer(Window& window)
    : m_window(window)
    , m_instance(std::make_unique<gfx::Instance>())
    , m_device(std::make_unique<gfx::Device>(*m_instance))
    , m_allocator(std::make_unique<gfx::Allocator>(*m_instance, *m_device))
    , m_swapchain(std::make_unique<gfx::Swapchain>(*m_instance, *m_device, m_window)) {
    ENIGMA_LOG_INFO("[renderer] constructed");
}

Renderer::~Renderer() {
    ENIGMA_LOG_INFO("[renderer] shutdown");
}

void Renderer::drawFrame() {
    // Real Vulkan frame is wired at step 38+.
}

} // namespace enigma
