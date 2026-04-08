#include "renderer/Renderer.h"

#include "core/Log.h"
#include "gfx/Allocator.h"
#include "gfx/DescriptorAllocator.h"
#include "gfx/Device.h"
#include "gfx/FrameContext.h"
#include "gfx/Instance.h"
#include "gfx/ShaderManager.h"
#include "gfx/Swapchain.h"
#include "platform/Window.h"
#include "renderer/TrianglePass.h"

namespace enigma {

Renderer::Renderer(Window& window)
    : m_window(window)
    , m_instance(std::make_unique<gfx::Instance>())
    , m_device(std::make_unique<gfx::Device>(*m_instance))
    , m_allocator(std::make_unique<gfx::Allocator>(*m_instance, *m_device))
    , m_swapchain(std::make_unique<gfx::Swapchain>(*m_instance, *m_device, m_window))
    , m_frames(std::make_unique<gfx::FrameContextSet>(*m_device))
    , m_descriptorAllocator(std::make_unique<gfx::DescriptorAllocator>(*m_device))
    , m_shaderManager(std::make_unique<gfx::ShaderManager>(*m_device))
    , m_trianglePass(std::make_unique<TrianglePass>(*m_device, *m_allocator, *m_descriptorAllocator)) {

    m_trianglePass->buildPipeline(*m_shaderManager,
                                  m_descriptorAllocator->layout(),
                                  m_swapchain->format());

    ENIGMA_LOG_INFO("[renderer] constructed");
}

Renderer::~Renderer() {
    if (m_device) {
        vkDeviceWaitIdle(m_device->logical());
    }
    ENIGMA_LOG_INFO("[renderer] shutdown");
}

void Renderer::drawFrame() {
    // Acquire + submit + present path lands at plan step 39.
    // Sync2 image layout transitions land at plan step 40.
}

} // namespace enigma
