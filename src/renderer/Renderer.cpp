#include "renderer/Renderer.h"

#include "core/Assert.h"
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

#include <algorithm>

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
    VkDevice dev = m_device->logical();
    gfx::FrameContext& frame = m_frames->get(m_frameIndex);

    // -------------------------------------------------------------------
    // Timeline semaphore wait: hold off until the FrameContext's last
    // submitted work (frameValue - MAX_FRAMES_IN_FLIGHT + 1) has retired.
    // For the first few frames this clamps to zero which is an immediate
    // no-wait.
    // -------------------------------------------------------------------
    const u64 waitValue = (frame.frameValue >= gfx::MAX_FRAMES_IN_FLIGHT)
                              ? (frame.frameValue - gfx::MAX_FRAMES_IN_FLIGHT + 1)
                              : 0;
    if (waitValue > 0) {
        VkSemaphoreWaitInfo waitInfo{};
        waitInfo.sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        waitInfo.semaphoreCount = 1;
        waitInfo.pSemaphores    = &frame.inFlight;
        waitInfo.pValues        = &waitValue;
        ENIGMA_VK_CHECK(vkWaitSemaphores(dev, &waitInfo, UINT64_MAX));
    }

    // -------------------------------------------------------------------
    // Acquire next swapchain image.
    // -------------------------------------------------------------------
    u32 imageIndex = 0;
    ENIGMA_VK_CHECK(vkAcquireNextImageKHR(dev, m_swapchain->handle(), UINT64_MAX,
                                          frame.imageAvailable, VK_NULL_HANDLE, &imageIndex));

    // -------------------------------------------------------------------
    // Record. Reset the per-frame command pool, begin the command
    // buffer, record the triangle pass, end.
    //
    // NOTE: At step 39 the color attachment is still in UNDEFINED layout
    // and no dynamic-rendering begin/end is wired. Validation will fire
    // at this commit. Step 40 adds the sync2 layout transitions and
    // vkCmdBeginRendering/vkCmdEndRendering to make the frame
    // validation-clean.
    //
    // BISECT WARNING: do not `git bisect` any unrelated regression
    // across the boundary between this commit and step 40 — the engine
    // is intentionally not validation-clean at this intermediate state.
    // -------------------------------------------------------------------
    ENIGMA_VK_CHECK(vkResetCommandPool(dev, frame.commandPool, 0));

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    ENIGMA_VK_CHECK(vkBeginCommandBuffer(frame.commandBuffer, &beginInfo));

    // Triangle pass draw is recorded here at step 40 once layout
    // transitions + dynamic rendering begin/end are in place. For step
    // 39 we intentionally record nothing so the submit/present
    // plumbing is exercised standalone.
    (void)imageIndex;

    ENIGMA_VK_CHECK(vkEndCommandBuffer(frame.commandBuffer));

    // -------------------------------------------------------------------
    // Submit. Waits on `imageAvailable` (binary), signals
    // `renderFinished` (binary) for present, plus the timeline
    // `inFlight` at value `frameValue + 1` for CPU/GPU pipelining.
    // -------------------------------------------------------------------
    const u64 signalValue = frame.frameValue + 1;

    const VkSemaphore        waitSems[]   = { frame.imageAvailable };
    const VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    const VkSemaphore        signalSems[] = { frame.renderFinished, frame.inFlight };
    const u64                signalValues[] = { 0, signalValue };

    VkTimelineSemaphoreSubmitInfo timelineInfo{};
    timelineInfo.sType                     = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timelineInfo.waitSemaphoreValueCount   = 0;
    timelineInfo.pWaitSemaphoreValues      = nullptr;
    timelineInfo.signalSemaphoreValueCount = 2;
    timelineInfo.pSignalSemaphoreValues    = signalValues;

    VkSubmitInfo submitInfo{};
    submitInfo.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pNext                = &timelineInfo;
    submitInfo.waitSemaphoreCount   = 1;
    submitInfo.pWaitSemaphores      = waitSems;
    submitInfo.pWaitDstStageMask    = waitStages;
    submitInfo.commandBufferCount   = 1;
    submitInfo.pCommandBuffers      = &frame.commandBuffer;
    submitInfo.signalSemaphoreCount = 2;
    submitInfo.pSignalSemaphores    = signalSems;

    ENIGMA_VK_CHECK(vkQueueSubmit(m_device->graphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE));
    frame.frameValue = signalValue;

    // -------------------------------------------------------------------
    // Present.
    // -------------------------------------------------------------------
    VkSwapchainKHR swapchain = m_swapchain->handle();

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores    = &frame.renderFinished;
    presentInfo.swapchainCount     = 1;
    presentInfo.pSwapchains        = &swapchain;
    presentInfo.pImageIndices      = &imageIndex;

    vkQueuePresentKHR(m_device->graphicsQueue(), &presentInfo);

    m_frameIndex = (m_frameIndex + 1) % gfx::MAX_FRAMES_IN_FLIGHT;
}

} // namespace enigma
