#pragma once

#include "core/Types.h"
#include "gfx/GpuProfiler.h"
#include "renderer/UpscalerSettings.h"

#include <volk.h>

#include <span>
#include <string>
#include <vector>

struct GLFWwindow;

namespace enigma::gfx {

class Device;

// ImGuiLayer
// ==========
// Owns the Dear ImGui Vulkan + GLFW backend lifecycle.
// Usage per frame:
//   layer.newFrame();
//   // ImGui:: calls here (panels, windows, etc.)
//   layer.render(cmd, swapchainImage, swapchainView, extent);
//
// render() emits its own vkCmdBeginRendering / vkCmdEndRendering and handles
// the image layout transition PRESENT_SRC_KHR -> COLOR_ATTACHMENT_OPTIMAL
// -> PRESENT_SRC_KHR internally.
class ImGuiLayer {
public:
    ImGuiLayer(Device& device,
               VkInstance    instance,
               GLFWwindow*   window,
               VkFormat      swapchainFormat,
               u32           imageCount);
    ~ImGuiLayer();

    ImGuiLayer(const ImGuiLayer&)            = delete;
    ImGuiLayer& operator=(const ImGuiLayer&) = delete;

    // Call once per frame before any ImGui:: window calls.
    void newFrame();

    // Record ImGui draw commands into cmd. Handles its own BeginRendering.
    // swapchainImage:  the VkImage being presented (needed for barriers).
    // swapchainView:   the VkImageView for that image.
    // incomingLayout:  the actual layout the image is in when render() is called.
    //   Pass VK_IMAGE_LAYOUT_PRESENT_SRC_KHR when only the render graph ran.
    //   Pass VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL when the upscaler ran last.
    // After this call the image is in VK_IMAGE_LAYOUT_PRESENT_SRC_KHR.
    void render(VkCommandBuffer cmd,
                VkImage         swapchainImage,
                VkImageView     swapchainView,
                VkExtent2D      extent,
                VkImageLayout   incomingLayout);

    // Convenience panel builders -- call between newFrame() and render().
    // cpuFrameTimeMs  : wall-clock frame delta (ms) for FPS display.
    // vramUsedMb/vramBudgetMb : device-local heap usage from vmaGetHeapBudgets.
    void drawGpuTimings(std::span<const GpuProfiler::ZoneResult> results,
                        f32 cpuFrameTimeMs,
                        f32 vramUsedMb, f32 vramBudgetMb);
    void drawSceneInfo(u32 primitiveCount, u32 tlasInstances);
    void drawUpscalerSettings(UpscalerSettings& settings);
    void drawPhysicsStats(f32 stepTimeMs, u32 bodyCount);
    void drawPhysicsDebugPanel(bool& enabled, bool& depthTest);

private:
    VkDevice         m_device = VK_NULL_HANDLE;
    VkDescriptorPool m_pool   = VK_NULL_HANDLE;
};

} // namespace enigma::gfx
