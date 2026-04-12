#include "gfx/ImGuiLayer.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "gfx/Device.h"
#include "renderer/Upscaler.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

namespace enigma::gfx {

ImGuiLayer::ImGuiLayer(Device&     device,
                       VkInstance  instance,
                       GLFWwindow* window,
                       VkFormat    swapchainFormat,
                       u32         imageCount)
    : m_device(device.logical()) {

    // ImGui needs its own descriptor pool with FREE_DESCRIPTOR_SET_BIT.
    // The bindless pool does not have this flag.
    VkDescriptorPoolSize poolSizes[] = {
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16 },
    };
    VkDescriptorPoolCreateInfo poolCI{};
    poolCI.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCI.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolCI.maxSets       = 16;
    poolCI.poolSizeCount = 1;
    poolCI.pPoolSizes    = poolSizes;
    ENIGMA_VK_CHECK(vkCreateDescriptorPool(m_device, &poolCI, nullptr, &m_pool));

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForVulkan(window, /*install_callbacks=*/true);

    VkPipelineRenderingCreateInfoKHR pipelineRenderingCI{};
    pipelineRenderingCI.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
    pipelineRenderingCI.colorAttachmentCount    = 1;
    pipelineRenderingCI.pColorAttachmentFormats = &swapchainFormat;

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.Instance            = instance; // needed to load vkCmdBeginRenderingKHR via vkGetInstanceProcAddr
    initInfo.PhysicalDevice      = device.physical();
    initInfo.Device              = device.logical();
    initInfo.QueueFamily         = device.graphicsQueueFamily();
    initInfo.Queue               = device.graphicsQueue();
    initInfo.DescriptorPool      = m_pool;
    initInfo.MinImageCount       = 2;
    initInfo.ImageCount          = imageCount;
    initInfo.MSAASamples         = VK_SAMPLE_COUNT_1_BIT;
    initInfo.UseDynamicRendering = true;
    initInfo.PipelineRenderingCreateInfo = pipelineRenderingCI;

    ImGui_ImplVulkan_Init(&initInfo);

    // Upload fonts via a one-shot command buffer on the graphics queue.
    ImGui_ImplVulkan_CreateFontsTexture();

    ENIGMA_LOG_INFO("[imgui] initialized (dynamic rendering, format={})", static_cast<u32>(swapchainFormat));
}

ImGuiLayer::~ImGuiLayer() {
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    if (m_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_device, m_pool, nullptr);
    }
}

void ImGuiLayer::newFrame() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void ImGuiLayer::render(VkCommandBuffer cmd,
                        VkImage         swapchainImage,
                        VkImageView     swapchainView,
                        VkExtent2D      extent,
                        VkImageLayout   incomingLayout) {
    ImGui::Render();

    // Transition incomingLayout -> COLOR_ATTACHMENT_OPTIMAL for rendering.
    // incomingLayout is PRESENT_SRC_KHR when only the render graph ran, or
    // COLOR_ATTACHMENT_OPTIMAL when the upscaler ran last. The caller must
    // pass the actual layout — guessing here causes validation errors.
    {
        VkImageMemoryBarrier2 barrier{};
        barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.srcStageMask        = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
        barrier.srcAccessMask       = 0;
        barrier.dstStageMask        = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        barrier.dstAccessMask       = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.oldLayout           = incomingLayout;
        barrier.newLayout           = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image               = swapchainImage;
        barrier.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        VkDependencyInfo dep{};
        dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &barrier;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    VkRenderingAttachmentInfo colorAttach{};
    colorAttach.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttach.imageView   = swapchainView;
    colorAttach.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttach.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;   // preserve scene
    colorAttach.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea.offset    = {0, 0};
    renderingInfo.renderArea.extent    = extent;
    renderingInfo.layerCount           = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments    = &colorAttach;

    vkCmdBeginRendering(cmd, &renderingInfo);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    vkCmdEndRendering(cmd);

    // Transition COLOR_ATTACHMENT_OPTIMAL -> PRESENT_SRC_KHR for present.
    {
        VkImageMemoryBarrier2 barrier{};
        barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.srcStageMask        = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        barrier.srcAccessMask       = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstStageMask        = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
        barrier.dstAccessMask       = 0;
        barrier.oldLayout           = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrier.newLayout           = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image               = swapchainImage;
        barrier.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        VkDependencyInfo dep{};
        dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &barrier;
        vkCmdPipelineBarrier2(cmd, &dep);
    }
}

void ImGuiLayer::drawGpuTimings(std::span<const GpuProfiler::ZoneResult> results,
                                f32 cpuFrameTimeMs,
                                f32 vramUsedMb, f32 vramBudgetMb) {
    ImGui::SetNextWindowPos({10.f, 10.f},  ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize({300.f, 230.f}, ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Performance")) {
        // CPU frame time + FPS
        const f32 fps = cpuFrameTimeMs > 0.f ? 1000.f / cpuFrameTimeMs : 0.f;
        ImGui::Text("CPU  %.2f ms   %.0f FPS", cpuFrameTimeMs, fps);

        // Total GPU time (sum of all profiled zones)
        f32 totalGpu = 0.f;
        for (const auto& r : results) totalGpu += r.durationMs;
        ImGui::Text("GPU  %.3f ms", totalGpu);

        // VRAM progress bar
        if (vramBudgetMb > 0.f) {
            char overlay[64];
            snprintf(overlay, sizeof(overlay), "%.0f / %.0f MB", vramUsedMb, vramBudgetMb);
            ImGui::ProgressBar(vramUsedMb / vramBudgetMb, {-1.f, 0.f}, overlay);
            ImGui::SameLine(0.f, ImGui::GetStyle().ItemInnerSpacing.x);
            ImGui::Text("VRAM");
        }

        ImGui::Separator();

        // Per-pass GPU breakdown
        for (const auto& r : results) {
            ImGui::Text("  %-22s %.3f ms", r.name.c_str(), r.durationMs);
        }
        if (results.empty()) {
            ImGui::TextDisabled("(no data yet)");
        }
    }
    ImGui::End();
}

void ImGuiLayer::drawSceneInfo(u32 primitiveCount, u32 tlasInstances) {
    ImGui::SetNextWindowPos({10.f, 160.f}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize({290.f, 70.f}, ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Scene")) {
        ImGui::Text("Primitives : %u", primitiveCount);
        ImGui::Text("TLAS inst  : %u", tlasInstances);
    }
    ImGui::End();
}

void ImGuiLayer::drawUpscalerSettings(UpscalerSettings& settings) {
    ImGui::SetNextWindowPos({10.f, 340.f}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize({290.f, 130.f}, ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Upscaler")) {
        const char* qualityNames[] = {
            "Ultra Performance", "Performance", "Balanced",
            "Quality", "Ultra Quality", "Native AA"
        };
        int q = static_cast<int>(settings.quality);
        if (ImGui::Combo("Quality", &q, qualityNames, 6)) {
            settings.quality = static_cast<UpscalerQuality>(q);
        }
        ImGui::SliderFloat("Sharpness", &settings.sharpness, 0.0f, 1.0f);
        ImGui::Checkbox("Frame Gen (RTX 40+)", &settings.frameGenEnabled);

        const char* backendName = "?";
        switch (settings.backend) {
            case UpscalerBackend::None: backendName = "None (pass-through)"; break;
            case UpscalerBackend::DLSS: backendName = "DLSS";  break;
            case UpscalerBackend::XeSS: backendName = "XeSS";  break;
            case UpscalerBackend::FSR:  backendName = "FSR";   break;
        }
        ImGui::LabelText("Backend", "%s", backendName);
    }
    ImGui::End();
}

void ImGuiLayer::drawPhysicsStats(f32 stepTimeMs, u32 bodyCount) {
    ImGui::SetNextWindowPos({10.f, 240.f}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize({290.f, 90.f}, ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Physics")) {
        ImGui::Text("Step time  : %.3f ms", stepTimeMs);
        ImGui::Text("Body count : %u",      bodyCount);
    }
    ImGui::End();
}

void ImGuiLayer::drawPhysicsDebugPanel(bool& enabled, bool& depthTest) {
    ImGui::SetNextWindowPos({10.f, 340.f}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize({260.f, 70.f}, ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Physics Debug")) {
        ImGui::Checkbox("Wireframes [F3]", &enabled);
        if (enabled) {
            ImGui::SameLine();
            ImGui::Checkbox("Depth Test", &depthTest);
        }
    }
    ImGui::End();
}

} // namespace enigma::gfx
