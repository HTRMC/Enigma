#include "gfx/RenderGraph.h"

#include "core/Assert.h"
#include "core/Log.h"

namespace enigma::gfx {

// ---------------------------------------------------------------------------
// Barrier helpers
// ---------------------------------------------------------------------------

VkPipelineStageFlags2 RenderGraph::srcStageFor(VkImageLayout layout) {
    switch (layout) {
        case VK_IMAGE_LAYOUT_UNDEFINED:
            // No prior work to wait for — TOP_OF_PIPE means "nothing".
            return VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;

        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            return VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;

        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
        case VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL:
            return VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                   VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;

        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            return VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;

        case VK_IMAGE_LAYOUT_GENERAL:
            return VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            return VK_PIPELINE_STAGE_2_TRANSFER_BIT;

        case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
            return VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;

        default:
            return VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    }
}

VkAccessFlags2 RenderGraph::srcAccessFor(VkImageLayout layout) {
    switch (layout) {
        case VK_IMAGE_LAYOUT_UNDEFINED:
        case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
            return 0;

        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            return VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;

        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
        case VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL:
            return VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            return VK_ACCESS_2_SHADER_READ_BIT;

        case VK_IMAGE_LAYOUT_GENERAL:
            return VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;

        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            return VK_ACCESS_2_TRANSFER_READ_BIT;

        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            return VK_ACCESS_2_TRANSFER_WRITE_BIT;

        default:
            return VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    }
}

void RenderGraph::emitBarrier(VkCommandBuffer       cmd,
                              const ImageResource&  img,
                              VkImageLayout         newLayout,
                              VkPipelineStageFlags2 dstStage,
                              VkAccessFlags2        dstAccess) {
    VkImageMemoryBarrier2 barrier{};
    barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask        = srcStageFor(img.currentLayout);
    barrier.srcAccessMask       = srcAccessFor(img.currentLayout);
    barrier.dstStageMask        = dstStage;
    barrier.dstAccessMask       = dstAccess;
    barrier.oldLayout           = img.currentLayout;
    barrier.newLayout           = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image               = img.image;
    barrier.subresourceRange    = {img.aspect, 0, 1, 0, 1};

    VkDependencyInfo dep{};
    dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers    = &barrier;
    vkCmdPipelineBarrier2(cmd, &dep);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

RGImageHandle RenderGraph::importImage(std::string_view   name,
                                       VkImage            image,
                                       VkImageView        view,
                                       VkFormat           format,
                                       VkImageLayout      initialLayout,
                                       VkImageLayout      finalLayout,
                                       VkImageAspectFlags aspect) {
    const u32 index = static_cast<u32>(m_images.size());
    m_images.push_back({std::string(name), image, view, format,
                        aspect, initialLayout, finalLayout});
    return RGImageHandle{index};
}

void RenderGraph::addRasterPass(RasterPassDesc desc) {
    RasterPassNode node{};
    node.name          = std::move(desc.name);
    node.colorTargets  = std::move(desc.colorTargets);
    node.depthTarget   = desc.depthTarget;
    node.sampledInputs = std::move(desc.sampledInputs);
    node.clearColor    = desc.clearColor;
    node.clearDepth    = desc.clearDepth;
    node.colorLoadOp   = desc.colorLoadOp;
    node.depthLoadOp   = desc.depthLoadOp;
    node.execute       = std::move(desc.execute);
    m_passes.push_back(std::move(node));
}

void RenderGraph::execute(VkCommandBuffer cmd, VkExtent2D extent) {
    for (auto& pass : m_passes) {
        // Pre-pass barriers (1): transition sampled inputs to SHADER_READ_ONLY_OPTIMAL.
        // These must be emitted before attachment barriers and before BeginRendering.
        for (const RGImageHandle h : pass.sampledInputs) {
            ENIGMA_ASSERT(h.valid());
            auto& img = m_images[h.index];
            constexpr VkImageLayout kReadLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            if (img.currentLayout != kReadLayout) {
                emitBarrier(cmd, img, kReadLayout,
                            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                            VK_ACCESS_2_SHADER_READ_BIT);
                img.currentLayout = kReadLayout;
            }
        }

        // Pre-pass barriers (2): transition each attachment to the required layout.
        for (const RGImageHandle h : pass.colorTargets) {
            ENIGMA_ASSERT(h.valid());
            auto& img = m_images[h.index];
            if (img.currentLayout != VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
                emitBarrier(cmd, img,
                            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
                img.currentLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            }
        }
        if (pass.depthTarget.valid()) {
            auto& img = m_images[pass.depthTarget.index];
            const VkImageLayout depthLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            if (img.currentLayout != depthLayout) {
                emitBarrier(cmd, img, depthLayout,
                            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                            VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT);
                img.currentLayout = depthLayout;
            }
        }

        // Begin dynamic rendering.
        std::vector<VkRenderingAttachmentInfo> colorAttachInfos;
        colorAttachInfos.reserve(pass.colorTargets.size());
        for (const RGImageHandle h : pass.colorTargets) {
            const auto& img = m_images[h.index];
            VkRenderingAttachmentInfo ai{};
            ai.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            ai.imageView   = img.view;
            ai.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            ai.loadOp      = pass.colorLoadOp;
            ai.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
            ai.clearValue.color = pass.clearColor;
            colorAttachInfos.push_back(ai);
        }

        VkRenderingAttachmentInfo depthInfo{};
        VkRenderingAttachmentInfo* pDepth = nullptr;
        if (pass.depthTarget.valid()) {
            const auto& img = m_images[pass.depthTarget.index];
            depthInfo.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            depthInfo.imageView   = img.view;
            depthInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depthInfo.loadOp      = pass.depthLoadOp;
            depthInfo.storeOp     = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depthInfo.clearValue.depthStencil = pass.clearDepth;
            pDepth = &depthInfo;
        }

        VkRenderingInfo renderingInfo{};
        renderingInfo.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
        renderingInfo.renderArea.offset    = {0, 0};
        renderingInfo.renderArea.extent    = extent;
        renderingInfo.layerCount           = 1;
        renderingInfo.colorAttachmentCount = static_cast<u32>(colorAttachInfos.size());
        renderingInfo.pColorAttachments    = colorAttachInfos.empty() ? nullptr : colorAttachInfos.data();
        renderingInfo.pDepthAttachment     = pDepth;

        vkCmdBeginRendering(cmd, &renderingInfo);
        if (pass.execute) {
            pass.execute(cmd, extent);
        }
        vkCmdEndRendering(cmd);
    }

    // Post-all-passes: transition imported images to their declared finalLayout.
    for (auto& img : m_images) {
        if (img.finalLayout == VK_IMAGE_LAYOUT_UNDEFINED ||
            img.currentLayout == img.finalLayout) {
            continue;
        }
        emitBarrier(cmd, img, img.finalLayout,
                    VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0);
        img.currentLayout = img.finalLayout;
    }
}

void RenderGraph::reset() {
    m_images.clear();
    m_passes.clear();
}

} // namespace enigma::gfx
