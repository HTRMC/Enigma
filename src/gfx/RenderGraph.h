#pragma once

#include "core/Types.h"
#include "gfx/RenderGraphResources.h"

#include <volk.h>

#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace enigma::gfx {

// RenderGraph — Minimal linear render graph for Phase 0B.
// ===========================================================
// Manages a flat ordered list of raster passes. Auto-generates
// Vulkan sync2 image barriers between passes by tracking each
// resource's current VkImageLayout and emitting barriers when a
// pass requires a different layout.
//
// Design constraints (Phase 0B scope):
//   - Linear pass execution order only (no topological sort).
//   - Imported images only (no transient allocation).
//   - Raster passes only (compute dispatches deferred to Phase 2).
//
// Usage per frame:
//   graph.reset();
//   auto h = graph.importImage("name", image, view, fmt,
//                               VK_IMAGE_LAYOUT_UNDEFINED,
//                               VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
//   graph.addRasterPass({ .name="MeshPass",
//                         .colorTargets={h},
//                         .execute=[](VkCommandBuffer cmd, VkExtent2D ext){...} });
//   graph.execute(cmd, extent);
class RenderGraph {
public:
    RenderGraph() = default;
    ~RenderGraph() = default;

    RenderGraph(const RenderGraph&)            = delete;
    RenderGraph& operator=(const RenderGraph&) = delete;
    RenderGraph(RenderGraph&&)                 = delete;
    RenderGraph& operator=(RenderGraph&&)      = delete;

    // Import an externally-owned image (swapchain, persistent attachment).
    //   initialLayout  — layout the image is in when the frame begins
    //   finalLayout    — layout to transition to after all passes complete;
    //                    use VK_IMAGE_LAYOUT_UNDEFINED to skip the final barrier
    RGImageHandle importImage(std::string_view     name,
                              VkImage              image,
                              VkImageView          view,
                              VkFormat             format,
                              VkImageLayout        initialLayout,
                              VkImageLayout        finalLayout,
                              VkImageAspectFlags   aspect = VK_IMAGE_ASPECT_COLOR_BIT);

    // Descriptor for a single raster pass.
    struct RasterPassDesc {
        std::string  name;

        // Images written as color attachments. Cleared via clearColor.
        std::vector<RGImageHandle> colorTargets;

        // Optional depth attachment. Cleared via clearDepth.
        RGImageHandle            depthTarget;

        // Images read as shader inputs in this pass. The render graph
        // emits layout transitions to SHADER_READ_ONLY_OPTIMAL for each
        // handle before vkCmdBeginRendering. An image must be imported
        // and must NOT also appear in colorTargets / depthTarget.
        std::vector<RGImageHandle> sampledInputs;

        VkClearColorValue        clearColor = {{0.0f, 0.0f, 0.0f, 1.0f}};
        VkClearDepthStencilValue clearDepth = {0.0f, 0};

        // Load ops for color / depth. Default: CLEAR.
        // Switch to LOAD to accumulate across passes on the same attachment.
        VkAttachmentLoadOp  colorLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        VkAttachmentLoadOp  depthLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;

        // Called between vkCmdBeginRendering / vkCmdEndRendering.
        std::function<void(VkCommandBuffer, VkExtent2D)> execute;
    };

    void addRasterPass(RasterPassDesc desc);

    // Execute all registered passes in insertion order. For each pass:
    //   1. Emit pre-pass barriers for any image whose current layout
    //      differs from the layout required by the pass.
    //   2. Begin dynamic rendering, invoke execute(), end rendering.
    // After all passes, emit barriers to transition all imported images
    // to their declared finalLayout (if set and different from current).
    void execute(VkCommandBuffer cmd, VkExtent2D extent);

    // Clear all imported resources and pass nodes. Call once per frame
    // before re-building the graph.
    void reset();

private:
    struct ImageResource {
        std::string        name;
        VkImage            image         = VK_NULL_HANDLE;
        VkImageView        view          = VK_NULL_HANDLE;
        VkFormat           format        = VK_FORMAT_UNDEFINED;
        VkImageAspectFlags aspect        = VK_IMAGE_ASPECT_COLOR_BIT;
        VkImageLayout      currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkImageLayout      finalLayout   = VK_IMAGE_LAYOUT_UNDEFINED;
    };

    struct RasterPassNode {
        std::string                   name;
        std::vector<RGImageHandle>    colorTargets;
        RGImageHandle                 depthTarget;
        std::vector<RGImageHandle>    sampledInputs;
        VkClearColorValue             clearColor;
        VkClearDepthStencilValue      clearDepth;
        VkAttachmentLoadOp            colorLoadOp;
        VkAttachmentLoadOp            depthLoadOp;
        std::function<void(VkCommandBuffer, VkExtent2D)> execute;
    };

    // Derive the source pipeline stage and access mask from a layout
    // that is about to be transitioned away from.
    static VkPipelineStageFlags2 srcStageFor(VkImageLayout layout);
    static VkAccessFlags2        srcAccessFor(VkImageLayout layout);

    // Emit a single image memory barrier via vkCmdPipelineBarrier2.
    static void emitBarrier(VkCommandBuffer cmd,
                            const ImageResource& img,
                            VkImageLayout        newLayout,
                            VkPipelineStageFlags2 dstStage,
                            VkAccessFlags2        dstAccess);

    std::vector<ImageResource>  m_images;
    std::vector<RasterPassNode> m_passes;
};

} // namespace enigma::gfx
