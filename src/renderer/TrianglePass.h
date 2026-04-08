#pragma once

#include "core/Types.h"

#include <volk.h>

#include <filesystem>

struct VmaAllocation_T;
using VmaAllocation = VmaAllocation_T*;

namespace enigma::gfx {
class Allocator;
class Device;
class DescriptorAllocator;
class Pipeline;
class ShaderHotReload;
class ShaderManager;
}

namespace enigma {

// TrianglePass
// ============
// The one and only render pass at milestone 1. Owns:
//   - a host-visible SSBO containing 3 vec4 positions
//   - a bindless slot index (binding 2 of the global set)
//   - the graphics pipeline + layout for the triangle
//
// record() writes the draw into a pre-begun command buffer using
// dynamic rendering. Layout transitions for the color attachment are
// the Renderer's responsibility, not this pass's.
class TrianglePass {
public:
    TrianglePass(gfx::Device& device,
                 gfx::Allocator& allocator,
                 gfx::DescriptorAllocator& descriptorAllocator);
    ~TrianglePass();

    TrianglePass(const TrianglePass&)            = delete;
    TrianglePass& operator=(const TrianglePass&) = delete;
    TrianglePass(TrianglePass&&)                 = delete;
    TrianglePass& operator=(TrianglePass&&)      = delete;

    // Second-phase init: build the pipeline once the swapchain color
    // and depth formats are known (step 37). Kept separate from the
    // constructor so the SSBO can land at step 36 without requiring
    // the pipeline. `depthAttachmentFormat` flows straight into the
    // Pipeline; passing `VK_FORMAT_UNDEFINED` disables depth testing.
    void buildPipeline(gfx::ShaderManager& shaderManager,
                       VkDescriptorSetLayout globalSetLayout,
                       VkFormat colorAttachmentFormat,
                       VkFormat depthAttachmentFormat);

    // Record the draw into `cmd`. Caller is responsible for vkCmdBeginRendering
    // / vkCmdEndRendering and image layout transitions.
    void record(VkCommandBuffer cmd,
                VkDescriptorSet globalSet,
                VkExtent2D extent);

    // Register this pass's vertex + fragment shader files with the
    // given hot-reload watcher. Must be called after buildPipeline().
    // On a change detected during `ShaderHotReload::poll()` the
    // pipeline is safely rebuilt via `rebuildPipeline()`.
    void registerHotReload(gfx::ShaderHotReload& reloader);

    u32 bindlessSlot() const { return m_bindlessSlot; }

private:
    // Internal rebuild path used by the hot-reload callback. Waits
    // for the device to idle, tries to recompile both shaders, and —
    // on success — swaps in a new pipeline. Compile failures leave
    // the previous pipeline intact and emit an error log; the frame
    // loop continues unaffected.
    void rebuildPipeline();

    gfx::Device*              m_device    = nullptr;
    gfx::Allocator*           m_allocator = nullptr;

    VkBuffer                  m_vertexBuffer     = VK_NULL_HANDLE;
    VmaAllocation             m_vertexAllocation = nullptr;
    u32                       m_bindlessSlot     = 0;

    // Procedural checkerboard texture + default sampler. Both land in
    // the global bindless set and are addressed per-draw via push
    // constants — no per-material descriptor set needed.
    VkImage                   m_texImage      = VK_NULL_HANDLE;
    VmaAllocation             m_texAllocation = nullptr;
    VkImageView               m_texView       = VK_NULL_HANDLE;
    VkSampler                 m_sampler       = VK_NULL_HANDLE;
    u32                       m_textureSlot   = 0;
    u32                       m_samplerSlot   = 0;

    gfx::Pipeline*            m_pipeline = nullptr; // owned, but forward-declared
    bool                      m_firstRecord = true;

    // Hot-reload state captured from the initial buildPipeline() call.
    gfx::ShaderManager*       m_shaderManager   = nullptr;
    VkDescriptorSetLayout     m_globalSetLayout = VK_NULL_HANDLE;
    VkFormat                  m_colorFormat     = VK_FORMAT_UNDEFINED;
    VkFormat                  m_depthFormat     = VK_FORMAT_UNDEFINED;
    std::filesystem::path     m_vertPath;
    std::filesystem::path     m_fragPath;
};

} // namespace enigma
