#pragma once

#include "core/Types.h"

#include <volk.h>

struct VmaAllocation_T;
using VmaAllocation = VmaAllocation_T*;

namespace enigma::gfx {
class Allocator;
class Device;
class DescriptorAllocator;
class Pipeline;
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

    // Second-phase init: build the pipeline once the swapchain format
    // is known (step 37). Kept separate from the constructor so the
    // SSBO can land at step 36 without requiring the pipeline.
    void buildPipeline(gfx::ShaderManager& shaderManager,
                       VkDescriptorSetLayout globalSetLayout,
                       VkFormat colorAttachmentFormat);

    // Record the draw into `cmd`. Caller is responsible for vkCmdBeginRendering
    // / vkCmdEndRendering and image layout transitions.
    void record(VkCommandBuffer cmd,
                VkDescriptorSet globalSet,
                VkExtent2D extent);

    u32 bindlessSlot() const { return m_bindlessSlot; }

private:
    gfx::Device*              m_device    = nullptr;
    gfx::Allocator*           m_allocator = nullptr;

    VkBuffer                  m_vertexBuffer     = VK_NULL_HANDLE;
    VmaAllocation             m_vertexAllocation = nullptr;
    u32                       m_bindlessSlot     = 0;

    gfx::Pipeline*            m_pipeline = nullptr; // owned, but forward-declared
    bool                      m_firstRecord = true;
};

} // namespace enigma
