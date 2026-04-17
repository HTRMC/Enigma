#pragma once

#include "core/Types.h"

#include <volk.h>

#include <filesystem>

struct VmaAllocation_T;
using VmaAllocation = VmaAllocation_T*;

namespace enigma::gfx {
class Device;
class Allocator;
class DescriptorAllocator;
class Pipeline;
class ShaderManager;
class ShaderHotReload;
} // namespace enigma::gfx

namespace enigma {

// SMAAPass
// ========
// Morphological anti-aliasing in three raster sub-passes:
//
//  1. Edge detection  (smaa_edge.hlsl)
//     Luma-based; output R8G8_UNORM (r=horizontal boundary, g=vertical).
//
//  2. Blending weights (smaa_blend.hlsl)
//     Searches along each edge to find its extent; triangle weight function.
//     Output R8G8B8A8_UNORM (r=vertical-smooth weight, g=horizontal-smooth).
//
//  3. Neighbourhood blend (smaa_neighborhood.hlsl)
//     Samples ±0.5 texel perpendicular to the edge and lerps by weight.
//     Output: same format as the LDR intermediate / swapchain.
//
// The three sub-passes are invoked from separate RenderGraph raster passes
// with the correct resource dependencies. Each sub-pass is a fullscreen
// triangle with no depth attachment and no blending.
//
// Intermediate textures (edge + weight) are allocated at render resolution
// by allocate(). Call free() before resizing, then allocate() again.
class SMAAPass {
public:
    SMAAPass(gfx::Device& device, gfx::Allocator& allocator);
    ~SMAAPass();

    SMAAPass(const SMAAPass&)            = delete;
    SMAAPass& operator=(const SMAAPass&) = delete;

    // Allocate the two intermediate textures at the given resolution.
    // Must be called before buildPipelines(). If already allocated, free() first.
    void allocate(VkExtent2D extent, gfx::DescriptorAllocator& descriptorAllocator);
    void free   (gfx::DescriptorAllocator& descriptorAllocator);

    // Upload the precomputed AreaTex + SearchTex lookup textures that the
    // reference SMAA blend pass consumes. Records on `cmd`; caller must
    // submit + wait idle, then call releaseLookupUploadStaging() to free the
    // host-visible staging buffers. Textures live for the lifetime of the
    // pass (not per-resize).
    void uploadLookupTextures(VkCommandBuffer cmd,
                              gfx::DescriptorAllocator& descriptorAllocator);
    void releaseLookupUploadStaging();

    // Build the three fullscreen pipelines.
    // ldrFormat: format of both the input LDR intermediate and the final output
    //            (swapchain format). The edge and weight textures have fixed formats.
    void buildPipelines(gfx::ShaderManager&   shaderManager,
                        VkDescriptorSetLayout globalSetLayout,
                        VkFormat              ldrFormat);

    void registerHotReload(gfx::ShaderHotReload& reloader);

    // Called from inside RenderGraph execute lambdas (between BeginRendering/EndRendering).

    // Pass 1: edge detection. Reads ldrSampledSlot, writes to edgeView().
    void recordEdge(VkCommandBuffer cmd,
                    VkDescriptorSet globalSet,
                    VkExtent2D      extent,
                    u32             ldrSampledSlot,
                    u32             samplerSlot);

    // Pass 2: blending weights. Reads edgeSampledSlot(), writes to weightView().
    void recordBlend(VkCommandBuffer cmd,
                     VkDescriptorSet globalSet,
                     VkExtent2D      extent,
                     u32             samplerSlot);

    // Pass 3: neighbourhood blend. Reads ldrSampledSlot + weightSampledSlot(),
    //         writes to the swapchain attachment (set up by the render graph).
    void recordNeighborhood(VkCommandBuffer cmd,
                            VkDescriptorSet globalSet,
                            VkExtent2D      extent,
                            u32             ldrSampledSlot,
                            u32             samplerSlot);

    // Views and bindless slots used as render graph resource handles / shader inputs.
    VkImage     edgeImage()         const { return m_edgeImage; }
    VkImage     weightImage()       const { return m_weightImage; }
    VkImageView edgeView()          const { return m_edgeView; }
    VkImageView weightView()        const { return m_weightView; }
    u32         edgeSampledSlot()   const { return m_edgeSampledSlot; }
    u32         weightSampledSlot() const { return m_weightSampledSlot; }

    bool isAllocated() const { return m_edgeImage != VK_NULL_HANDLE; }

private:
    void rebuildPipelines();

    gfx::Device*    m_device    = nullptr;
    gfx::Allocator* m_allocator = nullptr;

    // Three fullscreen pipelines (one per SMAA sub-pass).
    gfx::Pipeline*  m_edgePipeline   = nullptr;
    gfx::Pipeline*  m_blendPipeline  = nullptr;
    gfx::Pipeline*  m_neighborPipeline = nullptr;

    // Intermediate textures owned by this pass.
    VkImage         m_edgeImage       = VK_NULL_HANDLE; // R8G8_UNORM
    VkImageView     m_edgeView        = VK_NULL_HANDLE;
    VmaAllocation   m_edgeAlloc       = nullptr;
    u32             m_edgeSampledSlot = UINT32_MAX;

    VkImage         m_weightImage       = VK_NULL_HANDLE; // RGBA8_UNORM
    VkImageView     m_weightView        = VK_NULL_HANDLE;
    VmaAllocation   m_weightAlloc       = nullptr;
    u32             m_weightSampledSlot = UINT32_MAX;

    // Precomputed Jimenez SMAA lookup textures (uploaded once at init).
    VkImage         m_areaTexImage       = VK_NULL_HANDLE; // R8G8_UNORM, 160×560
    VkImageView     m_areaTexView        = VK_NULL_HANDLE;
    VmaAllocation   m_areaTexAlloc       = nullptr;
    u32             m_areaTexSampledSlot = UINT32_MAX;

    VkImage         m_searchTexImage       = VK_NULL_HANDLE; // R8_UNORM, 64×16
    VkImageView     m_searchTexView        = VK_NULL_HANDLE;
    VmaAllocation   m_searchTexAlloc       = nullptr;
    u32             m_searchTexSampledSlot = UINT32_MAX;

    // Host-visible staging buffers for the lookup textures, kept alive
    // until releaseLookupUploadStaging() is called (post-submit, post-idle).
    VkBuffer        m_areaTexStaging        = VK_NULL_HANDLE;
    VmaAllocation   m_areaTexStagingAlloc   = nullptr;
    VkBuffer        m_searchTexStaging      = VK_NULL_HANDLE;
    VmaAllocation   m_searchTexStagingAlloc = nullptr;

    VkExtent2D      m_extent{};

    // Cached for hot-reload.
    gfx::ShaderManager*   m_shaderManager  = nullptr;
    VkDescriptorSetLayout m_globalSetLayout = VK_NULL_HANDLE;
    VkFormat              m_ldrFormat       = VK_FORMAT_UNDEFINED;

    std::filesystem::path m_edgePath;
    std::filesystem::path m_blendPath;
    std::filesystem::path m_neighborPath;
};

} // namespace enigma
