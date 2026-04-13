#pragma once

#include "core/Types.h"

#include <volk.h>

#include <filesystem>

// Forward declare VMA handles.
struct VmaAllocator_T;
using VmaAllocator = VmaAllocator_T*;
struct VmaAllocation_T;
using VmaAllocation = VmaAllocation_T*;

namespace enigma::gfx {
class Device;
class Allocator;
class DescriptorAllocator;
class ShaderManager;
class ShaderHotReload;
} // namespace enigma::gfx

namespace enigma {

class GpuMeshletBuffer;
class GpuSceneBuffer;
class IndirectDrawBuffer;
struct GpuInstance;
struct Meshlet;

// VisibilityBufferPass
// ====================
// Visibility buffer pass. Primary path: task + mesh + fragment pipeline
// (requires VK_EXT_mesh_shader). Fallback path for Min-tier GPUs: traditional
// vertex shader reading meshlet data from SSBOs.
//
// Each pixel stores:  vis_value = (instance_id << 16) | triangle_idx_in_meshlet
//
// MaterialEvalPass reconstructs full PBR attributes from the packed ID.
// Depth is shared with the GBuffer pass (borrowed VkImage / VkImageView).
class VisibilityBufferPass {
public:
    VisibilityBufferPass(gfx::Device& device, gfx::Allocator& allocator,
                         gfx::DescriptorAllocator& descriptors);
    ~VisibilityBufferPass();

    VisibilityBufferPass(const VisibilityBufferPass&)            = delete;
    VisibilityBufferPass& operator=(const VisibilityBufferPass&) = delete;

    // Allocate the R32_UINT visibility buffer at the given extent.
    // depthFormat must match the depth image that will be passed to record().
    void allocate(VkExtent2D extent, VkFormat depthFormat);

    // Build the task + mesh + fragment pipeline.
    void buildPipeline(gfx::ShaderManager& shaderManager,
                       VkDescriptorSetLayout globalSetLayout);

    void registerHotReload(gfx::ShaderHotReload& reloader);

    // Generate the VS-fallback indirect draw buffer from CPU-side scene/meshlet data.
    // Call after scene load on Min-tier GPUs (device.supportsMeshShaders() == false).
    // Must be called before the first record() on a non-mesh-shader device.
    void buildVsFallbackDraws(const GpuSceneBuffer& scene,
                              const GpuMeshletBuffer& meshlets);

    // Record the visibility buffer draw call.
    // depthView:  borrowed VkImageView for the GBuffer depth (D32_SFLOAT).
    // depthImage: the underlying VkImage (for layout transitions).
    void record(VkCommandBuffer           cmd,
                VkDescriptorSet           globalSet,
                VkExtent2D                extent,
                VkImageView               depthView,
                VkImage                   depthImage,
                const GpuSceneBuffer&     scene,
                const GpuMeshletBuffer&   meshlets,
                const IndirectDrawBuffer& indirect,
                u32                       cameraSlot);

    // Bindless sampled-image slot for the vis buffer (read by MaterialEvalPass).
    u32       vis_buffer_slot() const { return m_vis_slot; }
    VkImage   vis_image()       const { return m_vis_image; }
    VkImageView vis_image_view() const { return m_vis_view; }

private:
    void destroyVisImage();
    void rebuildPipeline();
    void buildVsFallbackPipeline();

    gfx::Device*              m_device         = nullptr;
    gfx::Allocator*           m_allocator      = nullptr;
    gfx::DescriptorAllocator* m_descriptors    = nullptr;
    gfx::ShaderManager*       m_shaderManager  = nullptr;
    VkDescriptorSetLayout     m_globalSetLayout = VK_NULL_HANDLE;

    // Raw Vulkan mesh pipeline (Pipeline class doesn't support mesh shaders).
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline       m_pipeline       = VK_NULL_HANDLE;

    VkFormat m_depthFormat = VK_FORMAT_D32_SFLOAT;

    VkImage       m_vis_image = VK_NULL_HANDLE;
    VkImageView   m_vis_view  = VK_NULL_HANDLE;
    VmaAllocation m_vis_alloc = nullptr;
    u32           m_vis_slot  = 0;

    VkExtent2D m_extent{};

    std::filesystem::path m_taskShaderPath;
    std::filesystem::path m_meshShaderPath;

    // VS fallback (Min-tier GPUs without VK_EXT_mesh_shader).
    bool         m_useMeshShaders        = true;
    VkPipeline   m_vsFallbackPipeline    = VK_NULL_HANDLE;
    VkBuffer     m_vsFallbackDrawBuffer  = VK_NULL_HANDLE;
    VmaAllocation m_vsFallbackDrawAlloc  = nullptr;
    u32          m_vsFallbackDrawCount   = 0;
    std::filesystem::path m_vsShaderPath;
};

} // namespace enigma
