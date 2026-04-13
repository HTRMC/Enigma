#pragma once

#include "core/Types.h"

#include <volk.h>

#include <filesystem>

namespace enigma::gfx {
class Device;
class Allocator;
class DescriptorAllocator;
class Pipeline;
class ShaderManager;
class ShaderHotReload;
} // namespace enigma::gfx

namespace enigma {

class GpuMeshletBuffer;
class GpuSceneBuffer;

// MaterialEvalPass
// ================
// Full-screen compute pass that consumes the R32_UINT visibility buffer and
// reconstructs full PBR G-buffer data for every covered pixel.
//
// Per pixel:
//   1. Decode (instance_id, triangle_id) from vis buffer.
//   2. Walk the instance's meshlets to find the owning meshlet + local tri.
//   3. Load the 3 vertex attributes and compute perspective-correct barycentrics.
//   4. Sample base-color / metalRough / normal map textures.
//   5. Write albedo, world-normal, metalRough, and motion-vector storage images.
//
// Dispatch: ceil(W/8) x ceil(H/8) workgroups of 8x8 threads.
// Shader:   material_eval.comp.hlsl, entry CSMain.
//
// Slot ownership: Renderer registers and owns all G-buffer sampled/storage slots.
// MaterialEvalPass receives them via prepare() — it only owns the image handles
// (for layout transitions) and the pipeline.
class MaterialEvalPass {
public:
    explicit MaterialEvalPass(gfx::Device& device);
    ~MaterialEvalPass();

    MaterialEvalPass(const MaterialEvalPass&)            = delete;
    MaterialEvalPass& operator=(const MaterialEvalPass&) = delete;

    // Store G-buffer image handles (for layout transitions) and pre-registered
    // slot IDs (owned by Renderer). Call after GBufferPass::allocate() and
    // after every resize. No Vulkan resource allocation happens here.
    void prepare(VkExtent2D extent,
                 VkImage albedoImage,     VkImage normalImage,
                 VkImage metalRoughImage, VkImage motionVecImage,
                 VkImage depthImage,
                 u32 albedoStorageSlot,     u32 normalStorageSlot,
                 u32 metalRoughStorageSlot, u32 motionVecStorageSlot,
                 u32 depthSampledSlot);

    // Build the compute pipeline from material_eval.comp.hlsl.
    void buildPipeline(gfx::ShaderManager& shaderManager,
                       VkDescriptorSetLayout globalSetLayout);

    void registerHotReload(gfx::ShaderHotReload& reloader);

    // Record the full-screen material evaluation dispatch.
    // visBufferSlot:      sampled-image slot from VisibilityBufferPass.
    // materialBufferSlot: bindless SSBO slot for the GpuMaterial[] array.
    void record(VkCommandBuffer         cmd,
                VkDescriptorSet         globalSet,
                VkExtent2D              extent,
                u32                     visBufferSlot,
                const GpuSceneBuffer&   scene,
                const GpuMeshletBuffer& meshlets,
                u32                     materialBufferSlot,
                u32                     cameraSlot);

private:
    void rebuildPipeline();

    gfx::Device*          m_device         = nullptr;
    gfx::Pipeline*        m_pipeline       = nullptr;
    gfx::ShaderManager*   m_shaderManager  = nullptr;
    VkDescriptorSetLayout m_globalSetLayout = VK_NULL_HANDLE;
    std::filesystem::path m_shaderPath;

    // Pre-registered slot IDs (owned by Renderer, not this pass).
    u32 m_albedo_slot     = 0;
    u32 m_normal_slot     = 0;
    u32 m_metalRough_slot = 0;
    u32 m_motionVec_slot  = 0;
    u32 m_depth_slot      = 0;

    // Borrowed VkImage handles for layout transitions in record().
    VkImage m_albedo_image     = VK_NULL_HANDLE;
    VkImage m_normal_image     = VK_NULL_HANDLE;
    VkImage m_metalRough_image = VK_NULL_HANDLE;
    VkImage m_motionVec_image  = VK_NULL_HANDLE;
    VkImage m_depth_image      = VK_NULL_HANDLE;

    VkExtent2D m_extent{};
};

} // namespace enigma
