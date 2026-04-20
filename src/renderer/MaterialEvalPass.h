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
    // slot IDs (owned by Renderer). Call after createGBufferImages() and
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
    // visBufferSlot:      sampled-image slot from VisibilityBufferPass (32-bit vis).
    // vis64BufferSlot:    bindless storage-image slot for the 64-bit Micropoly
    //                     vis image (M3.4). The =false shader never reads this
    //                     field (the #ifdef MP_ENABLE preprocessor gate strips
    //                     it at DXC compile time), but we still push it so both
    //                     pipeline variants share one push-constant range size.
    //                     Pass UINT32_MAX when no micropoly image exists.
    // materialBufferSlot: bindless SSBO slot for the GpuMaterial[] array.
    // enableMp:           when true, bind the MP_ENABLE=true pipeline variant
    //                     (lazy-built on first call via -D MP_ENABLE=1).
    //                     When false, bind the baseline pipeline whose SPIR-V
    //                     is byte-identical to the pre-M3.4 golden (Principle 1).
    // M5 micropoly material resolution inputs. When enableMp=true the shader's
    // mpWins branch walks the cluster DAG + page cache to reconstruct the
    // hit-triangle's world-space vertices + normals. Pass UINT32_MAX for any
    // slot when micropoly is disabled — the dead-code-eliminated =false
    // pipeline variant never reads them.
    struct MpResolveInputs {
        u32 dagBufferSlot        = UINT32_MAX;
        u32 pageToSlotSlot       = UINT32_MAX;
        u32 pageCacheSlot        = UINT32_MAX;
        u32 pageFirstDagNodeSlot = UINT32_MAX;
        u32 pageSlotBytes        = 0u;
        u32 pageCount            = 0u;
        u32 dagNodeCount         = 0u;
    };

    void record(VkCommandBuffer         cmd,
                VkDescriptorSet         globalSet,
                VkExtent2D              extent,
                u32                     visBufferSlot,
                const GpuSceneBuffer&   scene,
                const GpuMeshletBuffer& meshlets,
                u32                     materialBufferSlot,
                u32                     cameraSlot,
                u32                     vis64BufferSlot,
                bool                    enableMp,
                const MpResolveInputs&  mp = {});

private:
    // Build one pipeline variant from a pre-compiled VkShaderModule. The
    // caller is responsible for compiling with the right defines (-D MP_ENABLE=1
    // for the =true variant, no extra defines for the =false variant).
    // `enableMp` is retained as a documentary tag but has no effect on the
    // pipeline creation itself — the distinction is already baked into the
    // SPIR-V by DXC at compile time.
    gfx::Pipeline* buildPipeline_(VkShaderModule module, bool enableMp) const;

    // Lazy-build the MP_ENABLE=true pipeline on first use. Devices that
    // never hit the micropoly path never pay the compile cost.
    void ensureMpEnabledPipeline_();

    void rebuildPipeline();

    gfx::Device*          m_device              = nullptr;
    // Baseline pipeline: MP_ENABLE=false. SPIR-V is byte-identical to the
    // pre-M3.4 golden (Principle 1). Built eagerly in buildPipeline().
    gfx::Pipeline*        m_pipelineMpDisabled  = nullptr;
    // MP-enabled pipeline: MP_ENABLE=true. Lazy-built the first time
    // record() is invoked with enableMp=true.
    gfx::Pipeline*        m_pipelineMpEnabled   = nullptr;
    gfx::ShaderManager*   m_shaderManager       = nullptr;
    VkDescriptorSetLayout m_globalSetLayout     = VK_NULL_HANDLE;
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

    // Throttle guard for the MP-fallback error log in record(). Set true
    // on the first frame enableMp=true is requested while m_pipelineMpEnabled
    // remains null (i.e. the lazy MP_ENABLE=1 compile failed) so the log
    // fires once per pass lifetime rather than once per frame.
    bool m_mpFallbackLogged = false;
};

} // namespace enigma
