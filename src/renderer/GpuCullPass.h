#pragma once

#include "core/Types.h"

#include <volk.h>

#include <filesystem>

namespace enigma::gfx {
class Device;
class Pipeline;
class ShaderManager;
class ShaderHotReload;
} // namespace enigma::gfx

namespace enigma {

class GpuMeshletBuffer;
class GpuSceneBuffer;
class IndirectDrawBuffer;

// GpuCullPass
// ===========
// GPU-driven frustum culling compute pass. One thread per meshlet globally.
// Surviving meshlets atomically append a DrawMeshTasksIndirectCommandEXT to
// the commands buffer and store their global meshlet index in the surviving
// IDs buffer. The count buffer tracks how many survived.
//
// Dispatch: ceil(totalMeshlets / 64) workgroups of 64 threads each.
// Shader: gpu_cull.comp.hlsl, entry CSMain.
class GpuCullPass {
public:
    explicit GpuCullPass(gfx::Device& device);
    ~GpuCullPass();

    GpuCullPass(const GpuCullPass&)            = delete;
    GpuCullPass& operator=(const GpuCullPass&) = delete;

    void buildPipeline(gfx::ShaderManager& shaderManager,
                       VkDescriptorSetLayout globalSetLayout);

    void registerHotReload(gfx::ShaderHotReload& reloader);

    // Record the cull dispatch. Caller must have called IndirectDrawBuffer::reset_count()
    // beforehand. After this call, insert a compute->draw-indirect barrier.
    void record(VkCommandBuffer       cmd,
                VkDescriptorSet       globalSet,
                const GpuSceneBuffer& scene,
                const GpuMeshletBuffer& meshlets,
                const IndirectDrawBuffer& indirect,
                u32 cameraSlot);

private:
    void rebuildPipeline();

    gfx::Device*          m_device         = nullptr;
    gfx::Pipeline*        m_pipeline       = nullptr;
    gfx::ShaderManager*   m_shaderManager  = nullptr;
    VkDescriptorSetLayout m_globalSetLayout = VK_NULL_HANDLE;
    std::filesystem::path m_shaderPath;
};

} // namespace enigma
