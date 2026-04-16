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
    //
    // Two-pass CDLOD support: instanceOffset/instanceCount constrain
    // `findInstanceAndLocal` to a contiguous range of GpuInstance entries, and
    // meshletOffset/meshletCount control which meshlets in the global meshlet
    // buffer this dispatch processes. Pass (0, scene.instance_count(), 0,
    // meshlets.total_meshlet_count()) for a single-batch cull covering the
    // whole scene.
    void record(VkCommandBuffer       cmd,
                VkDescriptorSet       globalSet,
                const GpuSceneBuffer& scene,
                const GpuMeshletBuffer& meshlets,
                const IndirectDrawBuffer& indirect,
                u32 cameraSlot,
                u32 instanceOffset,
                u32 instanceCount,
                u32 meshletOffset,
                u32 meshletCount);

private:
    void rebuildPipeline();

    gfx::Device*          m_device         = nullptr;
    gfx::Pipeline*        m_pipeline       = nullptr;
    gfx::ShaderManager*   m_shaderManager  = nullptr;
    VkDescriptorSetLayout m_globalSetLayout = VK_NULL_HANDLE;
    std::filesystem::path m_shaderPath;
};

} // namespace enigma
