#pragma once

#include "core/Types.h"

#include <volk.h>

#include <filesystem>

namespace enigma {
struct Scene;
} // namespace enigma

namespace enigma::gfx {
class Device;
class Pipeline;
class ShaderHotReload;
class ShaderManager;
} // namespace enigma::gfx

namespace enigma {

// MeshPass
// ========
// Renders loaded glTF scene geometry using the bindless mesh pipeline.
// Each draw pushes an 80-byte push constant block containing the model
// matrix and bindless slot indices. Indexed drawing via vkCmdDrawIndexed.
//
// Falls through cleanly when no scene is set — record() is a no-op.
class MeshPass {
public:
    explicit MeshPass(gfx::Device& device);
    ~MeshPass();

    MeshPass(const MeshPass&)            = delete;
    MeshPass& operator=(const MeshPass&) = delete;
    MeshPass(MeshPass&&)                 = delete;
    MeshPass& operator=(MeshPass&&)      = delete;

    void buildPipeline(gfx::ShaderManager& shaderManager,
                       VkDescriptorSetLayout globalSetLayout,
                       VkFormat colorAttachmentFormat,
                       VkFormat depthAttachmentFormat);

    void record(VkCommandBuffer cmd,
                VkDescriptorSet globalSet,
                VkExtent2D extent,
                const Scene& scene,
                u32 cameraSlot);

    void registerHotReload(gfx::ShaderHotReload& reloader);

private:
    void rebuildPipeline();

    gfx::Device*          m_device          = nullptr;
    gfx::Pipeline*        m_pipeline        = nullptr;

    // Hot-reload state.
    gfx::ShaderManager*   m_shaderManager   = nullptr;
    VkDescriptorSetLayout m_globalSetLayout  = VK_NULL_HANDLE;
    VkFormat              m_colorFormat      = VK_FORMAT_UNDEFINED;
    VkFormat              m_depthFormat      = VK_FORMAT_UNDEFINED;
    std::filesystem::path m_shaderPath;
};

} // namespace enigma
