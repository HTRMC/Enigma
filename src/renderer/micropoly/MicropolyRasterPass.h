#pragma once

// MicropolyRasterPass.h
// ======================
// Per-frame HW rasterisation pass for the Micropoly subsystem (M3.3).
// Drives shaders/micropoly/mp_raster.{task,mesh}.hlsl via
// vkCmdDrawMeshTasksIndirectEXT, consuming the indirect-draw buffer
// produced by MicropolyCullPass (M3.2) and writing visibility into the
// 64-bit vis image owned by MicropolyPass (M3.1).
//
// The pass is a graphics pipeline with Task + Mesh + Fragment stages but
// NO color attachment — the fragment shader's only side-effect is an
// InterlockedMax on the bindless R64_UINT storage image (under reverse-Z
// the largest packed depth wins; see shaders/micropoly/mp_vis_pack.hlsl).
// vkCmdBeginRendering is used with colorAttachmentCount=0 and a null
// depth attachment so the HW rasteriser still sees a viewport/scissor.
//
// Principle 1: the pass is never constructed when MicropolyConfig::enabled
// is false, and not constructed when the device lacks VK_EXT_mesh_shader
// or VK_EXT_shader_image_atomic_int64 (supportsShaderImageInt64). On those
// devices MicropolyPass is already Disabled and this pass has no work to do.
//
// Render-graph slot (Renderer.cpp):
//   [streaming.beginFrame] -> [cluster cull dispatch] -> [RASTER (here)]
//   -> [MaterialEval merge: M3.4, not in this milestone]
//
// M3.3 scope: the pass is constructed and the draw is issued, but the
// downstream MaterialEvalPass merge is left to M3.4. Today the vis image
// is written but never read; confirming end-to-end correctness needs
// M3.4's debug overlay or a vis-image dump hook.
//
// Shader caveat (carried over from MicropolyCullPass): under DXC -Zpc -spirv
// the float4x4(v0..v3) constructor takes COLUMN vectors, so matrix SSBO
// loads use `transpose(float4x4(...))`. Don't "simplify" those lines.

#include "core/Types.h"

#include <volk.h>

#include <expected>
#include <filesystem>
#include <string>

namespace enigma::gfx {
class Allocator;
class DescriptorAllocator;
class Device;
class ShaderHotReload;
class ShaderManager;
} // namespace enigma::gfx

namespace enigma::renderer::micropoly {

enum class MicropolyRasterErrorKind {
    MeshShadersUnsupported,
    Int64ImageUnsupported,
    PipelineBuildFailed,
    InvalidVisImage,
};

struct MicropolyRasterError {
    MicropolyRasterErrorKind kind{};
    std::string              detail;
};

class MicropolyRasterPass {
public:
    // Non-throwing factory. Builds the Task+Mesh+Fragment graphics
    // pipeline. Returns an error when either required device capability
    // (mesh shader OR shaderImageInt64) is absent — the caller MUST check
    // these before constructing to honour Principle 1.
    static std::expected<MicropolyRasterPass, MicropolyRasterError> create(
        gfx::Device&              device,
        gfx::DescriptorAllocator& descriptors,
        gfx::ShaderManager&       shaderManager);

    ~MicropolyRasterPass();

    MicropolyRasterPass(const MicropolyRasterPass&)            = delete;
    MicropolyRasterPass& operator=(const MicropolyRasterPass&) = delete;
    MicropolyRasterPass(MicropolyRasterPass&& other) noexcept;
    MicropolyRasterPass& operator=(MicropolyRasterPass&& other) noexcept;

    // Per-frame inputs. See mp_raster.task.hlsl / mp_raster.mesh.hlsl push
    // blocks for the meaning of each bindless slot. All non-override fields
    // are required — UINT32_MAX on any of them produces undefined shader
    // behaviour (intentional: surface bugs early in testing rather than
    // silently skipping work).
    struct DispatchInputs {
        VkCommandBuffer cmd                          = VK_NULL_HANDLE;
        VkDescriptorSet globalSet                    = VK_NULL_HANDLE;

        // Indirect-draw buffer (from MicropolyCullPass) — the command array
        // lives at offset 16 with stride 16 bytes per command.
        VkBuffer indirectBuffer                      = VK_NULL_HANDLE;
        u32      indirectBufferBindlessIndex         = UINT32_MAX;

        // DAG + page-resident + page-bytes SSBOs, matching the push block.
        u32      dagBufferBindlessIndex              = UINT32_MAX;
        u32      pageToSlotBufferBindlessIndex       = UINT32_MAX;
        u32      pageCacheBufferBindlessIndex        = UINT32_MAX;

        // Camera + vis image.
        u32      cameraSlot                          = UINT32_MAX;
        u32      visImageBindlessIndex               = UINT32_MAX;

        // Runtime constants — dimensions of the render area (viewport) and
        // PageCache slot byte stride.
        VkExtent2D extent                            = {0u, 0u};
        u32        pageSlotBytes                     = 0u;
        u32        pageCount                         = 0u;
        u32        dagNodeCount                      = 0u;

        // Draw-command cap — this is kMpMaxIndirectDrawClusters from the
        // cull pass. vkCmdDrawMeshTasksIndirectEXT needs a "drawCount" and
        // indirect-count feature is optional; we pass maxDrawCount and let
        // the shader skip empty slots (count==0 header in front of the
        // buffer means the shader reads a draw cmd with groupCountX=0,
        // which is a legal no-op).
        //
        // Since our indirect buffer packs "count_header + per-cmd payloads",
        // and vkCmdDrawMeshTasksIndirectEXT reads a fixed array of commands
        // starting at the given offset, we use vkCmdDrawMeshTasksIndirectCountEXT
        // where the count is the header u32 at offset 0. This requires
        // VK_KHR_draw_indirect_count / Vulkan 1.2 feature
        // drawIndirectCount — already enabled across the engine.
        u32        maxClusters                       = 0u;
        // M4.4: rasterClassBuffer bindless slot — the task shader reads
        // this u32-per-drawSlot tag and bails (DispatchMesh(0,...)) for
        // clusters the dispatcher assigned to the SW compute path.
        u32        rasterClassBufferBindlessIndex    = UINT32_MAX;
        // M4.5: pageId -> firstDagNodeIdx SSBO bindless slot. Task shader
        // uses it to derive localClusterIdx for multi-cluster pages and
        // passes the derived value down via the task->mesh payload.
        u32        pageFirstDagNodeBufferBindlessIndex = UINT32_MAX;
    };

    // Record the raster draw into `cmd`. Caller is responsible for any
    // upstream barriers (cull-write -> draw-indirect + task/mesh read) and
    // for transitioning the vis image to VK_IMAGE_LAYOUT_GENERAL before
    // entry. This function:
    //   1) vkCmdBeginRendering with no color attachments + no depth.
    //   2) Binds pipeline + global descriptor set + push constants.
    //   3) vkCmdDrawMeshTasksIndirectCountEXT against the indirect buffer.
    //   4) vkCmdEndRendering.
    //   5) Emits a fragment-shader -> compute-shader barrier on the vis
    //      image so M3.4's MaterialEvalPass sees the writes.
    //
    // No-op guards: extent zero, indirectBuffer null, or maxClusters == 0
    // make this function return before any draw is emitted (but may still
    // emit begin/end rendering pairs? no — we early-return cleanly).
    void record(const DispatchInputs& inputs);

    // Register the task/mesh/fragment shaders with the hot-reload manager
    // so edits rebuild the pipeline in place. Mirrors MicropolyCullPass.
    void registerHotReload(gfx::ShaderHotReload& reloader);

    // Accessors — for tests + debug UI.
    VkPipeline       pipeline()       const { return m_pipeline; }
    VkPipelineLayout pipelineLayout() const { return m_pipelineLayout; }

private:
    MicropolyRasterPass(gfx::Device& device,
                        gfx::DescriptorAllocator& descriptors);

    // Release Vulkan objects. Idempotent.
    void destroy_();

    // Build / rebuild the graphics pipeline (task+mesh+frag). Returns
    // true on success. Called once at create() and on hot-reload events.
    bool rebuildPipeline_();

    gfx::Device*              m_device         = nullptr;
    gfx::DescriptorAllocator* m_descriptors    = nullptr;
    gfx::ShaderManager*       m_shaderManager  = nullptr;

    VkDescriptorSetLayout     m_globalSetLayout = VK_NULL_HANDLE;
    std::filesystem::path     m_taskShaderPath{};
    std::filesystem::path     m_meshShaderPath{};

    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline       m_pipeline       = VK_NULL_HANDLE;
};

const char* micropolyRasterErrorKindString(MicropolyRasterErrorKind kind);

} // namespace enigma::renderer::micropoly
