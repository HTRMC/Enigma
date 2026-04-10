#pragma once

#include "core/Types.h"

#include <volk.h>

namespace enigma::gfx {

class Device;

// Pipeline
// ========
// Thin graphics-pipeline helper for the dynamic-rendering world. Given
// a vertex + fragment shader module pair, a global descriptor set
// layout (bound at set=0), and the color attachment format, it builds
// a VkPipelineLayout + VkPipeline ready to record draws against.
//
// Key design contracts (per Architect synthesis item 4):
//   - The set=0 layout argument is REQUIRED (no null allowed). Every
//     pipeline in the engine shares the same global bindless set.
//   - Push constant range is fixed at {VERTEX|FRAGMENT, offset=0,
//     size=16} — 16 bytes for alignment friendliness; only the first
//     4 bytes carry a payload at milestone 1 (the bindless slot).
//   - Uses `VkPipelineRenderingCreateInfo` in pNext with a single
//     color attachment. No `VkRenderPass`, no `VkFramebuffer`.
//
// Second-caller design intent (Principle 6): a second pipeline with
// different shaders but the same set=0 layout is `Pipeline(device,
// otherVert, otherFrag, globalLayout, swapchainFormat)` — zero
// rewrite required.
class Pipeline {
public:
    // `depthAttachmentFormat` defaults to `VK_FORMAT_UNDEFINED`, which
    // disables depth testing and writing and leaves the
    // `VkPipelineRenderingCreateInfo::depthAttachmentFormat` unset so
    // the pipeline is legal to bind in a render pass without a depth
    // attachment. Passing a real depth format (e.g. D32_SFLOAT) turns
    // on `depthTestEnable` + `depthWriteEnable` with `COMPARE_OP_LESS`
    // and wires the matching pNext format — no separate API surface
    // for "depth-enabled" pipelines.
    //
    // `vertEntryPoint` and `fragEntryPoint` name the SPIR-V entry
    // points inside each shader module. DXC's `-spirv` output
    // PRESERVES the HLSL entry-point name (e.g. `VSMain`, `PSMain`)
    // in `OpEntryPoint`, so the pipeline's `pName` field must match
    // exactly or `vkCreateGraphicsPipelines` will fail at runtime
    // with "entry point not found". The names are captured as C
    // strings and must outlive the Pipeline constructor call (string
    // literals are fine; any dynamic storage must live at least that
    // long).
    struct CreateInfo {
        VkShaderModule        vertShader            = VK_NULL_HANDLE;
        const char*           vertEntryPoint        = "VSMain";
        VkShaderModule        fragShader            = VK_NULL_HANDLE;
        const char*           fragEntryPoint        = "PSMain";
        VkDescriptorSetLayout globalSetLayout       = VK_NULL_HANDLE;
        // Single color attachment (legacy / simple pass). Ignored when
        // colorAttachmentCount > 0.
        VkFormat              colorAttachmentFormat = VK_FORMAT_UNDEFINED;
        VkFormat              depthAttachmentFormat = VK_FORMAT_UNDEFINED;
        u32                   pushConstantSize      = 16;
        VkCompareOp           depthCompareOp        = VK_COMPARE_OP_LESS;
        VkCullModeFlagBits    cullMode              = VK_CULL_MODE_NONE;
        // MRT: populate these for multi-render-target pipelines.
        // When colorAttachmentCount > 0 these take precedence over
        // colorAttachmentFormat. Up to 8 color attachments supported.
        VkFormat              colorAttachmentFormats[8] = {};
        u32                   colorAttachmentCount      = 0;
    };

    Pipeline(Device& device, const CreateInfo& info);
    ~Pipeline();

    Pipeline(const Pipeline&)            = delete;
    Pipeline& operator=(const Pipeline&) = delete;
    Pipeline(Pipeline&&)                 = delete;
    Pipeline& operator=(Pipeline&&)      = delete;

    VkPipeline       handle() const { return m_pipeline;       }
    VkPipelineLayout layout() const { return m_pipelineLayout; }

private:
    Device*          m_device         = nullptr;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline       m_pipeline       = VK_NULL_HANDLE;
};

} // namespace enigma::gfx
