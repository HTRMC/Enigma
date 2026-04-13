#include "gfx/Pipeline.h"

#include "core/Assert.h"
#include "gfx/Device.h"

#include <array>

namespace enigma::gfx {

Pipeline::Pipeline(Device& device, const CreateInfo& info)
    : m_device(&device) {
    ENIGMA_ASSERT(info.globalSetLayout != VK_NULL_HANDLE);

    // -------------------------------------------------------------------
    // Compute pipeline fast path.
    // -------------------------------------------------------------------
    if (info.computeShader != VK_NULL_HANDLE) {
        ENIGMA_ASSERT(info.computeEntryPoint != nullptr);

        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushRange.offset     = 0;
        pushRange.size       = info.pushConstantSize;

        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.setLayoutCount         = 1;
        layoutInfo.pSetLayouts            = &info.globalSetLayout;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges    = &pushRange;

        ENIGMA_VK_CHECK(vkCreatePipelineLayout(m_device->logical(), &layoutInfo, nullptr, &m_pipelineLayout));

        VkPipelineShaderStageCreateInfo stageInfo{};
        stageInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stageInfo.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        stageInfo.module = info.computeShader;
        stageInfo.pName  = info.computeEntryPoint;

        VkComputePipelineCreateInfo pipelineCI{};
        pipelineCI.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineCI.stage  = stageInfo;
        pipelineCI.layout = m_pipelineLayout;

        ENIGMA_VK_CHECK(vkCreateComputePipelines(m_device->logical(), VK_NULL_HANDLE,
                                                  1, &pipelineCI, nullptr, &m_pipeline));
        return;
    }

    // -------------------------------------------------------------------
    // Graphics pipeline path.
    // -------------------------------------------------------------------
    ENIGMA_ASSERT(info.vertShader != VK_NULL_HANDLE);
    ENIGMA_ASSERT(info.fragShader != VK_NULL_HANDLE);
    ENIGMA_ASSERT(info.vertEntryPoint != nullptr);
    ENIGMA_ASSERT(info.fragEntryPoint != nullptr);

    // -------------------------------------------------------------------
    // Pipeline layout: set=0 is the global bindless set, push constant
    // range spans vertex + fragment stages.
    // -------------------------------------------------------------------
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushRange.offset     = 0;
    pushRange.size       = info.pushConstantSize;

    const VkDescriptorSetLayout setLayouts[2] = {
        info.globalSetLayout,
        info.additionalSetLayout, // may be VK_NULL_HANDLE
    };
    const u32 setLayoutCount = (info.additionalSetLayout != VK_NULL_HANDLE) ? 2u : 1u;

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount         = setLayoutCount;
    layoutInfo.pSetLayouts            = setLayouts;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges    = &pushRange;

    ENIGMA_VK_CHECK(vkCreatePipelineLayout(m_device->logical(), &layoutInfo, nullptr, &m_pipelineLayout));

    // -------------------------------------------------------------------
    // Shader stages.
    // -------------------------------------------------------------------
    const std::array<VkPipelineShaderStageCreateInfo, 2> stages = {{
        {
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            nullptr, 0,
            VK_SHADER_STAGE_VERTEX_BIT,
            info.vertShader,
            info.vertEntryPoint,
            nullptr,
        },
        {
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            nullptr, 0,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            info.fragShader,
            info.fragEntryPoint,
            nullptr,
        },
    }};

    // -------------------------------------------------------------------
    // Vertex input state: empty. Vertex positions come from the bindless
    // SSBO, not a vertex buffer.
    // -------------------------------------------------------------------
    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    // -------------------------------------------------------------------
    // Input assembly: triangle list.
    // -------------------------------------------------------------------
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    // -------------------------------------------------------------------
    // Viewport / scissor: dynamic (set per-frame so swapchain resize is
    // a no-op for pipeline rebuild).
    // -------------------------------------------------------------------
    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount  = 1;

    // -------------------------------------------------------------------
    // Rasterizer: default solid fill, back-face cull, CCW front.
    // -------------------------------------------------------------------
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode    = info.cullMode;
    rasterizer.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth   = 1.0f;

    // -------------------------------------------------------------------
    // Multisample: 1 sample (no MSAA).
    // -------------------------------------------------------------------
    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // -------------------------------------------------------------------
    // Color blend: one state per target. Alpha-blend if requested, else opaque.
    // -------------------------------------------------------------------
    const u32 numColorTargets = info.colorAttachmentCount > 0 ? info.colorAttachmentCount : 1;
    std::array<VkPipelineColorBlendAttachmentState, 8> colorBlendAttachments{};
    for (u32 i = 0; i < numColorTargets; ++i) {
        colorBlendAttachments[i].colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        if (info.blendEnable) {
            colorBlendAttachments[i].blendEnable         = VK_TRUE;
            colorBlendAttachments[i].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            colorBlendAttachments[i].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            colorBlendAttachments[i].colorBlendOp        = VK_BLEND_OP_ADD;
            colorBlendAttachments[i].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            colorBlendAttachments[i].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            colorBlendAttachments[i].alphaBlendOp        = VK_BLEND_OP_ADD;
        } else {
            colorBlendAttachments[i].blendEnable = VK_FALSE;
        }
    }

    VkPipelineColorBlendStateCreateInfo colorBlend{};
    colorBlend.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.attachmentCount = numColorTargets;
    colorBlend.pAttachments    = colorBlendAttachments.data();

    // -------------------------------------------------------------------
    // Dynamic states: viewport + scissor (set each frame).
    // -------------------------------------------------------------------
    const std::array<VkDynamicState, 2> dynamicStates = {{
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    }};

    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<u32>(dynamicStates.size());
    dynamicState.pDynamicStates    = dynamicStates.data();

    // -------------------------------------------------------------------
    // Depth/stencil state: enabled when the caller supplied a real
    // depth format. The struct is always populated so we can pass it
    // to `pDepthStencilState` unconditionally — the test/write flags
    // are what gate depth behavior, not the pointer being non-null.
    // -------------------------------------------------------------------
    const bool depthEnabled = info.depthAttachmentFormat != VK_FORMAT_UNDEFINED;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType                 = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable       = depthEnabled ? VK_TRUE : VK_FALSE;
    depthStencil.depthWriteEnable      = (depthEnabled && info.depthWriteEnable) ? VK_TRUE : VK_FALSE;
    depthStencil.depthCompareOp        = info.depthCompareOp;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable     = VK_FALSE;

    // -------------------------------------------------------------------
    // Dynamic rendering: VkPipelineRenderingCreateInfo in pNext. No
    // VkRenderPass anywhere. Supports single-attachment (legacy) and
    // multi-render-target (MRT) via colorAttachmentCount > 0.
    // -------------------------------------------------------------------
    const VkFormat* pColorFormats = info.colorAttachmentCount > 0
        ? info.colorAttachmentFormats
        : &info.colorAttachmentFormat;

    VkPipelineRenderingCreateInfo renderingInfo{};
    renderingInfo.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingInfo.colorAttachmentCount    = numColorTargets;
    renderingInfo.pColorAttachmentFormats = pColorFormats;
    renderingInfo.depthAttachmentFormat   = info.depthAttachmentFormat; // UNDEFINED = no depth

    // -------------------------------------------------------------------
    // Assemble the graphics pipeline.
    // -------------------------------------------------------------------
    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.pNext               = &renderingInfo;
    pipelineInfo.stageCount          = static_cast<u32>(stages.size());
    pipelineInfo.pStages             = stages.data();
    pipelineInfo.pVertexInputState   = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState      = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState   = &multisample;
    pipelineInfo.pDepthStencilState  = &depthStencil;
    pipelineInfo.pColorBlendState    = &colorBlend;
    pipelineInfo.pDynamicState       = &dynamicState;
    pipelineInfo.layout              = m_pipelineLayout;
    pipelineInfo.renderPass          = VK_NULL_HANDLE; // dynamic rendering
    pipelineInfo.subpass             = 0;

    ENIGMA_VK_CHECK(vkCreateGraphicsPipelines(m_device->logical(), VK_NULL_HANDLE,
                                              1, &pipelineInfo, nullptr, &m_pipeline));
}

Pipeline::~Pipeline() {
    if (m_device == nullptr) return;
    VkDevice dev = m_device->logical();
    if (m_pipeline       != VK_NULL_HANDLE) vkDestroyPipeline(dev, m_pipeline, nullptr);
    if (m_pipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(dev, m_pipelineLayout, nullptr);
}

} // namespace enigma::gfx
