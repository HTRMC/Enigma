#include "gfx/Pipeline.h"

#include "core/Assert.h"
#include "gfx/Device.h"

#include <array>

namespace enigma::gfx {

Pipeline::Pipeline(Device& device,
                   VkShaderModule vertShader,
                   VkShaderModule fragShader,
                   VkDescriptorSetLayout globalSetLayout,
                   VkFormat colorAttachmentFormat)
    : m_device(&device) {
    ENIGMA_ASSERT(vertShader != VK_NULL_HANDLE);
    ENIGMA_ASSERT(fragShader != VK_NULL_HANDLE);
    ENIGMA_ASSERT(globalSetLayout != VK_NULL_HANDLE);

    // -------------------------------------------------------------------
    // Pipeline layout: set=0 is the global bindless set, push constant
    // range is 16 bytes spanning vertex + fragment stages.
    // -------------------------------------------------------------------
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushRange.offset     = 0;
    pushRange.size       = 16;

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount         = 1;
    layoutInfo.pSetLayouts            = &globalSetLayout;
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
            vertShader,
            "main",
            nullptr,
        },
        {
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            nullptr, 0,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            fragShader,
            "main",
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
    rasterizer.cullMode    = VK_CULL_MODE_NONE; // triangle is 1-sided; no cull
    rasterizer.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth   = 1.0f;

    // -------------------------------------------------------------------
    // Multisample: 1 sample (no MSAA).
    // -------------------------------------------------------------------
    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // -------------------------------------------------------------------
    // Color blend: none (opaque overwrite on a single attachment).
    // -------------------------------------------------------------------
    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.blendEnable    = VK_FALSE;
    colorBlendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo colorBlend{};
    colorBlend.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments    = &colorBlendAttachment;

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
    // Dynamic rendering: VkPipelineRenderingCreateInfo in pNext. No
    // VkRenderPass anywhere.
    // -------------------------------------------------------------------
    VkPipelineRenderingCreateInfo renderingInfo{};
    renderingInfo.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingInfo.colorAttachmentCount    = 1;
    renderingInfo.pColorAttachmentFormats = &colorAttachmentFormat;

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
