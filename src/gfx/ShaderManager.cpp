#include "gfx/ShaderManager.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "gfx/Device.h"

#include <shaderc/shaderc.hpp>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace enigma::gfx {

namespace {

shaderc_shader_kind toShaderKind(ShaderManager::Stage stage) {
    switch (stage) {
        case ShaderManager::Stage::Vertex:   return shaderc_vertex_shader;
        case ShaderManager::Stage::Fragment: return shaderc_fragment_shader;
    }
    ENIGMA_ASSERT(false && "unknown shader stage");
    return shaderc_vertex_shader;
}

std::string readTextFile(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::in | std::ios::binary);
    if (!in) {
        ENIGMA_LOG_ERROR("[shader] failed to open: {}", path.string());
        ENIGMA_ASSERT(false && "shader source file not found");
        return {};
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

} // namespace

ShaderManager::ShaderManager(Device& device)
    : m_device(&device),
      m_compiler(new shaderc::Compiler()),
      m_options(new shaderc::CompileOptions()) {

    m_options->SetTargetEnvironment(shaderc_target_env_vulkan,
                                    shaderc_env_version_vulkan_1_3);
    m_options->SetTargetSpirv(shaderc_spirv_version_1_6);
#if ENIGMA_DEBUG
    m_options->SetOptimizationLevel(shaderc_optimization_level_zero);
    m_options->SetGenerateDebugInfo();
#else
    m_options->SetOptimizationLevel(shaderc_optimization_level_performance);
#endif
}

ShaderManager::~ShaderManager() {
    delete m_options;
    delete m_compiler;
}

VkShaderModule ShaderManager::compile(const std::filesystem::path& absolutePath, Stage stage) {
    const std::string source = readTextFile(absolutePath);
    ENIGMA_ASSERT(!source.empty());

    const std::string sourceName = absolutePath.filename().string();
    const shaderc_shader_kind kind = toShaderKind(stage);

    shaderc::SpvCompilationResult result =
        m_compiler->CompileGlslToSpv(source, kind, sourceName.c_str(), *m_options);

    if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
        ENIGMA_LOG_ERROR("[shader] compile failed: {}\n{}",
                         sourceName, result.GetErrorMessage());
        ENIGMA_ASSERT(false && "shader compilation failed");
        return VK_NULL_HANDLE;
    }

    const std::vector<u32> spirv(result.cbegin(), result.cend());

    VkShaderModuleCreateInfo info{};
    info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = spirv.size() * sizeof(u32);
    info.pCode    = spirv.data();

    VkShaderModule module = VK_NULL_HANDLE;
    ENIGMA_VK_CHECK(vkCreateShaderModule(m_device->logical(), &info, nullptr, &module));

    ENIGMA_LOG_INFO("[shader] compiled {} ({} words)", sourceName, spirv.size());
    return module;
}

} // namespace enigma::gfx
