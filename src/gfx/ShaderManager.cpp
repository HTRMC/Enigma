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

// Non-asserting source reader. Returns an empty string on any error
// so the caller (tryCompile) can decide whether failure is fatal or
// tolerable. Hot reload wants tolerable; the initial load wraps this
// with an assert via compile() below.
std::string readTextFileSafe(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::in | std::ios::binary);
    if (!in) {
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

VkShaderModule ShaderManager::tryCompile(const std::filesystem::path& absolutePath, Stage stage) {
    const std::string source = readTextFileSafe(absolutePath);
    if (source.empty()) {
        ENIGMA_LOG_ERROR("[shader] failed to open or empty source: {}", absolutePath.string());
        return VK_NULL_HANDLE;
    }

    const std::string sourceName = absolutePath.filename().string();
    const shaderc_shader_kind kind = toShaderKind(stage);

    shaderc::SpvCompilationResult result =
        m_compiler->CompileGlslToSpv(source, kind, sourceName.c_str(), *m_options);

    if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
        ENIGMA_LOG_ERROR("[shader] compile failed: {}\n{}",
                         sourceName, result.GetErrorMessage());
        return VK_NULL_HANDLE;
    }

    const std::vector<u32> spirv(result.cbegin(), result.cend());

    VkShaderModuleCreateInfo info{};
    info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = spirv.size() * sizeof(u32);
    info.pCode    = spirv.data();

    VkShaderModule module = VK_NULL_HANDLE;
    const VkResult vr = vkCreateShaderModule(m_device->logical(), &info, nullptr, &module);
    if (vr != VK_SUCCESS) {
        ENIGMA_LOG_ERROR("[shader] vkCreateShaderModule failed: {} (VkResult={})",
                         sourceName, static_cast<int>(vr));
        return VK_NULL_HANDLE;
    }

    ENIGMA_LOG_INFO("[shader] compiled {} ({} words)", sourceName, spirv.size());
    return module;
}

VkShaderModule ShaderManager::compile(const std::filesystem::path& absolutePath, Stage stage) {
    VkShaderModule module = tryCompile(absolutePath, stage);
    ENIGMA_ASSERT(module != VK_NULL_HANDLE && "shader compilation failed (initial load)");
    return module;
}

} // namespace enigma::gfx
