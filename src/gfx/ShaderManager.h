#pragma once

#include "core/Types.h"

#include <volk.h>

#include <filesystem>
#include <string>

namespace shaderc { class Compiler; class CompileOptions; }

namespace enigma::gfx {

class Device;

// ShaderManager
// =============
// Owns the shaderc compiler used to turn GLSL source files into
// `VkShaderModule` handles at runtime. The plan rejected offline
// glslc + SPIR-V build artifacts in favor of runtime compilation so
// future hot-reload can be added without a build-system change.
//
// Stages supported: vertex, fragment. Compute and other stages can be
// added by extending the `Stage` enum without touching callers.
//
// Second-caller design intent (Principle 6): any new shader pair
// (second pipeline, compute dispatch, post-process effect) is a
// `compile()` call — no header edit, no CMake touch.
class ShaderManager {
public:
    enum class Stage {
        Vertex,
        Fragment,
    };

    explicit ShaderManager(Device& device);
    ~ShaderManager();

    ShaderManager(const ShaderManager&)            = delete;
    ShaderManager& operator=(const ShaderManager&) = delete;
    ShaderManager(ShaderManager&&)                 = delete;
    ShaderManager& operator=(ShaderManager&&)      = delete;

    // Read a GLSL source file from disk, compile to SPIR-V via shaderc,
    // and create a VkShaderModule. Caller owns the returned handle and
    // must `vkDestroyShaderModule` it (typically once the pipeline that
    // consumes it has been created and the source modules are no longer
    // needed, per the standard Vulkan idiom).
    VkShaderModule compile(const std::filesystem::path& absolutePath, Stage stage);

private:
    Device*                  m_device   = nullptr;
    shaderc::Compiler*       m_compiler = nullptr;
    shaderc::CompileOptions* m_options  = nullptr;
};

} // namespace enigma::gfx
