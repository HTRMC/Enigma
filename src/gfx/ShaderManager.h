#pragma once

#include "core/Types.h"

#include <volk.h>

#include <filesystem>
#include <string>

namespace shaderc { class Compiler; class CompileOptions; }

// Forward declarations for DXC's COM interfaces. The full dxcapi.h
// pulls in Windows.h and the whole COM machinery, so we keep it out
// of this header and include it only in ShaderManager.cpp.
struct IDxcUtils;
struct IDxcCompiler3;

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

    // Read a shader source file from disk, compile to SPIR-V, and
    // create a VkShaderModule. Caller owns the returned handle and
    // must `vkDestroyShaderModule` it (typically once the pipeline
    // that consumes it has been created and the source modules are
    // no longer needed, per the standard Vulkan idiom).
    //
    // `entryPoint` defaults to "main" which matches the GLSL idiom
    // (every GLSL shader has a single `void main()`). HLSL shaders
    // name their entry points (`VSMain`, `PSMain`, etc.) so the HLSL
    // path requires an explicit override.
    VkShaderModule compile(const std::filesystem::path& absolutePath,
                           Stage stage,
                           const std::string& entryPoint = "main");

    // Non-fatal variant of compile(). Returns VK_NULL_HANDLE on any
    // failure (missing file, syntax error, vkCreateShaderModule
    // error) instead of asserting. Used by hot reload so a typo in a
    // shader source does not crash the engine — the caller keeps its
    // previous module/pipeline intact and the developer fixes and
    // saves again.
    VkShaderModule tryCompile(const std::filesystem::path& absolutePath,
                              Stage stage,
                              const std::string& entryPoint = "main");

private:
    // Non-fatal HLSL compile path via DXC. Dispatched to from
    // tryCompile() when the file extension is `.hlsl`. Kept as a
    // private member function (rather than a free helper) so it
    // can reach `m_dxcUtils` / `m_dxcCompiler` without plumbing.
    VkShaderModule tryCompileHLSL(const std::filesystem::path& absolutePath,
                                  Stage stage,
                                  const std::string& entryPoint);

    // Non-fatal GLSL compile path via shaderc. Used for `.vert` /
    // `.frag` / `.glsl` extensions. Will be removed once the HLSL
    // migration is complete.
    VkShaderModule tryCompileGLSL(const std::filesystem::path& absolutePath,
                                  Stage stage,
                                  const std::string& entryPoint);

    Device*                  m_device      = nullptr;
    shaderc::Compiler*       m_compiler    = nullptr;
    shaderc::CompileOptions* m_options     = nullptr;
    IDxcUtils*               m_dxcUtils    = nullptr;
    IDxcCompiler3*           m_dxcCompiler = nullptr;
};

} // namespace enigma::gfx
