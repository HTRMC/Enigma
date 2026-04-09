#pragma once

#include "core/Types.h"

#include <volk.h>

#include <filesystem>
#include <string>

// Forward declarations for DXC's COM interfaces. The full dxcapi.h
// pulls in Windows.h and the whole COM machinery, so we keep it out
// of this header and include it only in ShaderManager.cpp.
struct IDxcUtils;
struct IDxcCompiler3;
struct IDxcIncludeHandler;

namespace enigma::gfx {

class Device;

// ShaderManager
// =============
// Owns DXC instances used to turn HLSL source files into
// `VkShaderModule` handles at runtime. Runtime compilation (rather
// than offline `dxc` + SPIR-V artifacts) keeps the shader hot
// reload path a single `compile()` call away from any caller.
//
// Stages supported: vertex, fragment. Compute and other stages can
// be added by extending the `Stage` enum without touching callers.
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

    // Read an HLSL source file from disk, compile to SPIR-V via DXC,
    // and create a VkShaderModule. Caller owns the returned handle
    // and must `vkDestroyShaderModule` it (typically once the
    // pipeline that consumes it has been created and the source
    // modules are no longer needed, per the standard Vulkan idiom).
    //
    // `entryPoint` names the HLSL entry function to compile — one
    // file can contain multiple entry points (`VSMain`, `PSMain`,
    // `CSMain`, etc.) so the name is always explicit. There is no
    // default: HLSL convention is explicit naming, and forcing the
    // caller to say "VSMain" prevents copy-paste bugs where the
    // wrong stage silently compiles the wrong function.
    VkShaderModule compile(const std::filesystem::path& absolutePath,
                           Stage stage,
                           const std::string& entryPoint);

    // Non-fatal variant of compile(). Returns VK_NULL_HANDLE on any
    // failure (missing file, syntax error, vkCreateShaderModule
    // error) instead of asserting. Used by hot reload so a typo in
    // a shader source does not crash the engine — the caller keeps
    // its previous module/pipeline intact and the developer fixes
    // and saves again.
    VkShaderModule tryCompile(const std::filesystem::path& absolutePath,
                              Stage stage,
                              const std::string& entryPoint);

private:
    Device*              m_device         = nullptr;
    IDxcUtils*           m_dxcUtils       = nullptr;
    IDxcCompiler3*       m_dxcCompiler    = nullptr;
    IDxcIncludeHandler*  m_includeHandler = nullptr;
};

} // namespace enigma::gfx
