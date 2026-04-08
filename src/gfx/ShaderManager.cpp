#include "gfx/ShaderManager.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "gfx/Device.h"

#include <shaderc/shaderc.hpp>

// dxcapi.h uses Win32 typedefs (UINT32, LPCWSTR, LPCVOID, IUnknown,
// etc.) but does NOT include <Windows.h> itself — it assumes the
// caller has already pulled it in. We lean-and-mean + NOMINMAX to
// keep the pollution minimal but the Windows.h include is mandatory
// on MSVC or dxcapi.h will not parse.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <dxc/dxcapi.h>

#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
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

// Widen an ASCII-only identifier (HLSL entry point names and target
// profiles are all ASCII) into a wide string. DXC's Compile() argv
// is LPCWSTR so every arg needs widening. A raw char->wchar_t cast
// is correct for the 7-bit ASCII subset and avoids pulling in
// MultiByteToWideChar just for identifier widening.
std::wstring widenAscii(std::string_view narrow) {
    std::wstring out;
    out.reserve(narrow.size());
    for (char c : narrow) {
        out.push_back(static_cast<wchar_t>(static_cast<unsigned char>(c)));
    }
    return out;
}

// DXC target profile per shader stage. Shader Model 6.0 is the
// minimum DXC accepts; it maps 1:1 onto SPIR-V for our usage and
// enables the `NonUniformResourceIndex` intrinsic that the bindless
// shaders rely on.
const wchar_t* dxcProfile(ShaderManager::Stage stage) {
    switch (stage) {
        case ShaderManager::Stage::Vertex:   return L"vs_6_0";
        case ShaderManager::Stage::Fragment: return L"ps_6_0";
    }
    ENIGMA_ASSERT(false && "unknown shader stage for DXC profile");
    return L"vs_6_0";
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

    // DXC instances. `DxcCreateInstance` is a standalone factory and
    // does NOT require `CoInitialize` — see Microsoft DXC docs. Both
    // objects are thread-affine to the render thread in this engine,
    // so no additional synchronization is needed.
    const HRESULT utilsHr = DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&m_dxcUtils));
    ENIGMA_ASSERT(SUCCEEDED(utilsHr) && "DxcCreateInstance(CLSID_DxcUtils) failed");
    const HRESULT compilerHr = DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&m_dxcCompiler));
    ENIGMA_ASSERT(SUCCEEDED(compilerHr) && "DxcCreateInstance(CLSID_DxcCompiler) failed");
}

ShaderManager::~ShaderManager() {
    if (m_dxcCompiler) m_dxcCompiler->Release();
    if (m_dxcUtils)    m_dxcUtils->Release();
    delete m_options;
    delete m_compiler;
}

VkShaderModule ShaderManager::tryCompile(const std::filesystem::path& absolutePath,
                                         Stage stage,
                                         const std::string& entryPoint) {
    // Extension-based dispatch: `.hlsl` goes through DXC, everything
    // else (`.vert`, `.frag`, `.glsl`) goes through shaderc. The HLSL
    // migration will eventually delete the shaderc path; extension
    // dispatch keeps both compilers coexisting during the transition.
    const std::string ext = absolutePath.extension().string();
    if (ext == ".hlsl" || ext == ".HLSL") {
        return tryCompileHLSL(absolutePath, stage, entryPoint);
    }
    return tryCompileGLSL(absolutePath, stage, entryPoint);
}

VkShaderModule ShaderManager::tryCompileGLSL(const std::filesystem::path& absolutePath,
                                             Stage stage,
                                             const std::string& entryPoint) {
    const std::string source = readTextFileSafe(absolutePath);
    if (source.empty()) {
        ENIGMA_LOG_ERROR("[shader] failed to open or empty source: {}", absolutePath.string());
        return VK_NULL_HANDLE;
    }

    const std::string sourceName = absolutePath.filename().string();
    const shaderc_shader_kind kind = toShaderKind(stage);

    shaderc::SpvCompilationResult result =
        m_compiler->CompileGlslToSpv(source, kind, sourceName.c_str(),
                                     entryPoint.c_str(), *m_options);

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

    ENIGMA_LOG_INFO("[shader] compiled {} ({} words, glsl)", sourceName, spirv.size());
    return module;
}

VkShaderModule ShaderManager::tryCompileHLSL(const std::filesystem::path& absolutePath,
                                             Stage stage,
                                             const std::string& entryPoint) {
    const std::string source = readTextFileSafe(absolutePath);
    if (source.empty()) {
        ENIGMA_LOG_ERROR("[shader] failed to open or empty source: {}", absolutePath.string());
        return VK_NULL_HANDLE;
    }

    const std::string sourceName = absolutePath.filename().string();
    const std::wstring wEntry = widenAscii(entryPoint);
    const wchar_t* profile = dxcProfile(stage);

    // DXC source buffer — UTF-8 matches what ifstream gives us.
    DxcBuffer sourceBuffer{};
    sourceBuffer.Ptr      = source.data();
    sourceBuffer.Size     = source.size();
    sourceBuffer.Encoding = DXC_CP_UTF8;

    // Compile arguments:
    //   -E <entry>          entry point name (VSMain, PSMain, ...)
    //   -T <profile>        target profile (vs_6_0, ps_6_0)
    //   -spirv              emit SPIR-V instead of DXIL
    //   -fvk-use-dx-layout  keep D3D-style cbuffer layout so the C++
    //                       PushBlock struct layout matches byte-for-byte
    //   -HV 2021            HLSL 2021 language version
    //   -O0/-Zi/-Qembed_debug on debug, -O3 on release
    std::vector<LPCWSTR> args;
    args.reserve(16);
    args.push_back(L"-E");
    args.push_back(wEntry.c_str());
    args.push_back(L"-T");
    args.push_back(profile);
    args.push_back(L"-spirv");
    args.push_back(L"-fvk-use-dx-layout");
    args.push_back(L"-HV");
    args.push_back(L"2021");
#if ENIGMA_DEBUG
    args.push_back(L"-O0");
    args.push_back(L"-Zi");
    args.push_back(L"-Qembed_debug");
#else
    args.push_back(L"-O3");
#endif

    IDxcResult* result = nullptr;
    const HRESULT compileHr = m_dxcCompiler->Compile(
        &sourceBuffer,
        args.data(),
        static_cast<UINT32>(args.size()),
        nullptr, // no include handler — shaders are single-file
        IID_PPV_ARGS(&result));

    if (FAILED(compileHr) || result == nullptr) {
        ENIGMA_LOG_ERROR("[shader] DXC Compile HRESULT failed: 0x{:x} ({})",
                         static_cast<std::uint32_t>(compileHr), sourceName);
        if (result) result->Release();
        return VK_NULL_HANDLE;
    }

    HRESULT compileStatus = S_OK;
    result->GetStatus(&compileStatus);

    // Errors and warnings both arrive via DXC_OUT_ERRORS. On failure
    // we log + bail; on success with a non-empty error blob, we log
    // as warning and continue.
    IDxcBlobUtf8* errors = nullptr;
    result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errors), nullptr);

    if (FAILED(compileStatus)) {
        if (errors != nullptr && errors->GetStringLength() > 0) {
            ENIGMA_LOG_ERROR("[shader] HLSL compile failed: {}\n{}",
                             sourceName,
                             std::string_view(errors->GetStringPointer(),
                                              errors->GetStringLength()));
        } else {
            ENIGMA_LOG_ERROR("[shader] HLSL compile failed: {} (no error text)", sourceName);
        }
        if (errors) errors->Release();
        result->Release();
        return VK_NULL_HANDLE;
    }

    if (errors != nullptr && errors->GetStringLength() > 0) {
        ENIGMA_LOG_WARN("[shader] HLSL compile warnings: {}\n{}",
                        sourceName,
                        std::string_view(errors->GetStringPointer(),
                                         errors->GetStringLength()));
    }
    if (errors) errors->Release();

    IDxcBlob* spirvBlob = nullptr;
    result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&spirvBlob), nullptr);
    if (spirvBlob == nullptr || spirvBlob->GetBufferSize() == 0) {
        ENIGMA_LOG_ERROR("[shader] HLSL compile produced empty SPIR-V: {}", sourceName);
        if (spirvBlob) spirvBlob->Release();
        result->Release();
        return VK_NULL_HANDLE;
    }

    VkShaderModuleCreateInfo info{};
    info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = spirvBlob->GetBufferSize();
    info.pCode    = static_cast<const u32*>(spirvBlob->GetBufferPointer());

    VkShaderModule module = VK_NULL_HANDLE;
    const VkResult vr = vkCreateShaderModule(m_device->logical(), &info, nullptr, &module);

    spirvBlob->Release();
    result->Release();

    if (vr != VK_SUCCESS) {
        ENIGMA_LOG_ERROR("[shader] vkCreateShaderModule failed: {} (VkResult={})",
                         sourceName, static_cast<int>(vr));
        return VK_NULL_HANDLE;
    }

    ENIGMA_LOG_INFO("[shader] compiled {} ({} bytes SPIR-V, hlsl, entry={})",
                    sourceName, info.codeSize, entryPoint);
    return module;
}

VkShaderModule ShaderManager::compile(const std::filesystem::path& absolutePath,
                                      Stage stage,
                                      const std::string& entryPoint) {
    VkShaderModule module = tryCompile(absolutePath, stage, entryPoint);
    ENIGMA_ASSERT(module != VK_NULL_HANDLE && "shader compilation failed (initial load)");
    return module;
}

} // namespace enigma::gfx
