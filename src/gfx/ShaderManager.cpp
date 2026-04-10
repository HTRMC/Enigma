#include "gfx/ShaderManager.h"

#include "core/Assert.h"
#include "core/Log.h"
#include "gfx/Device.h"

// dxcapi.h uses Win32 typedefs (UINT32, LPCWSTR, LPCVOID) AND COM
// types (IUnknown::Release, IID_PPV_ARGS) without including
// <Windows.h> itself. WIN32_LEAN_AND_MEAN is deliberately NOT set
// because it excludes <ole2.h>, which in turn pulls in <unknwn.h>
// (IUnknown) and <combaseapi.h> (IID_PPV_ARGS) — both of which the
// DXC COM interfaces need at the call site. We still set NOMINMAX
// to keep `min`/`max` from becoming macros.
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
        case ShaderManager::Stage::Vertex:        return L"vs_6_0";
        case ShaderManager::Stage::Fragment:      return L"ps_6_0";
        // All RT stages share the HLSL shader library target. The Vulkan
        // stage distinction (rgen/rchit/rmiss/rahit/rint) is declared via
        // HLSL attributes ([shader("raygeneration")] etc.) inside the
        // source file; DXC routes them to the correct SPIR-V ExecutionModel.
        case ShaderManager::Stage::RayGeneration: return L"lib_6_3";
        case ShaderManager::Stage::ClosestHit:    return L"lib_6_3";
        case ShaderManager::Stage::Miss:          return L"lib_6_3";
        case ShaderManager::Stage::AnyHit:        return L"lib_6_3";
        case ShaderManager::Stage::Intersection:  return L"lib_6_3";
        case ShaderManager::Stage::Compute:       return L"cs_6_0";
    }
    ENIGMA_ASSERT(false && "unknown shader stage for DXC profile");
    return L"vs_6_0";
}

} // namespace

ShaderManager::ShaderManager(Device& device)
    : m_device(&device) {
    // DXC instances. `DxcCreateInstance` is a standalone factory and
    // does NOT require `CoInitialize` — see Microsoft DXC docs. Both
    // objects are thread-affine to the render thread in this engine,
    // so no additional synchronization is needed.
    const HRESULT utilsHr = DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&m_dxcUtils));
    ENIGMA_ASSERT(SUCCEEDED(utilsHr) && "DxcCreateInstance(CLSID_DxcUtils) failed");
    const HRESULT compilerHr = DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&m_dxcCompiler));
    ENIGMA_ASSERT(SUCCEEDED(compilerHr) && "DxcCreateInstance(CLSID_DxcCompiler) failed");

    const HRESULT includeHr = m_dxcUtils->CreateDefaultIncludeHandler(&m_includeHandler);
    ENIGMA_ASSERT(SUCCEEDED(includeHr) && "CreateDefaultIncludeHandler failed");
}

ShaderManager::~ShaderManager() {
    if (m_includeHandler) m_includeHandler->Release();
    if (m_dxcCompiler)    m_dxcCompiler->Release();
    if (m_dxcUtils)       m_dxcUtils->Release();
}

VkShaderModule ShaderManager::tryCompile(const std::filesystem::path& absolutePath,
                                         Stage stage,
                                         const std::string& entryPoint) {
    const std::string source = readTextFileSafe(absolutePath);
    if (source.empty()) {
        ENIGMA_LOG_ERROR("[shader] failed to open or empty source: {}", absolutePath.string());
        return VK_NULL_HANDLE;
    }

    const std::string sourceName = absolutePath.filename().string();
    const std::wstring wEntry = widenAscii(entryPoint);
    const std::wstring wIncludeDir = absolutePath.parent_path().wstring();
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
    args.push_back(L"-Zpc");  // column-major matrices (match glm layout)
    args.push_back(L"-I");
    args.push_back(wIncludeDir.c_str());
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
        m_includeHandler,
        IID_PPV_ARGS(&result));

    if (FAILED(compileHr) || result == nullptr) {
        ENIGMA_LOG_ERROR("[shader] DXC Compile HRESULT failed: 0x{:x} ({})",
                         static_cast<std::uint32_t>(compileHr), sourceName);
        if (result) result->Release();
        return VK_NULL_HANDLE;
    }

    HRESULT compileStatus = S_OK;
    const HRESULT statusHr = result->GetStatus(&compileStatus);
    if (FAILED(statusHr)) {
        ENIGMA_LOG_ERROR("[shader] DXC GetStatus HRESULT failed: 0x{:x} ({})",
                         static_cast<std::uint32_t>(statusHr), sourceName);
        result->Release();
        return VK_NULL_HANDLE;
    }

    // Errors and warnings both arrive via DXC_OUT_ERRORS. On failure
    // we log + bail; on success with a non-empty error blob, we log
    // as warning and continue.
    IDxcBlobUtf8* errors = nullptr;
    const HRESULT errorsOutHr = result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errors), nullptr);
    if (FAILED(errorsOutHr)) {
        ENIGMA_LOG_WARN("[shader] DXC GetOutput(DXC_OUT_ERRORS) HRESULT failed: 0x{:x} ({})",
                        static_cast<std::uint32_t>(errorsOutHr), sourceName);
    }

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
    const HRESULT objectOutHr = result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&spirvBlob), nullptr);
    if (FAILED(objectOutHr) || spirvBlob == nullptr || spirvBlob->GetBufferSize() == 0) {
        ENIGMA_LOG_ERROR("[shader] HLSL compile produced empty SPIR-V: {} (GetOutput HRESULT=0x{:x})",
                         sourceName, static_cast<std::uint32_t>(objectOutHr));
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

    ENIGMA_LOG_INFO("[shader] compiled {} ({} bytes SPIR-V, entry={})",
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
