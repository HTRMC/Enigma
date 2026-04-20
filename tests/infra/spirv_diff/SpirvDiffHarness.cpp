// SpirvDiffHarness — DXC-driven HLSL -> SPIR-V compilation + byte diff.
//
// Mirrors src/gfx/ShaderManager.cpp argv exactly so the captured
// SPIR-V matches runtime byte-for-byte; any drift would defeat the
// Principle-1 guarantee at M3.

#include "SpirvDiffHarness.h"

// dxcapi.h pulls in Win32 COM types; NOMINMAX avoids macro pollution
// under /std:c++latest. See ShaderManager.cpp for the full rationale.
#include <Windows.h>
#include <dxc/dxcapi.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace enigma::test_infra {

namespace {

// --- flag vector mirrored from ShaderManager::tryCompile --------------------
//
// Keep this list in lockstep with the runtime one. A drift here means
// the golden SPIR-V no longer matches what the engine produces — the
// whole point of the harness is defeated. See src/gfx/ShaderManager.cpp
// lines ~123-151 for the source.
//
// Flags match `ShaderManager::tryCompile` at its effective
// `#if ENIGMA_DEBUG` = 0 branch (the engine never defines ENIGMA_DEBUG
// in the shipped/canonical config — NDEBUG gates ENIGMA_DEBUG via
// src/core/Assert.h, and the release-track CMake builds all pass
// -DNDEBUG). The dormant `#if ENIGMA_DEBUG` arm in ShaderManager would
// emit `-O0 -Zi -Qembed_debug`; we intentionally do NOT mirror that
// here because the golden SPIR-V set is the shipped SPIR-V, not a
// Debug-config variant. See `src/gfx/ShaderManager.cpp:145-151` for
// the macro arms, and `src/core/Assert.h:22-26` for the gate. If a
// future pass wants a Debug-config golden set, capture a parallel
// tree under `golden/debug/` rather than swapping this list.

const wchar_t* dxcProfile(SpirvStage stage) {
    switch (stage) {
        case SpirvStage::Vertex:        return L"vs_6_0";
        case SpirvStage::Fragment:      return L"ps_6_0";
        case SpirvStage::Compute:       return L"cs_6_0";
        case SpirvStage::Task:          return L"as_6_5";
        case SpirvStage::Mesh:          return L"ms_6_5";
        case SpirvStage::RayGeneration: return L"lib_6_3";
        case SpirvStage::ClosestHit:    return L"lib_6_3";
        case SpirvStage::Miss:          return L"lib_6_3";
    }
    return L"vs_6_0";
}

std::wstring widenAscii(std::string_view narrow) {
    std::wstring out;
    out.reserve(narrow.size());
    for (char c : narrow) {
        out.push_back(static_cast<wchar_t>(static_cast<unsigned char>(c)));
    }
    return out;
}

std::string readFile(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::in | std::ios::binary);
    if (!in) return {};
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

bool writeBinary(const std::filesystem::path& p, const std::vector<std::uint8_t>& bytes) {
    const auto parent = p.parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
    }
    std::ofstream out(p, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!out) return false;
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    return out.good();
}

std::vector<std::uint8_t> readBinary(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::in | std::ios::binary | std::ios::ate);
    if (!in) return {};
    const std::streamsize size = in.tellg();
    if (size < 0) return {};
    in.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    if (!in.read(reinterpret_cast<char*>(bytes.data()), size)) return {};
    return bytes;
}

// --- DXC singleton ----------------------------------------------------------

struct DxcContext {
    IDxcUtils*          utils          = nullptr;
    IDxcCompiler3*      compiler       = nullptr;
    IDxcIncludeHandler* includeHandler = nullptr;

    ~DxcContext() {
        if (includeHandler) includeHandler->Release();
        if (compiler)       compiler->Release();
        if (utils)          utils->Release();
    }
};

bool initDxc(DxcContext& ctx) {
    if (FAILED(DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&ctx.utils)))) {
        std::fprintf(stderr, "[SpirvDiffHarness] DxcCreateInstance(CLSID_DxcUtils) failed\n");
        return false;
    }
    if (FAILED(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&ctx.compiler)))) {
        std::fprintf(stderr, "[SpirvDiffHarness] DxcCreateInstance(CLSID_DxcCompiler) failed\n");
        return false;
    }
    if (FAILED(ctx.utils->CreateDefaultIncludeHandler(&ctx.includeHandler))) {
        std::fprintf(stderr, "[SpirvDiffHarness] CreateDefaultIncludeHandler failed\n");
        return false;
    }
    return true;
}

// Compile one entry. Returns the raw SPIR-V bytes (empty on failure).
std::vector<std::uint8_t> compileEntry(DxcContext& dxc,
                                        const std::filesystem::path& absHlslPath,
                                        SpirvStage stage,
                                        const std::string& entryPoint,
                                        const std::vector<std::string>& defines = {}) {
    const std::string source = readFile(absHlslPath);
    if (source.empty()) {
        std::fprintf(stderr, "[SpirvDiffHarness] missing or empty source: %s\n",
                     absHlslPath.string().c_str());
        return {};
    }

    const std::wstring wEntry      = widenAscii(entryPoint);
    const std::wstring wIncludeDir = absHlslPath.parent_path().wstring();
    const wchar_t*     profile     = dxcProfile(stage);

    // Widen defines up front so wstrings stay live for the args vector.
    std::vector<std::wstring> wDefines;
    wDefines.reserve(defines.size());
    for (const auto& d : defines) {
        wDefines.push_back(widenAscii(d));
    }

    DxcBuffer sourceBuffer{};
    sourceBuffer.Ptr      = source.data();
    sourceBuffer.Size     = source.size();
    sourceBuffer.Encoding = DXC_CP_UTF8;

    // Mirror ShaderManager's argv order and flag set exactly.
    std::vector<LPCWSTR> args;
    args.push_back(L"-E"); args.push_back(wEntry.c_str());
    args.push_back(L"-T"); args.push_back(profile);
    args.push_back(L"-spirv");
    args.push_back(L"-fspv-target-env=vulkan1.3");
    args.push_back(L"-fvk-use-dx-layout");
    args.push_back(L"-Zpc");
    args.push_back(L"-I"); args.push_back(wIncludeDir.c_str());
    args.push_back(L"-HV"); args.push_back(L"2021");
    for (const auto& wd : wDefines) {
        args.push_back(L"-D");
        args.push_back(wd.c_str());
    }
    // architect-FixA: match the #else (ENIGMA_DEBUG == 0) arm of
    // ShaderManager::tryCompile at src/gfx/ShaderManager.cpp:149-150.
    // Release/RelWithDebInfo builds define NDEBUG so ENIGMA_DEBUG is
    // 0 and the runtime emits just -O3. The golden SPIR-V tree under
    // golden/ is captured with this same flag set; any drift here
    // defeats the Principle-1 bit-identity guarantee.
    args.push_back(L"-O3");

    IDxcResult* result = nullptr;
    const HRESULT hr = dxc.compiler->Compile(
        &sourceBuffer,
        args.data(),
        static_cast<UINT32>(args.size()),
        dxc.includeHandler,
        IID_PPV_ARGS(&result));
    if (FAILED(hr) || result == nullptr) {
        std::fprintf(stderr, "[SpirvDiffHarness] DXC Compile failed: 0x%08x (%s:%s)\n",
                     static_cast<unsigned>(hr),
                     absHlslPath.filename().string().c_str(), entryPoint.c_str());
        if (result) result->Release();
        return {};
    }

    // code-reviewer MAJOR-2 fix: DXC GetStatus itself returns HRESULT.
    // Mirrors src/gfx/ShaderManager.cpp:169-175.
    HRESULT status = S_OK;
    const HRESULT statusHr = result->GetStatus(&status);
    if (FAILED(statusHr)) {
        std::fprintf(stderr, "[SpirvDiffHarness] DXC GetStatus failed: 0x%08x (%s:%s)\n",
                     static_cast<unsigned>(statusHr),
                     absHlslPath.filename().string().c_str(), entryPoint.c_str());
        result->Release();
        return {};
    }

    IDxcBlobUtf8* errors = nullptr;
    result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errors), nullptr);
    if (FAILED(status)) {
        if (errors && errors->GetStringLength() > 0) {
            std::fprintf(stderr, "[SpirvDiffHarness] compile failed %s:%s\n%s\n",
                         absHlslPath.filename().string().c_str(), entryPoint.c_str(),
                         errors->GetStringPointer());
        } else {
            std::fprintf(stderr, "[SpirvDiffHarness] compile failed %s:%s (no error text)\n",
                         absHlslPath.filename().string().c_str(), entryPoint.c_str());
        }
        if (errors) errors->Release();
        result->Release();
        return {};
    }
    // Non-fatal warnings — print but continue.
    if (errors && errors->GetStringLength() > 0) {
        std::fprintf(stderr, "[SpirvDiffHarness] warnings for %s:%s\n%s\n",
                     absHlslPath.filename().string().c_str(), entryPoint.c_str(),
                     errors->GetStringPointer());
    }
    if (errors) errors->Release();

    IDxcBlob* blob = nullptr;
    result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&blob), nullptr);
    if (!blob || blob->GetBufferSize() == 0) {
        std::fprintf(stderr, "[SpirvDiffHarness] empty SPIR-V for %s:%s\n",
                     absHlslPath.filename().string().c_str(), entryPoint.c_str());
        if (blob) blob->Release();
        result->Release();
        return {};
    }

    std::vector<std::uint8_t> bytes(blob->GetBufferSize());
    std::memcpy(bytes.data(), blob->GetBufferPointer(), bytes.size());
    blob->Release();
    result->Release();
    return bytes;
}

// --- entry manifest ---------------------------------------------------------
//
// Derived from a sweep of src/renderer/*.cpp for .compile() call sites
// (see the initial investigation pass). Keeping the list explicit rather
// than auto-discovering it from directory globs keeps the golden set
// deterministic — adding a new shader is a deliberate act that also
// updates this manifest.

std::vector<ShaderEntry> buildManifest() {
    using S = SpirvStage;
    // {relativePath, stage, entry, goldenName}
    // goldenName convention: <stem>.<entry>.<stage>.spv — stem includes
    // any leading subdirectory flattened with underscores so every
    // golden lives in a flat golden/ directory.
    std::vector<ShaderEntry> m;

    auto add = [&](const char* path, SpirvStage stage, const char* entry) {
        ShaderEntry e;
        e.relativePath = path;
        e.stage        = stage;
        e.entryPoint   = entry;

        // Build a flat golden name. Replace '/' and '\\' with '_' and
        // drop the `.hlsl` suffix before appending entry+stage.
        std::string name = path;
        for (char& c : name) {
            if (c == '/' || c == '\\') c = '_';
        }
        const std::string suffix = ".hlsl";
        if (name.size() > suffix.size()
            && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
            name.resize(name.size() - suffix.size());
        }
        name += ".";
        name += entry;
        name += ".";
        switch (stage) {
            case S::Vertex:        name += "vs"; break;
            case S::Fragment:      name += "ps"; break;
            case S::Compute:       name += "cs"; break;
            case S::Task:          name += "as"; break;
            case S::Mesh:          name += "ms"; break;
            case S::RayGeneration: name += "rgen"; break;
            case S::ClosestHit:    name += "rchit"; break;
            case S::Miss:          name += "rmiss"; break;
        }
        name += ".spv";
        e.goldenName = name;
        m.push_back(std::move(e));
    };

    // --- geometry / lighting / post passes (VS + PS pairs) -----------------
    add("triangle.hlsl",                 S::Vertex,   "VSMain");
    add("triangle.hlsl",                 S::Fragment, "PSMain");
    add("mesh.hlsl",                     S::Vertex,   "VSMain");
    add("mesh.hlsl",                     S::Fragment, "PSMain");
    add("lighting.hlsl",                 S::Vertex,   "VSMain");
    add("lighting.hlsl",                 S::Fragment, "PSMain");
    add("clustered_forward.hlsl",        S::Vertex,   "VSMain");
    add("clustered_forward.hlsl",        S::Fragment, "PSMain");
    add("sky_background.hlsl",           S::Vertex,   "VSMain");
    add("sky_background.hlsl",           S::Fragment, "PSMain");
    add("post_process.hlsl",             S::Vertex,   "VSMain");
    add("post_process.hlsl",             S::Fragment, "PSMain");

    // --- debug passes ------------------------------------------------------
    add("debug_unlit.hlsl",              S::Vertex,   "VSMain");
    add("debug_unlit.hlsl",              S::Fragment, "PSMain");
    add("debug_blit.hlsl",               S::Vertex,   "VSMain");
    add("debug_blit.hlsl",               S::Fragment, "PSMain");
    add("debug_clusters.hlsl",           S::Vertex,   "VSMain");
    add("debug_clusters.hlsl",           S::Fragment, "PSMain");
    add("debug_detail_lighting.hlsl",    S::Vertex,   "VSMain");
    add("debug_detail_lighting.hlsl",    S::Fragment, "PSMain");
    add("debug_wireframe.frag.hlsl",     S::Fragment, "PSMain");
    add("debug_wireframe_terrain.frag.hlsl", S::Fragment, "PSMain");
    // M4.6 — MicropolyRasterClass debug overlay. Samples the 64-bit vis
    // image's rasterClassBits and writes per-pixel R/G/black. Two entries
    // (VS + PS) match the standard debug fullscreen-triangle shape.
    add("debug_micropoly_raster_class.hlsl", S::Vertex,   "VSMain");
    add("debug_micropoly_raster_class.hlsl", S::Fragment, "PSMain");
    // M6.1 — Micropoly LOD + residency heatmaps. Same fullscreen-triangle
    // shape as RasterClass; each adds its own VS + PS pair to the golden
    // manifest. Must stay separate from the M4.6 entry so a breakage in
    // one heatmap doesn't mask drift in the other.
    add("debug_micropoly_lod_heatmap.hlsl",       S::Vertex,   "VSMain");
    add("debug_micropoly_lod_heatmap.hlsl",       S::Fragment, "PSMain");
    add("debug_micropoly_residency_heatmap.hlsl", S::Vertex,   "VSMain");
    add("debug_micropoly_residency_heatmap.hlsl", S::Fragment, "PSMain");
    // M6.2b — Micropoly bounds (per-cluster bounding-sphere wireframe).
    // Fullscreen VS + PS; iterates DAG, projects spheres to pixel space.
    // Principle 1: existing goldens byte-identical — this only adds two.
    add("debug_micropoly_bounds.hlsl",            S::Vertex,   "VSMain");
    add("debug_micropoly_bounds.hlsl",            S::Fragment, "PSMain");
    // M6 plan §3.M6 — Micropoly BinOverflow heatmap (SW raster tile bin
    // fill-level). Reads tileBinCount + spillBuffer SSBOs. Same
    // fullscreen-triangle shape as other debug overlays — VS + PS pair.
    // Principle 1: existing goldens byte-identical — this only adds two.
    add("debug_micropoly_bin_overflow.hlsl",      S::Vertex,   "VSMain");
    add("debug_micropoly_bin_overflow.hlsl",      S::Fragment, "PSMain");
    add("physics_debug.hlsl",            S::Vertex,   "VSMain");
    add("physics_debug.hlsl",            S::Fragment, "PSMain");

    // --- visibility buffer (mesh / task / vs fallback) ---------------------
    add("visibility_buffer.task.hlsl",   S::Task,     "ASMain");
    add("visibility_buffer.mesh.hlsl",   S::Mesh,     "MSMain");
    add("visibility_buffer.mesh.hlsl",   S::Fragment, "PSMain");
    add("visibility_buffer_vs.hlsl",     S::Vertex,   "VSMain");
    add("visibility_buffer_vs.hlsl",     S::Fragment, "PSMain");

    // --- terrain CDLOD mesh pipeline ---------------------------------------
    add("terrain_cdlod.task.hlsl",       S::Task,     "ASMain");
    add("terrain_cdlod.mesh.hlsl",       S::Mesh,     "MSMain");
    add("terrain_cdlod.mesh.hlsl",       S::Fragment, "PSMain");

    // --- compute shaders ---------------------------------------------------
    add("hiz_build.comp.hlsl",           S::Compute,  "CSMain");
    add("gpu_cull.comp.hlsl",            S::Compute,  "CSMain");
    add("material_eval.comp.hlsl",       S::Compute,  "CSMain");
    add("denoise_spatial.hlsl",          S::Compute,  "CSMain");
    add("denoise_temporal.hlsl",         S::Compute,  "CSMain");
    add("atmosphere_transmittance.hlsl", S::Compute,  "CSMain");
    add("atmosphere_multiscatter.hlsl",  S::Compute,  "CSMain");
    add("atmosphere_skyview.hlsl",       S::Compute,  "CSMain");
    add("atmosphere_aerial_perspective.hlsl", S::Compute, "CSMain");
    // M3.2 — micropoly cluster cull. First-class entry point. The include-
    // only helpers (mp_vis_pack.hlsl, page_request_emit.hlsl) remain off
    // the manifest per their file-banner contract.
    add("micropoly/mp_cluster_cull.comp.hlsl", S::Compute, "CSMain");

    // M3.4 — second golden for material_eval compiled with -D MP_ENABLE=1.
    // This produces a distinct SPIR-V blob that includes the 64-bit vis
    // image read and micropoly merge path. The baseline entry above (no
    // defines) must remain byte-identical to the pre-M3.4 golden (Principle 1).
    {
        ShaderEntry e;
        e.relativePath = "material_eval.comp.hlsl";
        e.stage        = S::Compute;
        e.entryPoint   = "CSMain";
        e.goldenName   = "material_eval.comp.MP_ENABLE.CSMain.cs.spv";
        e.defines      = {"MP_ENABLE=1"};
        m.push_back(std::move(e));
    }

    // M3.3 — micropoly HW raster (task + mesh + fragment). The fragment
    // entry point lives in the same mesh-shader .hlsl file; manifest
    // covers all three entry points so each gets its own golden blob.
    add("micropoly/mp_raster.task.hlsl", S::Task,     "ASMain");
    add("micropoly/mp_raster.mesh.hlsl", S::Mesh,     "MSMain");
    add("micropoly/mp_raster.mesh.hlsl", S::Fragment, "PSMain");

    // M4.2 — micropoly SW raster binning (prep + bin). Both are single-entry
    // compute shaders. The fragment/rasterisation compute that consumes
    // these bins lands in M4.3 as `sw_raster.comp.hlsl:CSMain` (below).
    add("micropoly/sw_raster_bin_prep.comp.hlsl", S::Compute, "CSMain");
    add("micropoly/sw_raster_bin.comp.hlsl",      S::Compute, "CSMain");
    // M4.3 — SW raster fragment pass. Reads the tile-bin SSBOs populated
    // by sw_raster_bin.comp.hlsl and writes per-pixel samples to the
    // bindless R64_UINT vis image via InterlockedMax (reverse-Z).
    add("micropoly/sw_raster.comp.hlsl",          S::Compute, "CSMain");

    // --- SMAA (VS + PS triples: edge / blend / neighborhood) ---------------
    add("smaa_edge.hlsl",                S::Vertex,   "VSMain");
    add("smaa_edge.hlsl",                S::Fragment, "PSMain");
    add("smaa_blend.hlsl",               S::Vertex,   "VSMain");
    add("smaa_blend.hlsl",               S::Fragment, "PSMain");
    add("smaa_neighborhood.hlsl",        S::Vertex,   "VSMain");
    add("smaa_neighborhood.hlsl",        S::Fragment, "PSMain");

    // --- SSR / SSAO / CSM / wet-road raster fallbacks ----------------------
    add("ssr.hlsl",                      S::Vertex,   "VSMain");
    add("ssr.hlsl",                      S::Fragment, "PSMain");
    add("ssao.hlsl",                     S::Vertex,   "VSMain");
    add("ssao.hlsl",                     S::Fragment, "PSMain");
    add("csm.hlsl",                      S::Vertex,   "VSMain");
    add("csm.hlsl",                      S::Fragment, "PSMain");
    add("wetroad_fallback.hlsl",         S::Vertex,   "VSMain");
    add("wetroad_fallback.hlsl",         S::Fragment, "PSMain");

    // --- ray tracing libraries --------------------------------------------
    add("rt/reflection.rgen.hlsl",       S::RayGeneration, "RayGenMain");
    add("rt/reflection.rchit.hlsl",      S::ClosestHit,    "ClosestHitMain");
    add("rt/reflection.rmiss.hlsl",      S::Miss,          "MissMain");
    add("rt/gi.rgen.hlsl",               S::RayGeneration, "RayGenMain");
    add("rt/gi.rchit.hlsl",              S::ClosestHit,    "ClosestHitMain");
    add("rt/shadow.rgen.hlsl",           S::RayGeneration, "RayGenMain");
    add("rt/shadow.rmiss.hlsl",          S::Miss,          "MissMain");
    add("rt/wetroad.rgen.hlsl",          S::RayGeneration, "RayGenMain");
    // Note: RT reflection/wetroad share gi's reflection.rchit in the engine;
    // gi.rchit is the canonical chit used by reflection + gi.  shadow.rmiss
    // is the miss shader for shadow rays. wetroad reuses reflection.rchit/rmiss
    // at runtime; here we only compile the entry points that each .hlsl file
    // actually defines.

    return m;
}

// --- allowlist: byte ranges tolerated as "spurious drift" -------------------
//
// Today the allowlist is empty — under -Zpc -spirv -O0 -Zi -Qembed_debug
// the output is deterministic for a pinned DXC version. If a future SDK
// bump introduces a timestamp or compiler-version blob, add the range
// here with a justification comment. Example schema:
//
//     struct ByteRange { std::size_t offset; std::size_t length; };
//     constexpr ByteRange kAllowlist[] = {
//         // dxc v1.8 embeds a 16-byte build-id at offset 0x10 of each
//         // OpSource instruction. Safe to ignore for bit-identity checks.
//         // { 0x10, 16 },
//     };
//
// Explicit documentation in the header (see SpirvDiffHarness.h) mirrors
// this — any relaxation has to be justified and peer-reviewed.

struct ByteRange { std::size_t offset; std::size_t length; };
// Empty at HEAD — no tolerated DXC drift for the pinned Vulkan SDK.
// Array is declared as a std::array<> so a zero-length allowlist is a
// legal constant (MSVC rejects `ByteRange kAllowlist[0]`-style decls).
constexpr std::array<ByteRange, 0> kAllowlist{};

bool withinAllowlist(std::size_t offset) {
    for (const auto& r : kAllowlist) {
        if (offset >= r.offset && offset < r.offset + r.length) return true;
    }
    return false;
}

// --- per-process DXC instance -----------------------------------------------
//
// Constructed on first use, torn down at process exit. Cheaper than
// re-creating for each of ~50 shader entries.
DxcContext& dxcSingleton() {
    static DxcContext ctx;
    static bool initialized = false;
    if (!initialized) {
        initialized = initDxc(ctx); // if this fails, subsequent calls produce empty blobs
    }
    return ctx;
}

} // namespace

const std::vector<ShaderEntry>& allShaderEntries() {
    static const std::vector<ShaderEntry> manifest = buildManifest();
    return manifest;
}

std::vector<std::uint8_t> compileOne(const ShaderEntry& entry,
                                      const std::filesystem::path& shaderRoot) {
    DxcContext& dxc = dxcSingleton();
    if (!dxc.compiler) return {};
    const std::filesystem::path abs = shaderRoot / entry.relativePath;
    return compileEntry(dxc, abs, entry.stage, entry.entryPoint, entry.defines);
}

SpirvDiffResult diffShader(const ShaderEntry& entry,
                            const std::filesystem::path& shaderRoot,
                            const std::filesystem::path& goldenDir) {
    SpirvDiffResult r;
    r.entryName = entry.relativePath + ":" + entry.entryPoint;

    const std::vector<std::uint8_t> produced = compileOne(entry, shaderRoot);
    if (produced.empty()) {
        r.message = "compile failed: " + r.entryName;
        return r;
    }
    r.producedSize = produced.size();

    const std::filesystem::path goldenPath = goldenDir / entry.goldenName;
    if (!std::filesystem::exists(goldenPath)) {
        r.message = "golden missing: " + goldenPath.string()
                  + " (run spirv_diff_generate_baseline)";
        return r;
    }
    const std::vector<std::uint8_t> golden = readBinary(goldenPath);
    r.goldenSize = golden.size();
    if (golden.empty()) {
        r.message = "golden unreadable or empty: " + goldenPath.string();
        return r;
    }

    // code-reviewer MAJOR-3 / security LOW-3 fix: verify SPIR-V magic
    // before diffing. Defence in depth — a truncated or mis-filed golden
    // would otherwise produce misleading "first diff at byte 0" output.
    // The SPIR-V spec (§2.3) fixes the first word at 0x07230203 little-
    // endian. We check both the golden and the produced blob.
    constexpr std::uint32_t kSpirvMagic = 0x07230203u;
    if (golden.size() < 4 || std::memcmp(golden.data(), &kSpirvMagic, 4) != 0) {
        r.message = "golden is not a valid SPIR-V blob (bad magic): "
                  + goldenPath.string();
        return r;
    }
    if (produced.size() < 4 || std::memcmp(produced.data(), &kSpirvMagic, 4) != 0) {
        r.message = "produced blob is not valid SPIR-V (bad magic) for " + r.entryName;
        return r;
    }

    if (produced.size() != golden.size()) {
        // Size mismatch is always a real diff — allowlist is for ranges
        // WITHIN a constant-size blob only.
        r.mismatchOffset = 0;
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "size mismatch for %s: produced=%zu golden=%zu",
            r.entryName.c_str(), produced.size(), golden.size());
        r.message = buf;
        return r;
    }

    for (std::size_t i = 0; i < produced.size(); ++i) {
        if (produced[i] != golden[i]) {
            if (withinAllowlist(i)) continue;
            r.mismatchOffset = i;
            char buf[512];
            std::snprintf(buf, sizeof(buf),
                "%s: first diff at byte offset %zu (produced=0x%02x golden=0x%02x)",
                r.entryName.c_str(), i,
                static_cast<unsigned>(produced[i]),
                static_cast<unsigned>(golden[i]));
            r.message = buf;
            return r;
        }
    }

    r.passed = true;
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "%s OK (%zu bytes identical to golden)",
        r.entryName.c_str(), produced.size());
    r.message = buf;
    return r;
}

bool captureGolden(const ShaderEntry& entry,
                    const std::filesystem::path& shaderRoot,
                    const std::filesystem::path& goldenDir) {
    const std::vector<std::uint8_t> produced = compileOne(entry, shaderRoot);
    if (produced.empty()) {
        std::fprintf(stderr, "[SpirvDiffHarness] capture failed: %s:%s\n",
                     entry.relativePath.c_str(), entry.entryPoint.c_str());
        return false;
    }
    const std::filesystem::path outPath = goldenDir / entry.goldenName;
    if (!writeBinary(outPath, produced)) {
        std::fprintf(stderr, "[SpirvDiffHarness] failed to write golden: %s\n",
                     outPath.string().c_str());
        return false;
    }
    std::printf("[SpirvDiffHarness] captured %s (%zu bytes) -> %s\n",
                (entry.relativePath + ":" + entry.entryPoint).c_str(),
                produced.size(), outPath.string().c_str());
    return true;
}

} // namespace enigma::test_infra
