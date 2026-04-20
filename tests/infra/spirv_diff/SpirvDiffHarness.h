// SpirvDiffHarness
// ================
// Compile every HLSL shader in `shaders/` with the same DXC flags
// the engine uses at runtime (mirrored from src/gfx/ShaderManager.cpp)
// and byte-diff the resulting SPIR-V against a golden blob.
//
// Purpose (plan §3.M0b Principle 1): M3 introduces an `MP_ENABLE`
// spec constant; we must prove the SPIR-V is bit-identical to HEAD
// when `MP_ENABLE=false`. That proof needs a HEAD baseline to diff
// against — this harness captures and enforces it.
//
// Scope (M0b): freeze HEAD's SPIR-V. No MP_ENABLE wiring here; that
// is M3. M0b's golden is the pre-micropoly SPIR-V snapshot.
//
// Scope NOT covered:
// - We do NOT re-implement the Vulkan side of shader loading; this
//   harness stops at the SPIR-V blob.
// - We do NOT invoke the engine's ShaderManager class directly — it
//   depends on a full Vulkan device for VkShaderModule creation,
//   which this harness does not need. We duplicate the DXC flag list
//   (see DxcFlags in the .cpp) and keep it in sync by convention.
//   TBD(M0b-exec): if the flag list drifts, a future pass can refactor
//   ShaderManager to expose its argv vector from a shared header.

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace enigma::test_infra {

// Stage enumeration — mirrors ShaderManager::Stage (src/gfx/ShaderManager.h)
// but lives in test_infra to keep the harness independent of the
// engine link graph.
enum class SpirvStage {
    Vertex,
    Fragment,
    Compute,
    Task,
    Mesh,
    RayGeneration,
    ClosestHit,
    Miss,
};

// One entry per (hlsl file, entry point, stage, defines) tuple. The same
// `.hlsl` file can contribute multiple entries when it defines multiple
// entry functions (e.g. mesh.hlsl -> VSMain + PSMain), or when a single
// (file, entry) pair is compiled with different preprocessor defines and
// we want separate goldens per variant (distinguished by goldenName).
//
// Preprocessor-define note (M3.4 / material_eval MP_ENABLE gate): the
// two pipeline variants are compiled from the same HLSL source with
// different `-D` defines passed to DXC. The =false compile (no defines)
// never sees Int64 types and produces SPIR-V byte-identical to the
// pre-M3.4 golden (Principle 1). The =true compile passes
// `defines = {"MP_ENABLE=1"}` and produces a distinct SPIR-V blob that
// includes the micropoly merge path. Both are captured as separate goldens
// so any regression in either variant is caught immediately.
struct ShaderEntry {
    std::string              relativePath; // relative to shaders/ root, forward slashes
    SpirvStage               stage;
    std::string              entryPoint;
    std::string              goldenName;   // file name in golden/, without directory
    // Extra preprocessor defines passed as -D to DXC. Empty for most entries.
    std::vector<std::string> defines;
};

// Full list of shaders the harness covers. Built once in the .cpp to
// keep the header free of dependencies on std::vector static init.
const std::vector<ShaderEntry>& allShaderEntries();

// Result of a single SPIR-V diff.
struct SpirvDiffResult {
    bool                    passed = false;
    std::string             entryName;     // "mesh.hlsl:VSMain"
    std::size_t             producedSize = 0;
    std::size_t             goldenSize   = 0;
    std::size_t             mismatchOffset = 0; // first differing byte
    std::string             message;
};

// Compile one shader entry against the shader source directory rooted
// at `shaderRoot`, and diff against `goldenDir / entry.goldenName`.
// On mismatch, `message` quotes a short spirv-dis-style hint (first
// ~64 bytes hex) rather than invoking the external tool — we avoid
// making spirv-dis a build-time dependency of the test. The Vulkan
// SDK ships spirv-dis though, so devs can run it manually.
//
// Allowlist: the result is considered a pass if the diff falls within
// any of the documented byte-range tolerances below
// (`allowedDiffRanges`). Today the list is empty — DXC is stable for
// the pinned Vulkan SDK version we build against. If future SDK
// bumps reveal drift (e.g. a version timestamp embedded in the
// module), document the range here with a justification comment.
SpirvDiffResult diffShader(const ShaderEntry& entry,
                            const std::filesystem::path& shaderRoot,
                            const std::filesystem::path& goldenDir);

// Compile one shader entry and overwrite `goldenDir / entry.goldenName`.
// Used to bootstrap the baseline and — behind a humans-in-the-loop
// gate — to refresh the golden after an explicitly-justified change.
bool captureGolden(const ShaderEntry& entry,
                    const std::filesystem::path& shaderRoot,
                    const std::filesystem::path& goldenDir);

// Compile one shader entry and return the SPIR-V bytes. Empty vector
// on failure. Exposed for tooling that wants the raw blob — the
// harness itself uses the diffShader/captureGolden wrappers.
std::vector<std::uint8_t> compileOne(const ShaderEntry& entry,
                                      const std::filesystem::path& shaderRoot);

} // namespace enigma::test_infra
