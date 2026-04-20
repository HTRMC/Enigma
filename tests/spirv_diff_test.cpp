// spirv_diff_test — compiles every HLSL shader in shaders/ with the
// engine's exact DXC flag list and byte-compares against the golden
// SPIR-V blobs under tests/infra/spirv_diff/golden/.
//
// Exit code 0 iff every entry matches byte-for-byte. Exit 1 on any
// mismatch, missing golden, or compile failure.
//
// Usage:
//   spirv_diff_test                     (normal test run)
//   spirv_diff_test --generate-baseline (regenerate every golden, then exit)
//
// Both shaderRoot and goldenDir are compile-baked absolute paths
// (ENIGMA_SHADER_ROOT_DIR, ENIGMA_SPIRV_GOLDEN_DIR) so the test
// binary works regardless of CWD.

#include "infra/spirv_diff/SpirvDiffHarness.h"

#include <cstdio>
#include <cstring>
#include <filesystem>

#ifndef ENIGMA_SHADER_ROOT_DIR
#define ENIGMA_SHADER_ROOT_DIR ""
#endif
#ifndef ENIGMA_SPIRV_GOLDEN_DIR
#define ENIGMA_SPIRV_GOLDEN_DIR ""
#endif

namespace {

int runDiff() {
    const std::filesystem::path shaderRoot = ENIGMA_SHADER_ROOT_DIR;
    const std::filesystem::path goldenDir  = ENIGMA_SPIRV_GOLDEN_DIR;
    const auto& entries = enigma::test_infra::allShaderEntries();

    std::printf("[spirv_diff_test] checking %zu shader entries against goldens in %s\n",
                entries.size(), goldenDir.string().c_str());

    std::size_t failed = 0;
    for (const auto& e : entries) {
        const auto r = enigma::test_infra::diffShader(e, shaderRoot, goldenDir);
        if (!r.passed) {
            std::fprintf(stderr, "[spirv_diff_test] FAIL: %s\n", r.message.c_str());
            ++failed;
        } else {
            std::printf("[spirv_diff_test]   %s\n", r.message.c_str());
        }
    }

    if (failed == 0) {
        std::printf("[spirv_diff_test] all %zu entries pass\n", entries.size());
        return 0;
    }
    std::fprintf(stderr, "[spirv_diff_test] %zu of %zu entries failed\n",
                 failed, entries.size());
    return 1;
}

int runCapture() {
    const std::filesystem::path shaderRoot = ENIGMA_SHADER_ROOT_DIR;
    const std::filesystem::path goldenDir  = ENIGMA_SPIRV_GOLDEN_DIR;
    const auto& entries = enigma::test_infra::allShaderEntries();

    std::printf("[spirv_diff_test] capturing baseline: %zu entries -> %s\n",
                entries.size(), goldenDir.string().c_str());

    std::size_t failed = 0;
    for (const auto& e : entries) {
        if (!enigma::test_infra::captureGolden(e, shaderRoot, goldenDir)) {
            ++failed;
        }
    }
    if (failed == 0) {
        std::printf("[spirv_diff_test] wrote %zu golden blobs\n", entries.size());
        return 0;
    }
    std::fprintf(stderr, "[spirv_diff_test] baseline capture had %zu failures\n", failed);
    return 1;
}

} // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--generate-baseline") == 0
            || std::strcmp(argv[i], "--capture-baseline") == 0) {
            return runCapture();
        }
    }
    return runDiff();
}
