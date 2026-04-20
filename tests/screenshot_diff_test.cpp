// screenshot_diff_test — exercises the ScreenshotDiffHarness against
// the committed golden PNG. Exit code 0 on pass, 1 on fail.
//
// Usage:
//   screenshot_diff_test                   (normal test run, compares vs golden)
//   screenshot_diff_test --capture-baseline (regenerate golden, then exit)
//
// The reference directory is resolved relative to the compiled-in
// repo root (ENIGMA_TEST_REFERENCE_DIR) so the test binary works
// regardless of CWD.

#include "infra/screenshot_diff/ScreenshotDiffHarness.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

#ifndef ENIGMA_TEST_REFERENCE_DIR
#define ENIGMA_TEST_REFERENCE_DIR ""
#endif

namespace {

constexpr const char* kSceneName = "builtin_clear_color";

std::filesystem::path referencePngPath() {
    return std::filesystem::path(ENIGMA_TEST_REFERENCE_DIR) / "builtin_clear_color.png";
}

int runDiff() {
    const std::filesystem::path ref = referencePngPath();
    const auto result = enigma::test_infra::runScreenshotDiff(kSceneName, ref, 0);
    std::printf("[screenshot_diff_test] %s\n", result.message.c_str());
    return result.passed ? 0 : 1;
}

int runCapture() {
    const std::filesystem::path out = referencePngPath();
    const bool ok = enigma::test_infra::captureBaseline(kSceneName, out);
    return ok ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--capture-baseline") == 0) {
            return runCapture();
        }
    }
    return runDiff();
}
