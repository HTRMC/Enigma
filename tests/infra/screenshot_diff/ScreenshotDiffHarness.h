// ScreenshotDiffHarness
// =====================
// Offscreen Vulkan capture + PNG reference compare for the Enigma
// micropoly pipeline (plan §3.M0b). Self-contained: builds its own
// VkInstance / VkDevice in headless mode (no surface, no window),
// so it runs on CI without a display. The harness is deliberately
// NOT coupled to enigma::gfx::Device — that class assumes a surface
// and pulls in the whole engine, which is out of scope for a unit
// test harness. Mirroring the subset of Vulkan state needed here is
// ~200 lines of boilerplate and keeps the M0b deliverable
// infrastructure-only.
//
// Principle-1 verification (ralplan-micropolygon.md §2): we need to
// prove "bit-identical when micropoly toggled off" at M3. This harness
// is the mechanism — it captures an RGBA8 image to a known buffer and
// byte-compares against a golden PNG. Default tolerance is 0 (strict
// pixel identity); relaxations must be explicit.
//
// The built-in test scene is intentionally minimal: vkCmdClearColorImage
// on an RGBA8 color target with a fixed clear color. That exercises the
// capture / readback / PNG decode / diff path — which is what M0b
// validates — without pulling in any engine shader. Downstream callers
// (M3 onward) can subclass by adding their own render function between
// begin/end; for now the public API is a single runScreenshotDiff() that
// captures the built-in scene.

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

namespace enigma::test_infra {

// Fixed dimensions for the built-in clear-color test scene. Small
// enough that the golden PNG stays tiny; large enough to exercise
// row-padding edge cases in vkCmdCopyImageToBuffer (bufferRowLength).
constexpr std::uint32_t kScreenshotDiffWidth  = 128;
constexpr std::uint32_t kScreenshotDiffHeight = 128;

// Result of a single harness invocation. On success, `passed` is true
// and `diffPixels` is 0 (or below tolerance). On any failure —
// capture, PNG decode, size mismatch, pixel delta — `passed` is false
// and `message` contains a human-readable reason.
struct ScreenshotDiffResult {
    bool        passed     = false;
    std::uint32_t width     = 0;
    std::uint32_t height    = 0;
    std::uint64_t diffPixels = 0; // # pixels differing by more than tolerance in any channel
    std::uint32_t maxDelta   = 0; // largest per-channel absolute delta observed
    std::string  message;
};

// Run the built-in scene (solid-color clear), read back pixels, and
// compare to `referencePng`. Fails loudly (passed=false, message set)
// if the reference file does not exist — callers run the
// captureBaseline() path first to produce it.
//
// `pixelTolerance` is a per-channel absolute delta threshold. The
// default of 0 means strict identity: any byte difference fails.
// Non-zero tolerance is provided only for downstream harnesses that
// layer lossy post-process (tonemap, bloom, ...) on top; M0b uses 0.
ScreenshotDiffResult runScreenshotDiff(const std::string& testSceneName,
                                       const std::filesystem::path& referencePng,
                                       std::uint32_t pixelTolerance = 0);

// Capture the built-in scene once and write it to `outPng`. Used to
// bootstrap the golden reference file (first run, or after an
// explicitly-justified visual change). Returns true on success.
bool captureBaseline(const std::string& testSceneName,
                     const std::filesystem::path& outPng);

} // namespace enigma::test_infra
