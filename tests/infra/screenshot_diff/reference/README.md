# Screenshot-diff reference images

This directory holds PNG golden images captured by the
`ScreenshotDiffHarness` (see `../ScreenshotDiffHarness.h`). Each
golden is the byte-exact expected output of a specific test scene, and
the `screenshot_diff_test` build target compares every run against
the goldens here at tolerance 0 (strict identity).

## How goldens are produced

Goldens are **not committed as placeholders**. They are produced by
running the harness in baseline-capture mode, then committed. This
keeps the reference synchronized with the current shader + capture
pipeline.

From `build/` (or wherever CMake produced the test binary):

```
screenshot_diff_test --capture-baseline
```

The tool captures the built-in test scene and writes
`builtin_clear_color.png` to this directory. Re-running
`screenshot_diff_test` (no flag) after the capture verifies
bit-identity against the just-written file.

## CI gate

PRs that perturb the golden PNGs (or the SPIR-V goldens under
`../spirv_diff/golden/`) **without an explicit, justified bump of the
golden set** must fail CI. The micropoly milestone plan
(`.omc/plans/ralplan-micropolygon.md`, Principle 1) requires
bit-identity-when-disabled; losing a golden round-trip silently would
violate that guarantee.

Updating a golden is a deliberate act: delete the affected PNG, re-run
`screenshot_diff_generate_baseline`, and commit the new PNG alongside
the commit that justifies the visual change.

## Current goldens

- `builtin_clear_color.png` — 128x128, fixed clear color
  (rgba8 = 32, 96, 160, 255). Produced by the `ScreenshotDiffHarness`
  built-in scene; exercises capture / readback / PNG round-trip.
