# SPIR-V golden blobs

This directory holds one `.spv` blob per (HLSL file, entry point,
stage) triple, captured at HEAD by the `SpirvDiffHarness`. The
`spirv_diff_test` build target recompiles every shader with the
engine's exact DXC flag list (mirrored from
`src/gfx/ShaderManager.cpp`) and byte-compares the output against
these goldens.

Naming convention: `<file-stem>.<entry>.<stage>.spv`, where the
stage token is one of `vs ps cs as ms rgen rchit rmiss`. Paths
containing `/` are flattened to `_` so this directory stays flat.

## Regenerating goldens

```
spirv_diff_test --generate-baseline
```

(Or equivalently, `--capture-baseline`.)

This recompiles every entry listed in the manifest
(`SpirvDiffHarness.cpp :: buildManifest()`) and overwrites the
matching golden. Re-running `spirv_diff_test` (no flag) after the
regeneration verifies identity.

## CI gate — Principle 1

Per `.omc/plans/ralplan-micropolygon.md` §3.M0b and the Principle 1
invariant (bit-identity when micropoly is toggled off), **PRs that
perturb any golden here without explicit justification must fail
CI**. At M3 the micropoly `MP_ENABLE` spec constant is introduced;
with it false, the SPIR-V must remain byte-identical to the goldens
in this directory.

Running the gate:

```
spirv_diff_test
```

Exit code 0 iff every entry matches byte-for-byte (or falls within
the allowlist documented in `SpirvDiffHarness.cpp`). Exit 1 on any
mismatch, missing golden, or compile failure.

## Allowlist policy

The allowlist (`SpirvDiffHarness.cpp :: kAllowlist`) is currently
**empty**. Under the engine's DXC invocation
(`-E <entry> -T <profile> -spirv -fspv-target-env=vulkan1.3
-fvk-use-dx-layout -Zpc -I <shaderDir> -HV 2021 -O3`), the output
is bit-deterministic for the pinned Vulkan SDK version. These flags
mirror `ShaderManager::tryCompile` at its `#if ENIGMA_DEBUG` = 0
arm (the shipped/canonical build defines NDEBUG so ENIGMA_DEBUG
resolves to 0; see `src/core/Assert.h:22-26` and
`src/gfx/ShaderManager.cpp:145-151`). If a future SDK bump
introduces non-functional drift (e.g. a timestamp or compiler-
version blob), add the tolerated byte range to `kAllowlist` with
an inline comment justifying the exception. Empty allowlist is the
default; keeping it tight is what makes the Principle-1 guarantee
meaningful.

## DXC flag drift

If `src/gfx/ShaderManager.cpp` gains a new DXC flag (or drops one),
the harness mirror in `SpirvDiffHarness.cpp :: compileEntry()` must
be updated in the same commit and every golden regenerated. A
subtle flag drift would produce "passing" goldens that no longer
represent the engine's runtime SPIR-V, which defeats the
Principle-1 guarantee.

## Shader coverage

63 goldens at HEAD, spanning every `.hlsl` compiled by the engine's
ShaderManager (raster VS/PS, mesh/task, compute, SMAA, atmosphere,
ray tracing). The full manifest is in
`SpirvDiffHarness.cpp :: buildManifest()`; adding a new shader is a
deliberate act that also updates that list.
