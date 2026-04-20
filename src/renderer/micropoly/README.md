# Micropoly subsystem (M0a)

Scaffolding landed in Milestone **M0a** of the micropolygon roadmap
(`.omc/plans/ralplan-micropolygon.md`). Only the feature gate, device probes,
capability classifier, and an empty pass shell exist at this milestone.
Real work starts in **M2/M3** (page store + software raster).

## HW Capability Matrix

The classifier in `MicropolyCapability.cpp` maps five probed feature bits
(`meshShader`, `atomicInt64`, `imageInt64`, `sparseResidency`, `rayTracing`)
onto one of six rows below. The selected row drives a single
`"micropoly: <status>"` log line at boot.

| Row | Mesh shader | atomic_int64 | image_int64 | Sparse residency | RT | Micropoly status                                                  |
|-----|-------------|--------------|-------------|------------------|----|-------------------------------------------------------------------|
| (a) | yes         | yes          | yes         | yes              | yes| **Full:** SW+HW raster + RT shadows + VSM available (VSM unused when RT on) |
| (b) | yes         | yes          | yes         | no               | yes| **RT shadow only:** SW+HW raster + RT shadows; no VSM             |
| (c) | yes         | yes          | yes         | yes              | no | **VSM shadow:** SW+HW raster + VSM; non-RT tier (8 ms soft)       |
| (d) | yes         | yes          | yes         | no               | no | **Geometry but no shadows:** SW+HW raster; micropoly content casts no shadows; logged as degraded |
| (e) | yes         | yes          | no          | any              | any| **HW-only micropoly:** HW raster path only; SW classification falls back to HW |
| (f) | missing mesh shader *or* missing atomic_int64 | - | - | - | - | Micropoly disabled; engine boots on existing renderer |

Row (f) is a hard gate — if either `VK_EXT_mesh_shader` or
`VkPhysicalDeviceShaderAtomicInt64Features::shaderBufferInt64Atomics` is
absent, neither raster path is viable and the subsystem refuses to run.

## Visibility-buffer format decision

Locked at M0a in `MicropolyPass`'s constructor per plan §3.M0a:

- **Preferred:** `VK_FORMAT_R64_UINT` when the device reports storage-image
  usage support via `vkGetPhysicalDeviceImageFormatProperties`.
- **Fallback:** `VK_FORMAT_R32G32_UINT`, aliased for u64 atomic-min via the
  `SPV_EXT_shader_image_int64` capability exposed by
  `VK_EXT_shader_image_atomic_int64`. M3 validates real atomic behaviour on
  AMD drivers (risk R3 blacklist fallback in plan §5).
- **Unavailable:** `VK_FORMAT_UNDEFINED`. The pass is guaranteed inactive
  on this device regardless of `MicropolyConfig::enabled`.

## Files in this directory

| File                      | Role                                                          |
|---------------------------|---------------------------------------------------------------|
| `MicropolyConfig.h`       | Runtime flags (enabled, forceSW, forceHW, pageCacheMB, debugOverlay) |
| `MicropolyCapability.h/.cpp` | Device probe + `HwMatrixRow` classifier + boot status string |
| `MicropolyPass.h/.cpp`    | Empty pass shell; owns vis-buffer format + (future) VkImage   |
| `README.md`               | This file                                                     |

The unit test living in `tests/micropoly_capability_test.cpp` exercises all
six matrix rows by synthetically masking the input bits.

## Invariants — do not break

1. **Disabled = bit-identical** (Principle 1, plan §0.5.1): when
   `MicropolyConfig::enabled == false`, `MicropolyPass::record()` must not
   issue any command, allocate any resource, or update any descriptor.
2. **No shader touching in M0a** (Preamble 0.5.2): this directory must not
   contain `.hlsl` files before Milestone M3.
3. **No existing pass mutated** (Preamble 0.5.3): HiZ, GpuCull,
   VisibilityBuffer, MaterialEval, Lighting, RT passes, Sky, PostProcess
   are off-limits at M0a.
