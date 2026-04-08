# Enigma

A self-authored C++26 rendering engine with an architectural posture
inspired by Unreal Engine 5 and Decima. Milestone 1 is a
validation-clean, resize-safe, bindless-driven triangle rendered via
Vulkan 1.3 dynamic rendering.

## Requirements

- **Windows 10 or 11** (primary verified platform)
- **CMake >= 3.28**
- **Vulkan SDK >= 1.3** (the build uses `Vulkan::shaderc_combined` for
  runtime GLSL compilation; set the `VULKAN_SDK` environment variable)
- **MSVC 17.10+** (or a Clang / GCC toolchain with C++26 preview
  support; `/std:c++latest` on MSVC, `-std=c++2c` elsewhere)
- **Internet access on first build** (volk, VMA, GLFW, and glm are
  pulled in via CMake `FetchContent` at configure time)

## Build

```sh
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Debug
./build/Debug/Enigma.exe
```

The `POST_BUILD` step copies `shaders/*.vert` and `shaders/*.frag`
next to the executable, so you can run `Enigma.exe` from any working
directory.

## Architecture

Milestone 1 subsystem layout under `src/`:

```
core/      Types, Assert, Log, Paths
platform/  Window (GLFW)
gfx/       Instance, Device, Allocator (VMA), Swapchain, FrameContext,
           DescriptorAllocator (bindless), ShaderManager (shaderc),
           Pipeline, Validation
renderer/  Renderer, TrianglePass
```

Key design commitments:

- **Vulkan 1.3 dynamic rendering only** — no `VkRenderPass`, no
  `VkFramebuffer` anywhere.
- **4-binding bindless global descriptor set**
  (sampled image / storage image / storage buffer / sampler) with
  `UPDATE_AFTER_BIND` + `PARTIALLY_BOUND` on all bindings and
  `VARIABLE_DESCRIPTOR_COUNT` on the sampler binding (the only
  variable-count slot per Vulkan spec).
- **Triangle vertex positions flow through the bindless storage
  buffer binding** (not hardcoded in shader, not a per-draw VBO).
  `nonuniformEXT` indexing is live.
- **Hybrid sync model** — binary semaphores for image acquire /
  present, timeline semaphore for CPU/GPU frame pipelining.
- **Validation-clean via counter**, not abort-on-warning. The debug
  messenger increments an atomic counter on any WARNING / ERROR; the
  `Renderer` destructor asserts the counter is zero after
  `vkDeviceWaitIdle`.
- **BDA is explicitly dropped** at this milestone — scaffold-without-
  usage violates the "bindless actually used" principle.

## Verification

Shell requirement: **POSIX-ish** (Git Bash, MSYS2, or WSL on Windows).
The verification checks use POSIX shell syntax:

```sh
# 1. Clean build
rm -rf build && cmake -S . -B build && cmake --build build

# 7. Commit hygiene
test $(git log --all --format='%B' | grep -ci 'co-authored') -eq 0 \
    || echo "FAIL: co-authored trailer present"
test $(git log --oneline | wc -l) -ge 30 \
    || echo "FAIL: commit count below fine-grained bar"

# 9. No legacy render pass API
test $(grep -rE 'VkRenderPass|VkFramebuffer|vkCreateRenderPass|vkCreateFramebuffer' src/ shaders/ | wc -l) -eq 0 \
    || echo "FAIL: legacy render pass API present"

# 10. No BDA
test $(grep -rE 'BUFFER_DEVICE_ADDRESS|bufferDeviceAddress|vkGetBufferDeviceAddress' src/ | wc -l) -eq 0 \
    || echo "FAIL: BDA references present"

# 11. nonuniformEXT usage in vertex shader (AC8)
grep -q 'nonuniformEXT' shaders/triangle.vert \
    || echo "FAIL: triangle.vert missing nonuniformEXT"
```

See `.omc/plans/enigma-triangle-plan.md` for the full verification
suite and the commit-by-commit implementation plan.
