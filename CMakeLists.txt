cmake_minimum_required(VERSION 3.28)
project(Enigma LANGUAGES CXX)

# C++26 requested. CMake 4.2's MSVC compiler module currently caps the
# tracked CXX_STANDARD_LATEST at 23 (no CXX26 dialect flag), so we set the
# tracked standard to 23 here and opt into C++26-preview features at the
# target level below via `/std:c++latest` on MSVC and `-std=c++2c` elsewhere.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif()

# -----------------------------------------------------------------------------
# Dependencies (FetchContent, SHA-pinned)
# -----------------------------------------------------------------------------
include(${CMAKE_SOURCE_DIR}/cmake/Dependencies.cmake)

# -----------------------------------------------------------------------------
# Vulkan SDK — provides headers, validation layer runtime, and DXC (the
# DirectX Shader Compiler with SPIR-V backend enabled, via the
# Vulkan::dxc_lib imported target). The engine is HLSL-only after the
# milestone 2 migration; the legacy shaderc path has been removed in full.
# Minimum version 1.3 is required for dynamic rendering, descriptor
# indexing, sync2, and timeline semaphores as core features. The `dxc`
# component was added to CMake's FindVulkan module in 3.25, so CMake 3.28
# (the minimum above) supports it natively.
# -----------------------------------------------------------------------------
find_package(Vulkan 1.3 REQUIRED COMPONENTS dxc)

# -----------------------------------------------------------------------------
# Target
# -----------------------------------------------------------------------------
add_executable(Enigma)

target_sources(Enigma PRIVATE
    main.cpp
    src/core/Log.cpp
    src/core/Paths.cpp
    src/platform/Window.cpp
    src/input/Input.cpp
    src/engine/Clock.cpp
    src/engine/Application.cpp
    src/engine/Engine.cpp
    src/scene/Camera.cpp
    src/scene/CameraController.cpp
    src/scene/FollowCamera.cpp
    src/scene/Scene.cpp
    src/world/HeightmapLoader.cpp
    src/world/CdlodTerrain.cpp
    src/asset/GltfLoader.cpp
    src/asset/MpAssetFormat.cpp
    src/asset/MpAssetReader.cpp
    src/renderer/Renderer.cpp
    src/renderer/MeshPass.cpp
    src/renderer/LightingPass.cpp
    src/gfx/Instance.cpp
    src/gfx/Validation.cpp
    src/gfx/Device.cpp
    src/gfx/DeviceAdopt.cpp
    src/gfx/Allocator.cpp
    src/gfx/AllocatorAdopt.cpp
    src/gfx/Swapchain.cpp
    src/gfx/FrameContext.cpp
    src/gfx/DescriptorAllocator.cpp
    src/gfx/GpuProfiler.cpp
    src/gfx/RenderGraph.cpp
    src/gfx/ShaderManager.cpp
    src/gfx/ShaderHotReload.cpp
    src/gfx/UploadContext.cpp
    src/gfx/Pipeline.cpp
    src/gfx/AccelerationStructure.cpp
    src/gfx/RTPipeline.cpp
    src/renderer/RTReflectionPass.cpp
    src/renderer/RTGIPass.cpp
    src/renderer/RTShadowPass.cpp
    src/renderer/WetRoadPass.cpp
    src/renderer/Denoiser.cpp
    src/renderer/AtmospherePass.cpp
    src/renderer/SkyBackgroundPass.cpp
    src/renderer/PostProcessPass.cpp
    src/renderer/TrianglePass.cpp
    src/renderer/MeshletBuilder.cpp
    src/renderer/GpuMeshletBuffer.cpp
    src/renderer/GpuSceneBuffer.cpp
    src/renderer/IndirectDrawBuffer.cpp
    src/renderer/HiZPass.cpp
    src/renderer/GpuCullPass.cpp
    src/renderer/VisibilityBufferPass.cpp
    src/renderer/DebugVisualizationPass.cpp
    src/renderer/MaterialEvalPass.cpp
    src/renderer/ClusteredForwardPass.cpp
    src/renderer/SMAAPass.cpp
    src/renderer/micropoly/MicropolyCapability.cpp
    src/renderer/micropoly/MicropolyPass.cpp
    src/renderer/micropoly/MicropolyCullPass.cpp
    src/renderer/micropoly/MicropolyRasterPass.cpp
    src/renderer/micropoly/MicropolyRasterPassError.cpp
    src/renderer/micropoly/MicropolySwRasterPass.cpp
    src/renderer/micropoly/MicropolySwRasterPassError.cpp
    src/renderer/micropoly/MicropolyBlasManager.cpp
    src/renderer/micropoly/MicropolyBlasManagerError.cpp
    src/renderer/micropoly/ResidencyManager.cpp
    src/renderer/micropoly/AsyncIOWorker.cpp
    src/renderer/micropoly/PageCache.cpp
    src/renderer/micropoly/RequestQueue.cpp
    src/renderer/micropoly/MicropolyStreaming.cpp
    src/renderer/UpscalerFactory.cpp
    src/renderer/UpscalerSettings.cpp
    src/renderer/DLSSUpscaler.cpp
    src/renderer/XeSSUpscaler.cpp
    src/renderer/FSRUpscaler.cpp
    src/physics/PhysicsWorld.cpp
    src/physics/PhysicsShapes.cpp
    src/physics/PhysicsDebugRenderer.cpp
    src/physics/VehicleConfig.cpp
    src/physics/VehicleController.cpp
    src/physics/PhysicsInterpolation.cpp
    src/physics/CrumpleZone.cpp
    src/physics/DeformationSystem.cpp
    src/ecs/World.cpp
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
    src/gfx/ImGuiLayer.cpp
    src/ui/MicropolySettingsPanel.cpp
)

target_include_directories(Enigma PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(Enigma PRIVATE
    volk::volk
    GPUOpen::VulkanMemoryAllocator
    glfw
    glm::glm
    Vulkan::dxc_lib
    fastgltf
    Jolt
    libzstd_static
)

# -----------------------------------------------------------------------------
# Mark third-party include directories as SYSTEM so their (often noisy)
# warnings do not cascade into the Enigma target's warnings-as-errors policy.
# FetchContent populates these *_SOURCE_DIR variables after the
# FetchContent_MakeAvailable() calls in Dependencies.cmake.
# -----------------------------------------------------------------------------
target_include_directories(Enigma SYSTEM PRIVATE
    ${volk_SOURCE_DIR}
    ${VulkanMemoryAllocator_SOURCE_DIR}/include
    ${glfw_SOURCE_DIR}/include
    ${glm_SOURCE_DIR}
    ${fastgltf_SOURCE_DIR}/include
    ${stb_SOURCE_DIR}
    ${joltphysics_SOURCE_DIR}
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)

# volk loads Vulkan at runtime; we only need the Vulkan headers transitively
# through volk. We do NOT link Vulkan::Vulkan directly.
# IMGUI_IMPL_VULKAN_USE_VOLK: tells imgui_impl_vulkan.cpp to include volk.h
# instead of vulkan.h and to skip the LoadFunctions() assertion — required
# because VK_NO_PROTOTYPES suppresses all extern Vulkan declarations.
target_compile_definitions(Enigma PRIVATE
    VK_NO_PROTOTYPES
    IMGUI_IMPL_VULKAN_USE_VOLK
    $<$<PLATFORM_ID:Windows>:VK_USE_PLATFORM_WIN32_KHR>
    $<$<PLATFORM_ID:Windows>:NOMINMAX>
    GLFW_INCLUDE_NONE
    ENIGMA_SHADER_SOURCE_DIR="${CMAKE_SOURCE_DIR}/shaders"
    $<$<CONFIG:Release>:JPH_NO_DEBUG>
)

# Warnings-as-errors scoped to the Enigma target only. Third-party headers
# pulled in by FetchContent are marked SYSTEM in a later step so their
# warnings do not cascade into this strict policy.
if(MSVC)
    target_compile_options(Enigma PRIVATE
        /std:c++latest /W4 /WX /permissive- /Zc:preprocessor
        # C4127 "conditional expression is constant" fires on ENIGMA_ASSERT
        # macros (do-while(0) and compile-time false conditions) — those
        # are legitimate patterns, not real warnings.
        /wd4127
    )
else()
    target_compile_options(Enigma PRIVATE
        -std=c++2c
        -Wall -Wextra -Wpedantic -Werror
        -Wno-missing-field-initializers
    )
endif()

# -----------------------------------------------------------------------------
# Shader copy: make shaders/* available next to the built executable so that
# Paths::shaderDir() (resolved relative to argv[0]) can find them regardless
# of the current working directory.
# -----------------------------------------------------------------------------
add_custom_command(TARGET Enigma POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/shaders
            $<TARGET_FILE_DIR:Enigma>/shaders
    COMMENT "Copying shaders next to Enigma executable"
)

# -----------------------------------------------------------------------------
# Milestone 0: P2996 feasibility probe
# Compile-only target — not linked against Vulkan/Jolt/etc.
# PASS (exits 0): proceed with P2996 path for Phase 1 ECS Core
# FAIL (compile error): commit to ENIGMA_COMPONENT macro path
# -----------------------------------------------------------------------------
add_executable(p2996_probe tests/p2996_probe.cpp)
if(MSVC)
    target_compile_options(p2996_probe PRIVATE /std:c++latest /W4 /WX /permissive- /Zc:preprocessor)
else()
    target_compile_options(p2996_probe PRIVATE -std=c++2c -Wall -Wextra -Werror)
endif()

# -----------------------------------------------------------------------------
# ECS compile test — validates the ECS core library compiles and passes
# basic runtime checks. Does not link against Vulkan/Jolt/etc.
# -----------------------------------------------------------------------------
add_executable(ecs_compile_test tests/ecs_compile_test.cpp src/ecs/World.cpp)
target_include_directories(ecs_compile_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(ecs_compile_test PRIVATE glm::glm)
target_include_directories(ecs_compile_test SYSTEM PRIVATE ${glm_SOURCE_DIR})
if(MSVC)
    target_compile_options(ecs_compile_test PRIVATE /std:c++latest /W4 /WX /permissive- /Zc:preprocessor /wd4127)
else()
    target_compile_options(ecs_compile_test PRIVATE -std=c++2c -Wall -Wextra -Wpedantic -Werror)
endif()

# -----------------------------------------------------------------------------
# Micropoly capability classifier unit test (M0a exit criterion).
# Verifies the HW matrix classifier maps every synthetic feature-mask
# permutation to the correct row. Does NOT link Vulkan — the classifier
# itself is a pure function; only the probe wrapper touches Vulkan.
# -----------------------------------------------------------------------------
add_executable(micropoly_capability_test
    tests/micropoly_capability_test.cpp
    src/renderer/micropoly/MicropolyCapability.cpp)
target_include_directories(micropoly_capability_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
# The classifier-only path doesn't touch Vulkan, but MicropolyCapability.cpp
# includes gfx/Device.h transitively (volk.h, which in turn includes the
# Vulkan headers). Add the Vulkan SDK include dir (from find_package) and
# the volk source dir so the test TU can parse those headers — we do NOT
# actually call any Vulkan function or link Vulkan.
target_link_libraries(micropoly_capability_test PRIVATE Vulkan::Headers)
target_include_directories(micropoly_capability_test SYSTEM PRIVATE ${volk_SOURCE_DIR})
target_compile_definitions(micropoly_capability_test PRIVATE VK_NO_PROTOTYPES)
if(MSVC)
    target_compile_options(micropoly_capability_test PRIVATE /std:c++latest /W4 /WX /permissive- /Zc:preprocessor /wd4127)
else()
    target_compile_options(micropoly_capability_test PRIVATE -std=c++2c -Wall -Wextra -Wpedantic -Werror)
endif()

# -----------------------------------------------------------------------------
# ECS benchmark — AC 1.2: 1M entity Query<Position,Velocity>.for_each() < 2ms (DRAM-bound)
# -----------------------------------------------------------------------------
add_executable(ecs_bench tests/ecs_bench.cpp src/ecs/World.cpp)
target_include_directories(ecs_bench PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(ecs_bench PRIVATE glm::glm)
target_include_directories(ecs_bench SYSTEM PRIVATE ${glm_SOURCE_DIR})
if(MSVC)
    target_compile_options(ecs_bench PRIVATE /std:c++latest /W4 /WX /permissive- /Zc:preprocessor /wd4127 /O2)
else()
    target_compile_options(ecs_bench PRIVATE -std=c++2c -Wall -Wextra -Wpedantic -Werror -O2)
endif()

# -----------------------------------------------------------------------------
# Micropoly M0b: screenshot-diff + SPIR-V-diff test infrastructure
# ---------------------------------------------------------------------------
# These two harnesses implement the Principle-1 verification machinery
# described in .omc/plans/ralplan-micropolygon.md §3.M0b: without them,
# M3 cannot prove "bit-identical when toggled off" because there is no
# frozen baseline to diff against.
#
# - screenshot_diff_test  : headless Vulkan capture of a built-in scene,
#                           byte-compare to a golden PNG (tolerance 0).
# - spirv_diff_test       : recompile every .hlsl with the same DXC flag
#                           list as ShaderManager.cpp, byte-compare to
#                           a golden .spv per (file, entry, stage).
#
# Both accept a --generate-baseline / --capture-baseline flag that
# regenerates the goldens on demand; the first run of each harness is
# a baseline capture, subsequent runs enforce identity.
# -----------------------------------------------------------------------------

add_executable(screenshot_diff_test
    tests/screenshot_diff_test.cpp
    tests/infra/screenshot_diff/ScreenshotDiffHarness.cpp)
target_include_directories(screenshot_diff_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests)
# Volk is the engine's Vulkan loader; the harness reuses it so it
# matches the main binary's runtime behavior. stb_image + stb_image_write
# both live in the already-fetched stb repo — we stamp the impls
# inside ScreenshotDiffHarness.cpp to avoid linking a separate archive.
target_link_libraries(screenshot_diff_test PRIVATE volk::volk)
target_include_directories(screenshot_diff_test SYSTEM PRIVATE
    ${volk_SOURCE_DIR}
    ${stb_SOURCE_DIR})
target_compile_definitions(screenshot_diff_test PRIVATE
    VK_NO_PROTOTYPES
    $<$<PLATFORM_ID:Windows>:VK_USE_PLATFORM_WIN32_KHR>
    $<$<PLATFORM_ID:Windows>:NOMINMAX>
    ENIGMA_TEST_REFERENCE_DIR="${CMAKE_SOURCE_DIR}/tests/infra/screenshot_diff/reference")
if(MSVC)
    target_compile_options(screenshot_diff_test PRIVATE
        /std:c++latest /W4 /WX /permissive- /Zc:preprocessor /wd4127)
else()
    target_compile_options(screenshot_diff_test PRIVATE
        -std=c++2c -Wall -Wextra -Wpedantic -Werror
        -Wno-missing-field-initializers)
endif()

add_executable(spirv_diff_test
    tests/spirv_diff_test.cpp
    tests/infra/spirv_diff/SpirvDiffHarness.cpp)
target_include_directories(spirv_diff_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests)
# DXC via Vulkan::dxc_lib (same imported target the main engine links).
# No Vulkan loader needed — this harness stops at the SPIR-V blob.
target_link_libraries(spirv_diff_test PRIVATE Vulkan::dxc_lib)
target_compile_definitions(spirv_diff_test PRIVATE
    $<$<PLATFORM_ID:Windows>:NOMINMAX>
    ENIGMA_SHADER_ROOT_DIR="${CMAKE_SOURCE_DIR}/shaders"
    ENIGMA_SPIRV_GOLDEN_DIR="${CMAKE_SOURCE_DIR}/tests/infra/spirv_diff/golden")
if(MSVC)
    target_compile_options(spirv_diff_test PRIVATE
        /std:c++latest /W4 /WX /permissive- /Zc:preprocessor /wd4127)
else()
    target_compile_options(spirv_diff_test PRIVATE
        -std=c++2c -Wall -Wextra -Wpedantic -Werror)
endif()

# Copy dxcompiler.dll next to both test binaries — same reason as the
# main Enigma target (Windows DLL search-order pollution).
if(WIN32)
    add_custom_command(TARGET spirv_diff_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$ENV{VULKAN_SDK}/Bin/dxcompiler.dll"
                "$<TARGET_FILE_DIR:spirv_diff_test>/dxcompiler.dll"
        COMMENT "Copying dxcompiler.dll next to spirv_diff_test")
endif()

# -----------------------------------------------------------------------------
# M3.1: mp_vis_pack_compile_test — smoke test that shaders/micropoly/mp_vis_pack.hlsl
# compiles cleanly under DXC with the engine's exact flag set. The file is
# include-only and has no standalone golden blob — this test writes a tiny
# wrapper that exercises PackMpVis64/UnpackMpVis64 and verifies DXC produces
# a non-empty SPIR-V object. Reuses the SpirvDiffHarness DXC plumbing.
# -----------------------------------------------------------------------------
add_executable(mp_vis_pack_compile_test
    tests/mp_vis_pack_compile_test.cpp
    tests/infra/spirv_diff/SpirvDiffHarness.cpp)
target_include_directories(mp_vis_pack_compile_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(mp_vis_pack_compile_test PRIVATE Vulkan::dxc_lib)
target_compile_definitions(mp_vis_pack_compile_test PRIVATE
    $<$<PLATFORM_ID:Windows>:NOMINMAX>
    ENIGMA_SHADER_ROOT_DIR="${CMAKE_SOURCE_DIR}/shaders"
    ENIGMA_SPIRV_GOLDEN_DIR="${CMAKE_SOURCE_DIR}/tests/infra/spirv_diff/golden")
if(MSVC)
    target_compile_options(mp_vis_pack_compile_test PRIVATE
        /std:c++latest /W4 /WX /permissive- /Zc:preprocessor /wd4127)
else()
    target_compile_options(mp_vis_pack_compile_test PRIVATE
        -std=c++2c -Wall -Wextra -Wpedantic -Werror)
endif()
if(WIN32)
    add_custom_command(TARGET mp_vis_pack_compile_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$ENV{VULKAN_SDK}/Bin/dxcompiler.dll"
                "$<TARGET_FILE_DIR:mp_vis_pack_compile_test>/dxcompiler.dll"
        COMMENT "Copying dxcompiler.dll next to mp_vis_pack_compile_test")
endif()

# Copy bundled assets (default glTF model) next to the executable.
if(EXISTS "${CMAKE_SOURCE_DIR}/assets")
    add_custom_command(TARGET Enigma POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_SOURCE_DIR}/assets
                $<TARGET_FILE_DIR:Enigma>/assets
        COMMENT "Copying assets next to Enigma executable"
    )
endif()

# -----------------------------------------------------------------------------
# Vulkan SDK dxcompiler.dll copy: Windows has upwards of 20 copies of
# dxcompiler.dll floating around (Windows 10/11 SDK, Visual Studio, Epic,
# Unity, Chrome/Edge/Firefox, Steam, RenderDoc, ...). All of them except
# the Vulkan SDK's are Microsoft's D3D-only build WITHOUT SPIR-V codegen —
# they emit `SPIR-V CodeGen not available. Please recompile with
# -DENABLE_SPIRV_CODEGEN=ON.` when called with `-spirv`. Windows' DLL
# search order picks whichever one it finds first on PATH, so without an
# explicit copy the Enigma binary is at the mercy of the dev environment's
# PATH and loads whatever random D3D-only version wins the race.
#
# The fix is to copy the Vulkan-SPIR-V-enabled dxcompiler.dll next to the
# exe before launch — Windows' DLL search order puts the application
# directory FIRST, ahead of System32 / PATH / everything else. Our version
# is then guaranteed to win regardless of dev-machine pollution.
# -----------------------------------------------------------------------------
add_custom_command(TARGET Enigma POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$ENV{VULKAN_SDK}/Bin/dxcompiler.dll"
            "$<TARGET_FILE_DIR:Enigma>/dxcompiler.dll"
    COMMENT "Copying Vulkan SDK dxcompiler.dll next to Enigma executable (SPIR-V-enabled)"
)

# -----------------------------------------------------------------------------
# Milestone 1a: enigma-mpbake — offline Micropoly Asset (.mpa) bake tool.
#
# Pure CPU, no Vulkan, no volk. Links meshoptimizer + METIS + zstd + fastgltf
# + glm. Shares the `src/asset/MpAssetFormat.cpp` translation unit with the
# main Enigma target so header/writer/reader agree on the on-disk layout by
# construction.
#
# M1a scope: stubbed-out pipeline stages that parse args and print a scaffold
# notice; M1b wires `meshopt_buildMeshlets` + `METIS_PartGraphKway`; M1c
# wires the zstd page encoder and round-trip test.
# -----------------------------------------------------------------------------
add_executable(enigma-mpbake
    tools/mpbake/main.cpp
    tools/mpbake/GltfIngest.cpp
    tools/mpbake/ClusterBuilder.cpp
    tools/mpbake/DagBuilder.cpp
    tools/mpbake/PageWriter.cpp
    tools/mpbake/Simplify.cpp
    src/asset/MpAssetFormat.cpp
)
target_include_directories(enigma-mpbake PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tools/mpbake
)
target_include_directories(enigma-mpbake SYSTEM PRIVATE
    ${glm_SOURCE_DIR}
    ${fastgltf_SOURCE_DIR}/include
)
target_link_libraries(enigma-mpbake PRIVATE
    fastgltf
    glm::glm
    meshoptimizer
    metis
    libzstd_static
)
target_compile_definitions(enigma-mpbake PRIVATE
    $<$<PLATFORM_ID:Windows>:NOMINMAX>
    IDXTYPEWIDTH=32
    REALTYPEWIDTH=32
)
if(MSVC)
    target_compile_options(enigma-mpbake PRIVATE
        /std:c++latest /W4 /WX /permissive- /Zc:preprocessor /wd4127)
else()
    target_compile_options(enigma-mpbake PRIVATE
        -std=c++2c -Wall -Wextra -Wpedantic -Werror
        -Wno-missing-field-initializers)
endif()

# -----------------------------------------------------------------------------
# M1b.1: gltf_ingest_test — unit test for the offline glTF ingest stage.
# Exercises success + FileNotFound + GltfParseFailed branches against the
# shipped assets/DamagedHelmet.glb. CPU-only; no Vulkan.
# -----------------------------------------------------------------------------
add_executable(gltf_ingest_test
    tests/gltf_ingest_test.cpp
    tools/mpbake/GltfIngest.cpp
)
target_include_directories(gltf_ingest_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tools
)
target_include_directories(gltf_ingest_test SYSTEM PRIVATE
    ${glm_SOURCE_DIR}
    ${fastgltf_SOURCE_DIR}/include
)
target_link_libraries(gltf_ingest_test PRIVATE
    fastgltf
    glm::glm
)
target_compile_definitions(gltf_ingest_test PRIVATE
    $<$<PLATFORM_ID:Windows>:NOMINMAX>
)
if(MSVC)
    target_compile_options(gltf_ingest_test PRIVATE
        /std:c++latest /W4 /WX /permissive- /Zc:preprocessor /wd4127)
else()
    target_compile_options(gltf_ingest_test PRIVATE
        -std=c++2c -Wall -Wextra -Wpedantic -Werror
        -Wno-missing-field-initializers)
endif()

# -----------------------------------------------------------------------------
# M1b.2: cluster_builder_test — unit test for the offline cluster-builder
# stage. Exercises the happy path (DamagedHelmet.glb, full ingest → cluster
# pipeline, determinism check) plus EmptyInput and IndexOutOfRange error
# branches. CPU-only; no Vulkan. Reuses GltfIngest.cpp + ClusterBuilder.cpp
# from tools/mpbake.
# -----------------------------------------------------------------------------
add_executable(cluster_builder_test
    tests/cluster_builder_test.cpp
    tools/mpbake/GltfIngest.cpp
    tools/mpbake/ClusterBuilder.cpp
)
target_include_directories(cluster_builder_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tools
)
target_include_directories(cluster_builder_test SYSTEM PRIVATE
    ${glm_SOURCE_DIR}
    ${fastgltf_SOURCE_DIR}/include
)
target_link_libraries(cluster_builder_test PRIVATE
    fastgltf
    glm::glm
    meshoptimizer
)
target_compile_definitions(cluster_builder_test PRIVATE
    $<$<PLATFORM_ID:Windows>:NOMINMAX>
)
if(MSVC)
    target_compile_options(cluster_builder_test PRIVATE
        /std:c++latest /W4 /WX /permissive- /Zc:preprocessor /wd4127)
else()
    target_compile_options(cluster_builder_test PRIVATE
        -std=c++2c -Wall -Wextra -Wpedantic -Werror
        -Wno-missing-field-initializers)
endif()

# -----------------------------------------------------------------------------
# M1b.3: simplify_test — unit test for the offline simplify stage.
# Exercises the happy path (DamagedHelmet.glb ingest + cluster + take first N
# clusters as a synthetic group + simplify), EmptyGroup, OptionsOutOfRange,
# and determinism (build twice + hash) error branches. CPU-only; no Vulkan.
# Reuses GltfIngest.cpp + ClusterBuilder.cpp + Simplify.cpp from tools/mpbake.
# -----------------------------------------------------------------------------
add_executable(simplify_test
    tests/simplify_test.cpp
    tools/mpbake/GltfIngest.cpp
    tools/mpbake/ClusterBuilder.cpp
    tools/mpbake/Simplify.cpp
)
target_include_directories(simplify_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tools
)
target_include_directories(simplify_test SYSTEM PRIVATE
    ${glm_SOURCE_DIR}
    ${fastgltf_SOURCE_DIR}/include
)
target_link_libraries(simplify_test PRIVATE
    fastgltf
    glm::glm
    meshoptimizer
)
target_compile_definitions(simplify_test PRIVATE
    $<$<PLATFORM_ID:Windows>:NOMINMAX>
)
if(MSVC)
    target_compile_options(simplify_test PRIVATE
        /std:c++latest /W4 /WX /permissive- /Zc:preprocessor /wd4127)
else()
    target_compile_options(simplify_test PRIVATE
        -std=c++2c -Wall -Wextra -Wpedantic -Werror
        -Wno-missing-field-initializers)
endif()

# -----------------------------------------------------------------------------
# M1b.4: dag_builder_test — unit test for the offline DAG-builder stage.
# Exercises the happy path (DamagedHelmet.glb full ingest -> cluster -> DAG),
# EmptyInput, OptionsOutOfRange, and determinism (build twice + hash) branches.
# CPU-only; no Vulkan. Reuses all four mpbake source TUs.
# -----------------------------------------------------------------------------
add_executable(dag_builder_test
    tests/dag_builder_test.cpp
    tools/mpbake/GltfIngest.cpp
    tools/mpbake/ClusterBuilder.cpp
    tools/mpbake/Simplify.cpp
    tools/mpbake/DagBuilder.cpp
)
target_include_directories(dag_builder_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tools
)
target_include_directories(dag_builder_test SYSTEM PRIVATE
    ${glm_SOURCE_DIR}
    ${fastgltf_SOURCE_DIR}/include
)
target_link_libraries(dag_builder_test PRIVATE
    fastgltf
    glm::glm
    meshoptimizer
    metis
)
target_compile_definitions(dag_builder_test PRIVATE
    $<$<PLATFORM_ID:Windows>:NOMINMAX>
    IDXTYPEWIDTH=32
    REALTYPEWIDTH=32
)
if(MSVC)
    target_compile_options(dag_builder_test PRIVATE
        /std:c++latest /W4 /WX /permissive- /Zc:preprocessor /wd4127)
else()
    target_compile_options(dag_builder_test PRIVATE
        -std=c++2c -Wall -Wextra -Wpedantic -Werror
        -Wno-missing-field-initializers)
endif()

# -----------------------------------------------------------------------------
# M1c: page_writer_test — round-trip the full offline pipeline and verify
# that the produced `.mpa` decompresses to data byte-identical to the
# in-memory DagResult. Also covers header-magic / version corruption and
# two-run determinism of the writer. CPU-only.
# -----------------------------------------------------------------------------
add_executable(page_writer_test
    tests/page_writer_test.cpp
    tools/mpbake/GltfIngest.cpp
    tools/mpbake/ClusterBuilder.cpp
    tools/mpbake/Simplify.cpp
    tools/mpbake/DagBuilder.cpp
    tools/mpbake/PageWriter.cpp
    src/asset/MpAssetFormat.cpp
    src/asset/MpAssetReader.cpp
)
target_include_directories(page_writer_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tools
)
target_include_directories(page_writer_test SYSTEM PRIVATE
    ${glm_SOURCE_DIR}
    ${fastgltf_SOURCE_DIR}/include
)
target_link_libraries(page_writer_test PRIVATE
    fastgltf
    glm::glm
    meshoptimizer
    metis
    libzstd_static
)
target_compile_definitions(page_writer_test PRIVATE
    $<$<PLATFORM_ID:Windows>:NOMINMAX>
    IDXTYPEWIDTH=32
    REALTYPEWIDTH=32
)
if(MSVC)
    target_compile_options(page_writer_test PRIVATE
        /std:c++latest /W4 /WX /permissive- /Zc:preprocessor /wd4127)
else()
    target_compile_options(page_writer_test PRIVATE
        -std=c++2c -Wall -Wextra -Wpedantic -Werror
        -Wno-missing-field-initializers)
endif()

# -----------------------------------------------------------------------------
# M1c: mp_asset_reader_test — narrow runtime-side exercises for the reader
# beyond what the writer round-trip covers (missing file, truncated file,
# valid file). CPU-only; shares the asset TUs with the main Enigma target.
# -----------------------------------------------------------------------------
add_executable(mp_asset_reader_test
    tests/mp_asset_reader_test.cpp
    tools/mpbake/GltfIngest.cpp
    tools/mpbake/ClusterBuilder.cpp
    tools/mpbake/Simplify.cpp
    tools/mpbake/DagBuilder.cpp
    tools/mpbake/PageWriter.cpp
    src/asset/MpAssetFormat.cpp
    src/asset/MpAssetReader.cpp
)
target_include_directories(mp_asset_reader_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tools
)
target_include_directories(mp_asset_reader_test SYSTEM PRIVATE
    ${glm_SOURCE_DIR}
    ${fastgltf_SOURCE_DIR}/include
)
target_link_libraries(mp_asset_reader_test PRIVATE
    fastgltf
    glm::glm
    meshoptimizer
    metis
    libzstd_static
)
target_compile_definitions(mp_asset_reader_test PRIVATE
    $<$<PLATFORM_ID:Windows>:NOMINMAX>
    IDXTYPEWIDTH=32
    REALTYPEWIDTH=32
)
if(MSVC)
    target_compile_options(mp_asset_reader_test PRIVATE
        /std:c++latest /W4 /WX /permissive- /Zc:preprocessor /wd4127)
else()
    target_compile_options(mp_asset_reader_test PRIVATE
        -std=c++2c -Wall -Wextra -Wpedantic -Werror
        -Wno-missing-field-initializers)
endif()

# -----------------------------------------------------------------------------
# M2.1: residency_lru_test — unit test for the CPU-side LRU residency
# tracker. Four cases: basic insert/hit, eviction under pressure,
# beginFrame + touch keeps a page alive, stats counters match scenario.
# Pure CPU; no Vulkan, no mpbake.
# -----------------------------------------------------------------------------
add_executable(residency_lru_test
    tests/residency_lru_test.cpp
    src/renderer/micropoly/ResidencyManager.cpp
)
target_include_directories(residency_lru_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_definitions(residency_lru_test PRIVATE
    $<$<PLATFORM_ID:Windows>:NOMINMAX>
)
if(MSVC)
    target_compile_options(residency_lru_test PRIVATE
        /std:c++latest /W4 /WX /permissive- /Zc:preprocessor /wd4127)
else()
    target_compile_options(residency_lru_test PRIVATE
        -std=c++2c -Wall -Wextra -Wpedantic -Werror
        -Wno-missing-field-initializers)
endif()

# -----------------------------------------------------------------------------
# M2.1: async_io_worker_test — unit test for the background page-reader
# thread. Three cases: happy-path read vs MpAssetReader::fetchPage,
# shutdown drains in-flight + rejects later enqueues, bad fileOffset
# surfaces a failure completion. Reuses the mpbake pipeline to produce
# a real .mpa from DamagedHelmet.glb.
# -----------------------------------------------------------------------------
add_executable(async_io_worker_test
    tests/async_io_worker_test.cpp
    src/renderer/micropoly/AsyncIOWorker.cpp
    tools/mpbake/GltfIngest.cpp
    tools/mpbake/ClusterBuilder.cpp
    tools/mpbake/Simplify.cpp
    tools/mpbake/DagBuilder.cpp
    tools/mpbake/PageWriter.cpp
    src/asset/MpAssetFormat.cpp
    src/asset/MpAssetReader.cpp
)
target_include_directories(async_io_worker_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tools
)
target_include_directories(async_io_worker_test SYSTEM PRIVATE
    ${glm_SOURCE_DIR}
    ${fastgltf_SOURCE_DIR}/include
)
target_link_libraries(async_io_worker_test PRIVATE
    fastgltf
    glm::glm
    meshoptimizer
    metis
    libzstd_static
)
target_compile_definitions(async_io_worker_test PRIVATE
    $<$<PLATFORM_ID:Windows>:NOMINMAX>
    IDXTYPEWIDTH=32
    REALTYPEWIDTH=32
)
if(MSVC)
    target_compile_options(async_io_worker_test PRIVATE
        /std:c++latest /W4 /WX /permissive- /Zc:preprocessor /wd4127)
else()
    target_compile_options(async_io_worker_test PRIVATE
        -std=c++2c -Wall -Wextra -Wpedantic -Werror
        -Wno-missing-field-initializers)
endif()

# -----------------------------------------------------------------------------
# M2.2: page_cache_test — unit test for the GPU-side fixed-address page pool.
# Eight cases: creation + stats, exhaustion, realloc-after-free, double-free,
# out-of-range, options validation, slotByteOffset, basic thread safety.
# Requires a headless Vulkan device — links volk + VMA and brings up a
# minimal Instance/Device locally (pattern from ScreenshotDiffHarness).
# -----------------------------------------------------------------------------
add_executable(page_cache_test
    tests/page_cache_test.cpp
    src/gfx/AllocatorAdopt.cpp
    src/renderer/micropoly/PageCache.cpp
)
target_include_directories(page_cache_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_include_directories(page_cache_test SYSTEM PRIVATE
    ${volk_SOURCE_DIR}
    ${VulkanMemoryAllocator_SOURCE_DIR}/include
)
target_link_libraries(page_cache_test PRIVATE
    volk::volk
    GPUOpen::VulkanMemoryAllocator
)
target_compile_definitions(page_cache_test PRIVATE
    VK_NO_PROTOTYPES
    $<$<PLATFORM_ID:Windows>:VK_USE_PLATFORM_WIN32_KHR>
    $<$<PLATFORM_ID:Windows>:NOMINMAX>
)
if(MSVC)
    target_compile_options(page_cache_test PRIVATE
        /std:c++latest /W4 /WX /permissive- /Zc:preprocessor /wd4127)
else()
    target_compile_options(page_cache_test PRIVATE
        -std=c++2c -Wall -Wextra -Wpedantic -Werror
        -Wno-missing-field-initializers)
endif()

# -----------------------------------------------------------------------------
# M2.3: request_queue_test — unit test for the GPU-visible coherent ring
# buffer used by compute shaders to emit page requests. Four cases: creation
# + zero stats, manual header write + drain, drain into smaller-than-count
# output span, overflow flag detection. Requires a headless Vulkan device.
# -----------------------------------------------------------------------------
add_executable(request_queue_test
    tests/request_queue_test.cpp
    src/gfx/AllocatorAdopt.cpp
    src/renderer/micropoly/RequestQueue.cpp
)
target_include_directories(request_queue_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_include_directories(request_queue_test SYSTEM PRIVATE
    ${volk_SOURCE_DIR}
    ${VulkanMemoryAllocator_SOURCE_DIR}/include
)
target_link_libraries(request_queue_test PRIVATE
    volk::volk
    GPUOpen::VulkanMemoryAllocator
)
target_compile_definitions(request_queue_test PRIVATE
    VK_NO_PROTOTYPES
    $<$<PLATFORM_ID:Windows>:VK_USE_PLATFORM_WIN32_KHR>
    $<$<PLATFORM_ID:Windows>:NOMINMAX>
)
if(MSVC)
    target_compile_options(request_queue_test PRIVATE
        /std:c++latest /W4 /WX /permissive- /Zc:preprocessor /wd4127)
else()
    target_compile_options(request_queue_test PRIVATE
        -std=c++2c -Wall -Wextra -Wpedantic -Werror
        -Wno-missing-field-initializers)
endif()

# -----------------------------------------------------------------------------
# M2.3: micropoly_streaming_test — integration test for the streaming
# orchestrator. Bakes DamagedHelmet.glb into a .mpa, pokes the request
# queue with real pageIds, drives beginFrame() through dispatch + async IO
# + completion consumption, verifies residency + page-cache slot mapping.
# The M2.3 transfer-queue stub leaves uploadsScheduled=0 — asserted here.
# -----------------------------------------------------------------------------
add_executable(micropoly_streaming_test
    tests/micropoly_streaming_test.cpp
    src/gfx/DeviceAdopt.cpp
    src/gfx/AllocatorAdopt.cpp
    src/renderer/micropoly/MicropolyStreaming.cpp
    src/renderer/micropoly/RequestQueue.cpp
    src/renderer/micropoly/PageCache.cpp
    src/renderer/micropoly/ResidencyManager.cpp
    src/renderer/micropoly/AsyncIOWorker.cpp
    tools/mpbake/GltfIngest.cpp
    tools/mpbake/ClusterBuilder.cpp
    tools/mpbake/Simplify.cpp
    tools/mpbake/DagBuilder.cpp
    tools/mpbake/PageWriter.cpp
    src/asset/MpAssetFormat.cpp
    src/asset/MpAssetReader.cpp
)
target_include_directories(micropoly_streaming_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tools
)
target_include_directories(micropoly_streaming_test SYSTEM PRIVATE
    ${volk_SOURCE_DIR}
    ${VulkanMemoryAllocator_SOURCE_DIR}/include
    ${glm_SOURCE_DIR}
    ${fastgltf_SOURCE_DIR}/include
)
target_link_libraries(micropoly_streaming_test PRIVATE
    volk::volk
    GPUOpen::VulkanMemoryAllocator
    fastgltf
    glm::glm
    meshoptimizer
    metis
    libzstd_static
)
target_compile_definitions(micropoly_streaming_test PRIVATE
    VK_NO_PROTOTYPES
    $<$<PLATFORM_ID:Windows>:VK_USE_PLATFORM_WIN32_KHR>
    $<$<PLATFORM_ID:Windows>:NOMINMAX>
    IDXTYPEWIDTH=32
    REALTYPEWIDTH=32
)
if(MSVC)
    target_compile_options(micropoly_streaming_test PRIVATE
        /std:c++latest /W4 /WX /permissive- /Zc:preprocessor /wd4127)
else()
    target_compile_options(micropoly_streaming_test PRIVATE
        -std=c++2c -Wall -Wextra -Wpedantic -Werror
        -Wno-missing-field-initializers)
endif()

# -----------------------------------------------------------------------------
# M2.4b: throughput_bench — measures sustained page-read throughput for the
# Micropoly streaming IO subsystem. Three side-by-side numbers:
#   (a) AsyncIOWorker round-trip (IOCP + zstd + callback)
#   (b) Raw IOCP throughput (Windows; plan target >= 3 GB/s)
#   (c) Raw IOCP + zstd decompress (target >= 1 GB/s)
# Not a pass/fail gate — prints numbers, always returns 0. Dev hardware
# below the 3 GB/s target is reported as-is; the absolute number is
# workstation-dependent.
# -----------------------------------------------------------------------------
add_executable(throughput_bench
    tests/throughput_bench.cpp
    src/renderer/micropoly/AsyncIOWorker.cpp
    tools/mpbake/GltfIngest.cpp
    tools/mpbake/ClusterBuilder.cpp
    tools/mpbake/Simplify.cpp
    tools/mpbake/DagBuilder.cpp
    tools/mpbake/PageWriter.cpp
    src/asset/MpAssetFormat.cpp
    src/asset/MpAssetReader.cpp
)
target_include_directories(throughput_bench PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tools
)
target_include_directories(throughput_bench SYSTEM PRIVATE
    ${glm_SOURCE_DIR}
    ${fastgltf_SOURCE_DIR}/include
)
target_link_libraries(throughput_bench PRIVATE
    fastgltf
    glm::glm
    meshoptimizer
    metis
    libzstd_static
)
target_compile_definitions(throughput_bench PRIVATE
    $<$<PLATFORM_ID:Windows>:NOMINMAX>
    IDXTYPEWIDTH=32
    REALTYPEWIDTH=32
)
if(MSVC)
    target_compile_options(throughput_bench PRIVATE
        /std:c++latest /W4 /WX /permissive- /Zc:preprocessor /wd4127)
else()
    target_compile_options(throughput_bench PRIVATE
        -std=c++2c -Wall -Wextra -Wpedantic -Werror
        -Wno-missing-field-initializers)
endif()

# -----------------------------------------------------------------------------
# M2.4a: cold_cache_stress_test — integration test that drives the streaming
# orchestrator at its cold-cache peak. Bakes DamagedHelmet.glb, requests ALL
# pages in quick succession across 3 reset cycles, asserts residency counts,
# PageCache slot use, and that residency/slot ordering stays consistent across
# cycles. Uses the same headless Vulkan bundle + Device::adopt() pattern as
# micropoly_streaming_test.
# -----------------------------------------------------------------------------
add_executable(cold_cache_stress_test
    tests/cold_cache_stress_test.cpp
    src/gfx/DeviceAdopt.cpp
    src/gfx/AllocatorAdopt.cpp
    src/renderer/micropoly/MicropolyStreaming.cpp
    src/renderer/micropoly/RequestQueue.cpp
    src/renderer/micropoly/PageCache.cpp
    src/renderer/micropoly/ResidencyManager.cpp
    src/renderer/micropoly/AsyncIOWorker.cpp
    tools/mpbake/GltfIngest.cpp
    tools/mpbake/ClusterBuilder.cpp
    tools/mpbake/Simplify.cpp
    tools/mpbake/DagBuilder.cpp
    tools/mpbake/PageWriter.cpp
    src/asset/MpAssetFormat.cpp
    src/asset/MpAssetReader.cpp
)
target_include_directories(cold_cache_stress_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tools
)
target_include_directories(cold_cache_stress_test SYSTEM PRIVATE
    ${volk_SOURCE_DIR}
    ${VulkanMemoryAllocator_SOURCE_DIR}/include
    ${glm_SOURCE_DIR}
    ${fastgltf_SOURCE_DIR}/include
)
target_link_libraries(cold_cache_stress_test PRIVATE
    volk::volk
    GPUOpen::VulkanMemoryAllocator
    fastgltf
    glm::glm
    meshoptimizer
    metis
    libzstd_static
)
target_compile_definitions(cold_cache_stress_test PRIVATE
    VK_NO_PROTOTYPES
    $<$<PLATFORM_ID:Windows>:VK_USE_PLATFORM_WIN32_KHR>
    $<$<PLATFORM_ID:Windows>:NOMINMAX>
    IDXTYPEWIDTH=32
    REALTYPEWIDTH=32
)
if(MSVC)
    target_compile_options(cold_cache_stress_test PRIVATE
        /std:c++latest /W4 /WX /permissive- /Zc:preprocessor /wd4127)
else()
    target_compile_options(cold_cache_stress_test PRIVATE
        -std=c++2c -Wall -Wextra -Wpedantic -Werror
        -Wno-missing-field-initializers)
endif()

# -----------------------------------------------------------------------------
# M3.2: micropoly_cull_test — smoke test for MicropolyCullPass's public
# surface. Three cases: layout constant sanity, error-kind stringifier
# round-trip, headless Vulkan bundle bringup (SKIPs on no-device CI boxes).
# Does NOT exercise the live compute dispatch — that is covered end-to-end
# by the Enigma binary when MicropolyConfig::enabled=true; the smoke suite
# guards against header/ABI regressions without requiring a full
# Device::adopt + ShaderManager runtime.
# -----------------------------------------------------------------------------
add_executable(micropoly_cull_test
    tests/micropoly_cull_test.cpp
)
target_include_directories(micropoly_cull_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_include_directories(micropoly_cull_test SYSTEM PRIVATE
    ${volk_SOURCE_DIR}
    ${VulkanMemoryAllocator_SOURCE_DIR}/include
)
target_link_libraries(micropoly_cull_test PRIVATE
    volk::volk
    GPUOpen::VulkanMemoryAllocator
)
target_compile_definitions(micropoly_cull_test PRIVATE
    VK_NO_PROTOTYPES
    $<$<PLATFORM_ID:Windows>:VK_USE_PLATFORM_WIN32_KHR>
    $<$<PLATFORM_ID:Windows>:NOMINMAX>
)
if(MSVC)
    target_compile_options(micropoly_cull_test PRIVATE
        /std:c++latest /W4 /WX /permissive- /Zc:preprocessor /wd4127)
else()
    target_compile_options(micropoly_cull_test PRIVATE
        -std=c++2c -Wall -Wextra -Wpedantic -Werror
        -Wno-missing-field-initializers)
endif()

# -----------------------------------------------------------------------------
# M3.3: micropoly_raster_test — smoke test for MicropolyRasterPass's public
# surface (error-kind enum integrity, push-block size sanity, header-only
# layout constants). Full pipeline bringup requires a live ShaderManager +
# real device + DXC — out of scope for a header-linked smoke test. The
# end-to-end raster dispatch is exercised live by the Enigma binary when
# MicropolyConfig::enabled=true + mesh-shader capable device.
# -----------------------------------------------------------------------------
add_executable(micropoly_raster_test
    tests/micropoly_raster_test.cpp
    src/renderer/micropoly/MicropolyRasterPassError.cpp
)
target_include_directories(micropoly_raster_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_include_directories(micropoly_raster_test SYSTEM PRIVATE
    ${volk_SOURCE_DIR}
    ${VulkanMemoryAllocator_SOURCE_DIR}/include
)
target_link_libraries(micropoly_raster_test PRIVATE
    volk::volk
    GPUOpen::VulkanMemoryAllocator
)
target_compile_definitions(micropoly_raster_test PRIVATE
    VK_NO_PROTOTYPES
    $<$<PLATFORM_ID:Windows>:VK_USE_PLATFORM_WIN32_KHR>
    $<$<PLATFORM_ID:Windows>:NOMINMAX>
)
if(MSVC)
    target_compile_options(micropoly_raster_test PRIVATE
        /std:c++latest /W4 /WX /permissive- /Zc:preprocessor /wd4127)
else()
    target_compile_options(micropoly_raster_test PRIVATE
        -std=c++2c -Wall -Wextra -Wpedantic -Werror
        -Wno-missing-field-initializers)
endif()

# -----------------------------------------------------------------------------
# M4.2: micropoly_sw_raster_bin_test — smoke test for MicropolySwRasterPass's
# public surface (error-kind enum integrity, stringifier round-trip, headless
# Vulkan bundle bringup). Full pipeline bringup requires a live ShaderManager +
# real device + DXC — out of scope for a header-linked smoke test. The
# end-to-end binning dispatch is exercised live by the Enigma binary when
# MicropolyConfig::enabled=true + mesh-shader capable device.
# -----------------------------------------------------------------------------
add_executable(micropoly_sw_raster_bin_test
    tests/micropoly_sw_raster_bin_test.cpp
    src/renderer/micropoly/MicropolySwRasterPassError.cpp
)
target_include_directories(micropoly_sw_raster_bin_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_include_directories(micropoly_sw_raster_bin_test SYSTEM PRIVATE
    ${volk_SOURCE_DIR}
    ${VulkanMemoryAllocator_SOURCE_DIR}/include
)
target_link_libraries(micropoly_sw_raster_bin_test PRIVATE
    volk::volk
    GPUOpen::VulkanMemoryAllocator
)
target_compile_definitions(micropoly_sw_raster_bin_test PRIVATE
    VK_NO_PROTOTYPES
    $<$<PLATFORM_ID:Windows>:VK_USE_PLATFORM_WIN32_KHR>
    $<$<PLATFORM_ID:Windows>:NOMINMAX>
)
if(MSVC)
    target_compile_options(micropoly_sw_raster_bin_test PRIVATE
        /std:c++latest /W4 /WX /permissive- /Zc:preprocessor /wd4127)
else()
    target_compile_options(micropoly_sw_raster_bin_test PRIVATE
        -std=c++2c -Wall -Wextra -Wpedantic -Werror
        -Wno-missing-field-initializers)
endif()

# -----------------------------------------------------------------------------
# M6.2a: micropoly_settings_panel_test — header-linked smoke test for the
# MicropolySettingsPanel class. Verifies the header includes cleanly, the
# class default-constructs, the default MicropolyConfig values land inside
# the panel's slider bounds, and the DebugMode symbols the overlay radio
# references exist. The ImGui widgets themselves are not exercised — a
# valid ImGui context requires the Vulkan backend, which is out of scope
# for a header-linked test binary (pattern copied from
# micropoly_raster_test).
# -----------------------------------------------------------------------------
add_executable(micropoly_settings_panel_test
    tests/micropoly_settings_panel_test.cpp
)
target_include_directories(micropoly_settings_panel_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
# The panel header forward-declares gfx::Device + DebugMode so the test TU
# does not need to link Vulkan or drag in the renderer stack. Only glm is
# pulled in because Renderer.h (included for DebugMode) transitively needs
# glm types through core/Math.h.
target_include_directories(micropoly_settings_panel_test SYSTEM PRIVATE
    ${volk_SOURCE_DIR}
    ${VulkanMemoryAllocator_SOURCE_DIR}/include
    ${glm_SOURCE_DIR}
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
    ${fastgltf_SOURCE_DIR}/include
    ${joltphysics_SOURCE_DIR}
)
target_compile_definitions(micropoly_settings_panel_test PRIVATE
    VK_NO_PROTOTYPES
    $<$<PLATFORM_ID:Windows>:VK_USE_PLATFORM_WIN32_KHR>
    $<$<PLATFORM_ID:Windows>:NOMINMAX>
    GLFW_INCLUDE_NONE
    $<$<CONFIG:Release>:JPH_NO_DEBUG>
)
if(MSVC)
    target_compile_options(micropoly_settings_panel_test PRIVATE
        /std:c++latest /W4 /WX /permissive- /Zc:preprocessor /wd4127)
else()
    target_compile_options(micropoly_settings_panel_test PRIVATE
        -std=c++2c -Wall -Wextra -Wpedantic -Werror
        -Wno-missing-field-initializers)
endif()

# -----------------------------------------------------------------------------
# M5a: micropoly_blas_manager_test — smoke test for MicropolyBlasManager's
# public surface (error-kind enum integrity, stringifier round-trip, headless
# Vulkan bundle bringup). Full MicropolyBlasManager::create()/buildForAsset()
# bringup requires a live gfx::Device + gfx::Allocator + a baked .mpa file
# + RT-capable hardware — out of scope for a header-linked smoke test. The
# end-to-end BLAS build is exercised live by the Enigma binary when
# MicropolyConfig::enabled=true on an RT-capable device.
# -----------------------------------------------------------------------------
add_executable(micropoly_blas_manager_test
    tests/micropoly_blas_manager_test.cpp
    src/renderer/micropoly/MicropolyBlasManagerError.cpp
)
target_include_directories(micropoly_blas_manager_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_include_directories(micropoly_blas_manager_test SYSTEM PRIVATE
    ${volk_SOURCE_DIR}
    ${VulkanMemoryAllocator_SOURCE_DIR}/include
    ${glm_SOURCE_DIR}
)
target_link_libraries(micropoly_blas_manager_test PRIVATE
    volk::volk
    GPUOpen::VulkanMemoryAllocator
    glm::glm
)
target_compile_definitions(micropoly_blas_manager_test PRIVATE
    VK_NO_PROTOTYPES
    $<$<PLATFORM_ID:Windows>:VK_USE_PLATFORM_WIN32_KHR>
    $<$<PLATFORM_ID:Windows>:NOMINMAX>
)
if(MSVC)
    target_compile_options(micropoly_blas_manager_test PRIVATE
        /std:c++latest /W4 /WX /permissive- /Zc:preprocessor /wd4127)
else()
    target_compile_options(micropoly_blas_manager_test PRIVATE
        -std=c++2c -Wall -Wextra -Wpedantic -Werror
        -Wno-missing-field-initializers)
endif()
