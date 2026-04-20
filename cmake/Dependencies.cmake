# -----------------------------------------------------------------------------
# Dependencies.cmake
#
# FetchContent-based dependency management for Enigma. Every dependency is
# pinned to an exact commit SHA (40-char hex) — tag names are tracked in the
# adjacent comment for provenance but the SHA is the source of truth.
# -----------------------------------------------------------------------------

include(FetchContent)

# cgltf and stb are header-only C libraries with no CMakeLists.txt, so
# FetchContent_MakeAvailable cannot be used. Allow the legacy
# FetchContent_Populate path on CMake 4.x without a dev warning.
if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
endif()

# -----------------------------------------------------------------------------
# volk — meta-loader for Vulkan. Eliminates the need to link against the
# Vulkan loader directly; loads entry points at runtime.
# Tag: vulkan-sdk-1.4.321.0
# -----------------------------------------------------------------------------
FetchContent_Declare(
    volk
    GIT_REPOSITORY https://github.com/zeux/volk.git
    GIT_TAG        a8da8ef3368482b0ee9b0ec0c6079a16a89c6924
)
FetchContent_MakeAvailable(volk)

# -----------------------------------------------------------------------------
# VulkanMemoryAllocator (VMA) — AMD's GPU memory allocator for Vulkan.
# Tag: v3.1.0
# -----------------------------------------------------------------------------
FetchContent_Declare(
    VulkanMemoryAllocator
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
    GIT_TAG        009ecd192c1289c7529bff248a16cfe896254816
)
FetchContent_MakeAvailable(VulkanMemoryAllocator)

# -----------------------------------------------------------------------------
# GLFW — windowing / input library. Docs/tests/examples are suppressed via
# cache variables set before MakeAvailable so the GLFW build scripts see them.
# Tag: 3.4
# -----------------------------------------------------------------------------
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        a74efa0d5628b74adc0426af4c5710e287fa7c2c
)
FetchContent_MakeAvailable(glfw)

# -----------------------------------------------------------------------------
# glm — header-only math library.
# Tag: 1.0.1
# -----------------------------------------------------------------------------
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        0af55ccecd98d4e5a8d1fad7de25ba429d60e863
)
FetchContent_MakeAvailable(glm)

# -----------------------------------------------------------------------------
# fastgltf — modern C++17 glTF 2.0 parser. SIMD-accelerated JSON parsing
# via simdjson, zero-copy buffer views, type-safe accessor iteration.
# Tag: v0.7.2 (v0.8.0 has C++23 template compat issues on MSVC /std:c++latest)
# -----------------------------------------------------------------------------
set(FASTGLTF_COMPILE_AS_CPP20 ON CACHE BOOL "" FORCE)
FetchContent_Declare(
    fastgltf
    GIT_REPOSITORY https://github.com/spnda/fastgltf.git
    GIT_TAG        c7ab6f9e07ad23e191c45ab6e9ffe6f58076fb32   # v0.7.2 (SHA resolved via `git ls-remote ... refs/tags/v0.7.2` on 2026-04-18)
)
FetchContent_MakeAvailable(fastgltf)

# -----------------------------------------------------------------------------
# stb — Sean Barrett's single-file public domain libraries. Only stb_image.h
# is used (PNG/JPEG/BMP/TGA decoding for glTF texture loading). Header-only.
# Commit: latest as of 2024-07 (no versioned releases)
# -----------------------------------------------------------------------------
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        f75e8d1cad7d90d72ef7a4661f1b994ef78b4e31
)
FetchContent_GetProperties(stb)
if(NOT stb_POPULATED)
    FetchContent_Populate(stb)
endif()

# -----------------------------------------------------------------------------
# Jolt Physics — MIT license, C++17, no-exception, no-RTTI build.
# Tag: v5.2.0
# -----------------------------------------------------------------------------
# Jolt enables /Wall /WX by default which causes Windows SDK C4865
# warnings to fail the build under /std:c++latest. Disable those options.
# USE_STATIC_MSVC_RUNTIME_LIBRARY OFF ensures Jolt links against /MDd
# matching Enigma's dynamic CRT.
set(OVERRIDE_CXX_FLAGS OFF CACHE BOOL "" FORCE)
set(ENABLE_ALL_WARNINGS OFF CACHE BOOL "" FORCE)
set(INTERPROCEDURAL_OPTIMIZATION OFF CACHE BOOL "" FORCE)
set(USE_STATIC_MSVC_RUNTIME_LIBRARY OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    JoltPhysics
    GIT_REPOSITORY https://github.com/jrouwe/JoltPhysics.git
    GIT_TAG        a63aa3b8e24cf95f3fab2613f9a3015b164ef62c
    GIT_SHALLOW    FALSE
    SOURCE_SUBDIR  Build
)
FetchContent_MakeAvailable(JoltPhysics)

# -----------------------------------------------------------------------------
# Dear ImGui — immediate-mode GUI library. Master branch (non-docking) — no
# DockSpace/SetNextWindowDockID usage in src/; if docking is wanted later,
# swap to the v1.91.5-docking tag SHA 97fa363adb1439d11b613d96bcda86d44ea16f4e.
# No CMakeLists.txt for library use; sources added directly to target.
# Tag: v1.91.5 (SHA resolved via `git ls-remote ... refs/tags/v1.91.5` on 2026-04-18)
# -----------------------------------------------------------------------------
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        f401021d5a5d56fe2304056c391e78f81c8d4b8f   # v1.91.5
)
FetchContent_GetProperties(imgui)
if(NOT imgui_POPULATED)
    FetchContent_Populate(imgui)
endif()

# -----------------------------------------------------------------------------
# meshoptimizer — Arseny Kapoulkine's mesh processing library. Used by the
# offline `enigma-mpbake` tool (M1) for cluster building via
# meshopt_buildMeshlets and simplification via meshopt_simplify. Not linked
# into the runtime Enigma binary.
# Tag: v0.21 (SHA resolved via `git ls-remote ... refs/tags/v0.21` on 2026-04-18)
# -----------------------------------------------------------------------------
set(MESHOPT_BUILD_DEMO     OFF CACHE BOOL "" FORCE)
set(MESHOPT_BUILD_GLTFPACK OFF CACHE BOOL "" FORCE)
set(MESHOPT_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(MESHOPT_WERROR         OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    meshoptimizer
    GIT_REPOSITORY https://github.com/zeux/meshoptimizer.git
    GIT_TAG        47aafa533b439a78b53cd2854c177db61be7e666
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(meshoptimizer)

# -----------------------------------------------------------------------------
# METIS — graph partitioning library (Karypis). Used by `enigma-mpbake` (M1)
# for DAG clustering via k-way partitioning of cluster adjacency graphs.
# Not linked into the runtime Enigma binary.
#
# Vendored under `vendor/metis/` (2026-04-18) from the scivision mirror at
# SHA 777472ae3cd15a8e6d1e5b7d6c347d21947e3ab2 — see vendor/metis/README.md
# for provenance, what was stripped, and the manual resync procedure.
# Vendoring eliminates the third-party-mirror trust boundary: a force-push
# on scivision/main can no longer silently substitute library source under
# `METIS_PartGraphKway`, which runs on attacker-controllable glTF-derived
# adjacency graphs in enigma-mpbake.
#
# GKlib is still FetchContent-pulled by vendor/metis/CMakeLists.txt (from
# the scivision GKlib mirror at a fixed SHA in the tarball URL). Vendoring
# GKlib too would close the last mirror dependency but increases the
# vendor footprint; defer unless an audit demands it.
# -----------------------------------------------------------------------------
set(IDXTYPEWIDTH 32 CACHE STRING "" FORCE)   # per plan §3.M1
set(REALTYPEWIDTH 32 CACHE STRING "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_SOURCE_DIR}/vendor/metis EXCLUDE_FROM_ALL)

# -----------------------------------------------------------------------------
# zstd — Facebook's Zstandard compression. Used by `enigma-mpbake` (M1) for
# per-page compression in the .mpa file format. Linked into the runtime
# Enigma binary as well so MpAssetReader can decompress pages at stream time.
#
# The project's top-level CMakeLists.txt is not the library build — the
# canonical entry for embedding is `build/cmake/`, which defines the
# `libzstd_static` static-library target we consume.
#
# Tag: v1.5.6 (SHA resolved via `git ls-remote ... refs/tags/v1.5.6` on 2026-04-18)
# -----------------------------------------------------------------------------
set(ZSTD_BUILD_PROGRAMS   OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_TESTS      OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_SHARED     OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_STATIC     ON  CACHE BOOL "" FORCE)
set(ZSTD_LEGACY_SUPPORT   OFF CACHE BOOL "" FORCE)
set(ZSTD_MULTITHREAD_SUPPORT OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    zstd
    GIT_REPOSITORY https://github.com/facebook/zstd.git
    GIT_TAG        35016bc1c0b9a2f7121b7ecc312100aad7d9f2ad
    GIT_SHALLOW    TRUE
    SOURCE_SUBDIR  build/cmake
)
FetchContent_MakeAvailable(zstd)

# -----------------------------------------------------------------------------
# Vendored-dep warning isolation.
# Enigma's own TUs build under /W4 /WX /permissive-. Third-party dep code may
# trip MSVC C4-level warnings (e.g. GKlib's gkregex.c fires C4311/C4312, zstd
# occasionally redefines INFINITY). Suppress /WX for the specific vendored
# library targets so a clean upstream bump can't silently turn red. Only
# METIS and libzstd_static need this — meshoptimizer honours its own
# MESHOPT_WERROR=OFF knob (set above at line 142).
# -----------------------------------------------------------------------------
if(TARGET metis)
    target_compile_options(metis PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/W0>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-w>)
endif()
if(TARGET libzstd_static)
    target_compile_options(libzstd_static PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/W0>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-w>)
endif()
