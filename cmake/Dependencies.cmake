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
    GIT_TAG        v0.7.2
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
