# -----------------------------------------------------------------------------
# Dependencies.cmake
#
# FetchContent-based dependency management for Enigma. Every dependency is
# pinned to an exact commit SHA (40-char hex) — tag names are tracked in the
# adjacent comment for provenance but the SHA is the source of truth.
# -----------------------------------------------------------------------------

include(FetchContent)

# -----------------------------------------------------------------------------
# volk — meta-loader for Vulkan. Eliminates the need to link against the
# Vulkan loader directly; loads entry points at runtime.
# Tag: vulkan-sdk-1.3.296.0
# -----------------------------------------------------------------------------
FetchContent_Declare(
    volk
    GIT_REPOSITORY https://github.com/zeux/volk.git
    GIT_TAG        0b17a763ba5643e32da1b2dee2a1ff7f1eb55d5c
)
FetchContent_MakeAvailable(volk)

# -----------------------------------------------------------------------------
# VulkanMemoryAllocator (VMA) — AMD's GPU memory allocator for Vulkan.
# Tag: v3.1.0
# -----------------------------------------------------------------------------
FetchContent_Declare(
    VulkanMemoryAllocator
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
    GIT_TAG        1732c4a74e0ca58d40e7c21c303cef7b70b75eb6
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
    GIT_TAG        7b6aead9fb88b3623e3b3725ebb42670cbe4c579
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
