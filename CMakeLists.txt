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
# Vulkan SDK — provides headers, validation layer runtime, and shaderc
# (runtime GLSL -> SPIR-V compilation via the Vulkan::shaderc_combined
# imported target). Minimum version 1.3 is required for dynamic rendering,
# descriptor indexing, sync2, and timeline semaphores as core features.
# -----------------------------------------------------------------------------
find_package(Vulkan 1.3 REQUIRED COMPONENTS shaderc_combined)

# -----------------------------------------------------------------------------
# Target
# -----------------------------------------------------------------------------
add_executable(Enigma)

target_sources(Enigma PRIVATE
    main.cpp
    src/core/Log.cpp
    src/core/Paths.cpp
    src/platform/Window.cpp
    src/engine/Clock.cpp
    src/engine/Application.cpp
    src/engine/Engine.cpp
    src/renderer/Renderer.cpp
    src/gfx/Instance.cpp
    src/gfx/Validation.cpp
    src/gfx/Device.cpp
    src/gfx/Allocator.cpp
    src/gfx/Swapchain.cpp
    src/gfx/FrameContext.cpp
    src/gfx/DescriptorAllocator.cpp
    src/gfx/ShaderManager.cpp
    src/gfx/Pipeline.cpp
    src/renderer/TrianglePass.cpp
)

target_include_directories(Enigma PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(Enigma PRIVATE
    volk::volk
    GPUOpen::VulkanMemoryAllocator
    glfw
    glm::glm
    Vulkan::shaderc_combined
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
)

# volk loads Vulkan at runtime; we only need the Vulkan headers transitively
# through volk. We do NOT link Vulkan::Vulkan directly.
target_compile_definitions(Enigma PRIVATE
    VK_NO_PROTOTYPES
    $<$<PLATFORM_ID:Windows>:VK_USE_PLATFORM_WIN32_KHR>
    GLFW_INCLUDE_NONE
    ENIGMA_SHADER_SOURCE_DIR="${CMAKE_SOURCE_DIR}/shaders"
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
