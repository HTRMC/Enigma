cmake_minimum_required(VERSION 3.28)
project(Enigma LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
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
)

target_include_directories(Enigma PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(Enigma PRIVATE
    volk::volk_headers
    GPUOpen::VulkanMemoryAllocator
    glfw
    glm::glm
    Vulkan::shaderc_combined
)

# volk loads Vulkan at runtime; we only need the Vulkan headers transitively
# through volk. We do NOT link Vulkan::Vulkan directly.
target_compile_definitions(Enigma PRIVATE
    VK_NO_PROTOTYPES
    $<$<PLATFORM_ID:Windows>:VK_USE_PLATFORM_WIN32_KHR>
    GLFW_INCLUDE_NONE
)

# Warnings-as-errors scoped to the Enigma target only. Third-party headers
# pulled in by FetchContent are marked SYSTEM in a later step so their
# warnings do not cascade into this strict policy.
if(MSVC)
    target_compile_options(Enigma PRIVATE
        /W4 /WX /permissive- /Zc:preprocessor
    )
else()
    target_compile_options(Enigma PRIVATE
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
