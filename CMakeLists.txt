cmake_minimum_required(VERSION 3.28)
project(Enigma LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif()

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
