#pragma once

#include <cstdio>
#include <cstdlib>

// Debug-only asserts. In release builds these compile away entirely.
//
//   ENIGMA_ASSERT(cond)          — debug-only; aborts on false.
//   ENIGMA_VERIFY(cond)          — always evaluates `cond`; aborts in debug
//                                   on false, silently continues in release.
//   ENIGMA_VK_CHECK(vk_result)   — checks VkResult == VK_SUCCESS; logs the
//                                   numeric enum and aborts in debug.

#if defined(_MSC_VER)
    #define ENIGMA_DEBUGBREAK() __debugbreak()
#elif defined(__GNUC__) || defined(__clang__)
    #define ENIGMA_DEBUGBREAK() __builtin_trap()
#else
    #define ENIGMA_DEBUGBREAK() std::abort()
#endif

#if !defined(NDEBUG)
    #define ENIGMA_DEBUG 1
#else
    #define ENIGMA_DEBUG 0
#endif

#if ENIGMA_DEBUG
    #define ENIGMA_ASSERT(cond)                                                \
        do {                                                                   \
            if (!(cond)) {                                                     \
                std::fprintf(stderr,                                           \
                    "[enigma] ASSERT failed: %s\n  at %s:%d\n",                \
                    #cond, __FILE__, __LINE__);                                \
                ENIGMA_DEBUGBREAK();                                           \
            }                                                                  \
        } while (0)
#else
    #define ENIGMA_ASSERT(cond) ((void)0)
#endif

#if ENIGMA_DEBUG
    #define ENIGMA_VERIFY(cond)                                                \
        do {                                                                   \
            if (!(cond)) {                                                     \
                std::fprintf(stderr,                                           \
                    "[enigma] VERIFY failed: %s\n  at %s:%d\n",                \
                    #cond, __FILE__, __LINE__);                                \
                ENIGMA_DEBUGBREAK();                                           \
            }                                                                  \
        } while (0)
#else
    #define ENIGMA_VERIFY(cond) ((void)(cond))
#endif

#if ENIGMA_DEBUG
    #define ENIGMA_VK_CHECK(expr)                                              \
        do {                                                                   \
            const VkResult _enigma_vk_res = (expr);                            \
            if (_enigma_vk_res != VK_SUCCESS) {                                \
                std::fprintf(stderr,                                           \
                    "[enigma] VK_CHECK failed: %s\n  VkResult = %d\n  at %s:%d\n", \
                    #expr, static_cast<int>(_enigma_vk_res),                   \
                    __FILE__, __LINE__);                                       \
                ENIGMA_DEBUGBREAK();                                           \
            }                                                                  \
        } while (0)
#else
    #define ENIGMA_VK_CHECK(expr) ((void)(expr))
#endif
