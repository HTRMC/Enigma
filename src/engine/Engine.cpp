#include "engine/Engine.h"

#include "core/Log.h"

namespace enigma {

namespace {
constexpr u32 kDefaultWindowWidth  = 1280;
constexpr u32 kDefaultWindowHeight = 720;
} // namespace

Engine::Engine()
    : m_window(kDefaultWindowWidth, kDefaultWindowHeight, "Enigma")
    , m_renderer(m_window)
    , m_clock() {
    ENIGMA_LOG_INFO("[engine] constructed ({}x{})", kDefaultWindowWidth, kDefaultWindowHeight);
}

Engine::~Engine() {
    ENIGMA_LOG_INFO("[engine] shutdown");
}

} // namespace enigma
