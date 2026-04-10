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
    , m_input(m_window)
    , m_clock() {
    m_physicsWorld = std::make_unique<PhysicsWorld>();

    // Spawn vehicle at a default position.
    m_vehicle = std::make_unique<VehicleController>(
        *m_physicsWorld, VehicleConfig::makeDefault(), vec3(0.0f, 0.5f, 0.0f));

    ENIGMA_LOG_INFO("[engine] constructed ({}x{})", kDefaultWindowWidth, kDefaultWindowHeight);
}

Engine::~Engine() {
    // Destroy vehicle before physics world.
    m_vehicle.reset();
    m_physicsWorld.reset();
    ENIGMA_LOG_INFO("[engine] shutdown");
}

} // namespace enigma
