#pragma once

#include "core/Types.h"
#include "engine/Clock.h"
#include "input/Input.h"
#include "physics/PhysicsWorld.h"
#include "physics/PhysicsInterpolation.h"
#include "physics/VehicleController.h"
#include "physics/VehicleConfig.h"
#include "platform/Window.h"
#include "renderer/Renderer.h"

#include <memory>

namespace enigma {

class Engine {
public:
    Engine();
    ~Engine();

    Engine(const Engine&)            = delete;
    Engine& operator=(const Engine&) = delete;
    Engine(Engine&&)                 = delete;
    Engine& operator=(Engine&&)      = delete;

    Window&       window()   { return m_window;   }
    Renderer&     renderer() { return m_renderer; }
    Clock&        clock()    { return m_clock;    }
    Input&        input()    { return m_input;    }
    PhysicsWorld& physics()  { return *m_physicsWorld; }

    VehicleController*     vehicle()      { return m_vehicle.get(); }
    PhysicsInterpolation&  interpolation() { return m_interpolation; }

private:
    Window   m_window;
    Renderer m_renderer;
    Input    m_input;
    Clock    m_clock;

    std::unique_ptr<PhysicsWorld>      m_physicsWorld;
    std::unique_ptr<VehicleController> m_vehicle;
    PhysicsInterpolation               m_interpolation;
};

} // namespace enigma
