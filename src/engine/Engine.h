#pragma once

#include "core/Types.h"
#include "engine/Clock.h"
#include "platform/Window.h"
#include "renderer/Renderer.h"

namespace enigma {

// Owns the long-lived subsystems. Construction order = `Window -> Renderer
// -> Clock`; destruction is reverse. The Engine does not own the Application
// loop; Application drives frame stepping.
class Engine {
public:
    Engine();
    ~Engine();

    Engine(const Engine&)            = delete;
    Engine& operator=(const Engine&) = delete;
    Engine(Engine&&)                 = delete;
    Engine& operator=(Engine&&)      = delete;

    Window&   window()   { return m_window;   }
    Renderer& renderer() { return m_renderer; }
    Clock&    clock()    { return m_clock;    }

private:
    Window   m_window;
    Renderer m_renderer;
    Clock    m_clock;
};

} // namespace enigma
