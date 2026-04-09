#pragma once

#include "core/Types.h"
#include "engine/Clock.h"
#include "input/Input.h"
#include "platform/Window.h"
#include "renderer/Renderer.h"

namespace enigma {

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
    Input&    input()    { return m_input;    }

private:
    Window   m_window;
    Renderer m_renderer;
    Input    m_input;
    Clock    m_clock;
};

} // namespace enigma
