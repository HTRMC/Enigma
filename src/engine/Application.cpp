#include "engine/Application.h"

#include "core/Log.h"
#include "core/Paths.h"
#include "engine/Engine.h"
#include "platform/Window.h"
#include "renderer/Renderer.h"

namespace enigma {

Application::Application()  = default;
Application::~Application() = default;

int Application::run(int argc, char** argv) {
    // Wire the Paths helper first so any later subsystem that needs to
    // resolve files relative to the executable (shaders, etc.) works even
    // if the binary was launched from an unrelated CWD.
    const char* argv0 = (argc > 0 && argv != nullptr) ? argv[0] : nullptr;
    Paths::init(argv0);

    ENIGMA_LOG_INFO("[app] starting, exe = {}", Paths::executablePath().string());

    Engine engine;
    auto& window   = engine.window();
    auto& renderer = engine.renderer();
    auto& clock    = engine.clock();

    while (!window.shouldClose()) {
        window.pollEvents();
        (void)clock.tick();
        renderer.drawFrame();
    }

    ENIGMA_LOG_INFO("[app] shutdown");
    return 0;
}

} // namespace enigma
