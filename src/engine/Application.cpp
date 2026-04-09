#include "engine/Application.h"

#include "asset/GltfLoader.h"
#include "core/Log.h"
#include "core/Paths.h"
#include "engine/Engine.h"
#include "input/Input.h"
#include "platform/Window.h"
#include "renderer/Renderer.h"
#include "scene/Camera.h"
#include "scene/CameraController.h"
#include "scene/Scene.h"

#include <volk.h>

#include <filesystem>
#include <optional>
#include <string>

namespace enigma {

Application::Application()  = default;
Application::~Application() = default;

int Application::run(int argc, char** argv) {
    const char* argv0 = (argc > 0 && argv != nullptr) ? argv[0] : nullptr;
    Paths::init(argv0);

    ENIGMA_LOG_INFO("[app] starting, exe = {}", Paths::executablePath().string());

    Engine engine;
    auto& window   = engine.window();
    auto& renderer = engine.renderer();
    auto& clock    = engine.clock();
    auto& input    = engine.input();

    // Camera: positioned to see the origin, looking forward.
    Camera camera({0.0f, 1.0f, 3.0f}, 60.0f, 0.1f);
    CameraController controller(camera, input);
    renderer.setCamera(&camera);

    // glTF scene loading: command-line path or bundled default.
    std::optional<Scene> scene;
    std::filesystem::path modelPath;
    if (argc > 1 && argv[1] != nullptr) {
        modelPath = argv[1];
    } else {
        modelPath = Paths::executablePath().parent_path() / "assets" / "DamagedHelmet.glb";
    }

    scene = loadGltf(modelPath,
                     renderer.device(),
                     renderer.allocator(),
                     renderer.descriptorAllocator());
    if (scene.has_value()) {
        renderer.setScene(&scene.value());
        ENIGMA_LOG_INFO("[app] loaded scene from: {}", modelPath.string());
    } else {
        ENIGMA_LOG_WARN("[app] failed to load scene: {}, falling back to triangle",
                        modelPath.string());
    }

    while (!window.shouldClose()) {
        window.pollEvents();
        const f32 dt = static_cast<f32>(clock.tick());
        input.update();
        controller.update(dt);
        renderer.drawFrame();
    }

    // Clean up scene before renderer teardown.
    if (scene.has_value()) {
        renderer.setScene(nullptr);
        // Wait for GPU to finish before destroying scene resources.
        vkDeviceWaitIdle(renderer.device().logical());
        scene->destroy(renderer.device(), renderer.allocator());
    }

    ENIGMA_LOG_INFO("[app] shutdown");
    return 0;
}

} // namespace enigma
