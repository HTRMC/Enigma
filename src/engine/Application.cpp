#include "engine/Application.h"

#include "asset/GltfLoader.h"
#include "core/Log.h"
#include "core/Paths.h"
#include "engine/Engine.h"
#include "input/Input.h"
#include "physics/PhysicsWorld.h"
#include "physics/VehicleController.h"
#include "platform/Window.h"
#include "renderer/Renderer.h"
#include "scene/Camera.h"
#include "scene/CameraController.h"
#include "scene/Scene.h"

#include <volk.h>

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
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

        // Physics step.
        engine.physics().step(dt);

        // Vehicle input from keyboard (WASD) and gamepad.
        {
            VehicleInput vi;
            if (input.isKeyDown(GLFW_KEY_W)) vi.throttle = 1.0f;
            if (input.isKeyDown(GLFW_KEY_S)) vi.brake    = 1.0f;
            if (input.isKeyDown(GLFW_KEY_A)) vi.steering  = -0.7f;
            if (input.isKeyDown(GLFW_KEY_D)) vi.steering  =  0.7f;
            if (input.isKeyDown(GLFW_KEY_SPACE)) vi.handbrake = true;

            // Gamepad override (if present).
            if (input.isGamepadPresent()) {
                const f32 gpThrottle = input.getGamepadAxis(0, Input::GAMEPAD_AXIS_RIGHT_TRIGGER);
                const f32 gpBrake    = input.getGamepadAxis(0, Input::GAMEPAD_AXIS_LEFT_TRIGGER);
                const f32 gpSteer    = input.getGamepadAxis(0, Input::GAMEPAD_AXIS_LEFT_X);
                // Triggers are in [-1,1]; remap to [0,1].
                if (gpThrottle > 0.0f) vi.throttle = std::max(vi.throttle, (gpThrottle + 1.0f) * 0.5f);
                if (gpBrake    > 0.0f) vi.brake    = std::max(vi.brake,    (gpBrake    + 1.0f) * 0.5f);
                if (std::abs(gpSteer) > 0.1f) vi.steering = gpSteer;
            }

            engine.vehicle()->setInput(vi);
            engine.vehicle()->update(dt);
        }

        // Physics interpolation snapshot.
        engine.interpolation().snapshot(engine.vehicle()->bodyId(), engine.physics());

        // Update scene nodes bound to physics bodies.
        if (scene.has_value()) {
            const f32 alpha = engine.physics().accumulator() / PhysicsWorld::kFixedDt;
            for (auto& node : scene->nodes) {
                if (node.physicsBodyId != 0xFFFFFFFFu) {
                    node.worldTransform = engine.interpolation().interpolatedTransform(
                        node.physicsBodyId, alpha);
                }
            }
        }

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
