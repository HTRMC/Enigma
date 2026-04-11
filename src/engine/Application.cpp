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
#include "scene/FollowCamera.h"
#include "scene/Scene.h"
#include "world/Terrain.h"

#include <volk.h>

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

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

    // Camera — position is overwritten each frame by the FollowCamera.
    Camera camera({0.0f, 3.0f, 8.0f}, 60.0f, 0.1f);
    renderer.setCamera(&camera);

    FollowCamera followCam(camera, /*armLength=*/8.0f, /*heightOffset=*/2.5f);

    // glTF scene loading: command-line path, BMW M4 GT3, or bundled default.
    std::optional<Scene> scene;
    std::filesystem::path modelPath;
    if (argc > 1 && argv[1] != nullptr) {
        modelPath = argv[1];
    } else {
        const std::filesystem::path bmwPath =
            Paths::executablePath().parent_path() / "assets" / "bmw_m4_gt3_nic.glb";
        if (std::filesystem::exists(bmwPath)) {
            modelPath = bmwPath;
        } else {
            modelPath = Paths::executablePath().parent_path() / "assets" / "DamagedHelmet.glb";
        }
    }

    scene = loadGltf(modelPath,
                     renderer.device(),
                     renderer.allocator(),
                     renderer.descriptorAllocator());
    // Rest transforms: record each node's initial GLB-space transform, with a
    // corrective Y-rotation applied so that the model's nose (+X in GLB space
    // for most car exports) aligns with the physics body's forward (+Z in Jolt).
    // Each frame: node.worldTransform = vehiclePhysicsTransform * restTransform[i]
    std::vector<mat4> nodeRestTransforms;

    if (scene.has_value()) {
        renderer.setScene(&scene.value());
        ENIGMA_LOG_INFO("[app] loaded scene from: {}", modelPath.string());

        // BMW M4 GT3 GLB: front of car is along +X in model space.
        // Physics body drives in +Z.  Rotate -90° around Y to align them.
        const mat4 correction = glm::rotate(mat4(1.0f),
                                            glm::radians(-90.0f),
                                            vec3(0.0f, 1.0f, 0.0f));
        nodeRestTransforms.resize(scene->nodes.size());
        for (u32 i = 0; i < static_cast<u32>(scene->nodes.size()); ++i) {
            nodeRestTransforms[i] = correction * scene->nodes[i].worldTransform;
        }
        ENIGMA_LOG_INFO("[app] recorded {} node rest transforms (correction: -90 Y)",
                        nodeRestTransforms.size());
    } else {
        ENIGMA_LOG_WARN("[app] failed to load scene: {}, falling back to triangle",
                        modelPath.string());
    }

    // Build a Jolt heightfield that exactly matches the visual terrain shader.
    // 512×512 samples over a 512×512 m area centred at the origin.
    // terrainHeight() must stay in sync with terrain_clipmap.hlsl terrainHeight().
    auto terrainHeight = [](float wx, float wz) -> float {
        return std::sin(wx * 0.05f) * std::cos(wz * 0.05f) * 2.0f
             + std::sin(wx * 0.13f + 1.1f) * std::sin(wz * 0.09f) * 0.8f;
    };
    {
        constexpr u32 kN      = 512;
        constexpr f32 kHFSize = 512.0f;
        constexpr f32 kOriX   = -kHFSize * 0.5f;
        constexpr f32 kOriZ   = -kHFSize * 0.5f;
        const f32     spacing = kHFSize / static_cast<f32>(kN - 1);
        std::vector<f32> hfHeights(kN * kN);
        for (u32 row = 0; row < kN; ++row) {
            for (u32 col = 0; col < kN; ++col) {
                hfHeights[row * kN + col] = terrainHeight(kOriX + col * spacing,
                                                          kOriZ + row * spacing);
            }
        }
        engine.physics().addHeightField(vec3(kOriX, 0.0f, kOriZ), kHFSize, kN, hfHeights);
        ENIGMA_LOG_INFO("[app] built terrain heightfield ({}×{}, {:.1f}m)", kN, kN, kHFSize);
    }

    // GPU-driven clipmap terrain — built and wired into the G-buffer pass.
    Terrain terrain(renderer.device(),
                    renderer.allocator(),
                    renderer.descriptorAllocator());
    terrain.buildPipeline(renderer.shaderManager(),
                          renderer.descriptorAllocator().layout(),
                          /*colorFormat=*/       GBufferPass::kAlbedoFormat,
                          /*depthFormat=*/       GBufferPass::kDepthFormat,
                          /*normalFormat=*/      GBufferPass::kNormalFormat,
                          /*metalRoughFormat=*/  GBufferPass::kMetalRoughFormat,
                          /*motionVecFormat=*/   GBufferPass::kMotionVecFormat);
    terrain.registerHotReload(renderer.shaderHotReload());
    renderer.setTerrain(&terrain);

    while (!window.shouldClose()) {
        window.pollEvents();
        const f32 dt = static_cast<f32>(clock.tick());
        input.update();

        // Vehicle input must be applied BEFORE physics.step() so Jolt
        // consumes the impulses in the same fixed-timestep sub-steps.
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

        // Physics step — consumes impulses set above.
        engine.physics().step(dt);

        // Physics interpolation snapshot.
        engine.interpolation().snapshot(engine.vehicle()->bodyId(), engine.physics());

        // Interpolated car transform for both rendering and the follow camera.
        const f32  alpha = engine.physics().accumulator() / PhysicsWorld::kFixedDt;
        const mat4 carTransform = engine.interpolation().interpolatedTransform(
            engine.vehicle()->bodyId(), alpha);

        // Drive ALL car mesh nodes from the physics body transform.
        // Uses pre-recorded rest transforms so each node's model-space offset
        // is preserved while the whole car tracks the Jolt body.
        if (scene.has_value() && !nodeRestTransforms.empty()) {
            for (u32 i = 0; i < static_cast<u32>(scene->nodes.size()); ++i) {
                scene->nodes[i].worldTransform = carTransform * nodeRestTransforms[i];
            }
        }

        // Spring-arm follow camera tracks the interpolated car transform.
        followCam.update(carTransform, dt);

        // Rebuild terrain chunk positions for this frame.
        terrain.update(camera.position);

        renderer.drawFrame();
    }

    // Clean up scene before renderer teardown.
    renderer.setTerrain(nullptr);
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
