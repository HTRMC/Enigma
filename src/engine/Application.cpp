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

    // Streaming heightfield — rebuilt when the car exits the covered area.
    // terrainHeight() must stay in sync with terrain_clipmap.hlsl terrainHeight().
    auto terrainHeight = [](float wx, float wz) -> float {
        return std::sin(wx * 0.05f) * std::cos(wz * 0.05f) * 2.0f
             + std::sin(wx * 0.13f + 1.1f) * std::sin(wz * 0.09f) * 0.8f;
    };
    // 512 samples × 1.002 m spacing → 512 m total coverage per side.
    // Rebuild when car is >150 m from the current HF centre (106 m edge buffer).
    // New centre is snapped to a 64 m grid to prevent thrashing.
    constexpr u32 kHFN       = 512;
    constexpr f32 kHFSize    = 512.0f;
    constexpr f32 kHFHalf    = kHFSize * 0.5f;
    constexpr f32 kHFRebuild = 150.0f;
    constexpr f32 kHFSnap    = 64.0f;

    u32  hfBodyId = ~0u;
    vec2 hfCenter = {0.0f, 0.0f};

    auto rebuildHeightField = [&](vec2 centre) {
        if (hfBodyId != ~0u) {
            engine.physics().removeBody(hfBodyId);
        }
        const f32 spacing = kHFSize / static_cast<f32>(kHFN - 1);
        const f32 oriX    = centre.x - kHFHalf;
        const f32 oriZ    = centre.y - kHFHalf;
        std::vector<f32> heights(kHFN * kHFN);
        for (u32 row = 0; row < kHFN; ++row) {
            for (u32 col = 0; col < kHFN; ++col) {
                heights[row * kHFN + col] = terrainHeight(oriX + col * spacing,
                                                          oriZ + row * spacing);
            }
        }
        hfBodyId = engine.physics().addHeightField(vec3(oriX, 0.0f, oriZ), kHFSize, kHFN, heights);
        hfCenter = centre;
        ENIGMA_LOG_INFO("[app] heightfield rebuilt at ({:.0f}, {:.0f})", centre.x, centre.y);
    };

    rebuildHeightField({0.0f, 0.0f});

    bool f3PrevDown = false;

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

        // Per-substep physics loop: snapshot BEFORE each step so prev/curr are
        // always the last two consecutive 8.33 ms physics states. This gives
        // smooth, jitter-free interpolation regardless of render frame time.
        engine.physics().addDt(dt);
        while (engine.physics().canStep()) {
            engine.interpolation().snapshot(engine.vehicle()->bodyId(), engine.physics());
            engine.physics().stepFixed();
        }
        engine.interpolation().updateCurr(engine.vehicle()->bodyId(), engine.physics());

        // Stream heightfield: rebuild when car moves >kHFRebuild m from current centre.
        {
            const vec4 carPosH = engine.vehicle()->bodyTransform()[3];
            const vec2 carXZ   = {carPosH.x, carPosH.z};
            if (glm::length(carXZ - hfCenter) > kHFRebuild) {
                const vec2 snapped = {
                    std::floor(carXZ.x / kHFSnap) * kHFSnap,
                    std::floor(carXZ.y / kHFSnap) * kHFSnap,
                };
                rebuildHeightField(snapped);
            }
        }

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

        // F3 edge-detect toggle for physics debug wireframe overlay.
        {
            const bool f3Down = input.isKeyDown(GLFW_KEY_F3);
            if (f3Down && !f3PrevDown) {
                renderer.physicsDebugRenderer().enabled =
                    !renderer.physicsDebugRenderer().enabled;
            }
            f3PrevDown = f3Down;
        }

        // Gather wireframe geometry this frame (clear + DrawBodies via Jolt).
        if (renderer.physicsDebugRenderer().enabled) {
            renderer.physicsDebugRenderer().gather(engine.physics().system());
        }

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
