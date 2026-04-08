#pragma once

namespace enigma {

// Top-level entry point used by `main.cpp`. Owns the Engine for the
// duration of `run()` and drives the frame loop.
class Application {
public:
    Application();
    ~Application();

    Application(const Application&)            = delete;
    Application& operator=(const Application&) = delete;
    Application(Application&&)                 = delete;
    Application& operator=(Application&&)      = delete;

    int run(int argc, char** argv);
};

} // namespace enigma
