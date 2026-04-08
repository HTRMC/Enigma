#pragma once

#include "core/Types.h"

#include <chrono>

namespace enigma {

// Steady-clock driven frame delta. `tick()` returns the seconds elapsed
// since the previous `tick()` call (or since construction for the first).
class Clock {
public:
    Clock();

    // Returns delta-seconds since the previous tick.
    f64 tick();

    // Seconds since construction (not affected by tick()).
    f64 elapsed() const;

private:
    using SteadyClock = std::chrono::steady_clock;
    using TimePoint   = SteadyClock::time_point;

    TimePoint m_start;
    TimePoint m_last;
};

} // namespace enigma
