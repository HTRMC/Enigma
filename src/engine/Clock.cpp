#include "engine/Clock.h"

namespace enigma {

Clock::Clock()
    : m_start(SteadyClock::now())
    , m_last(m_start) {}

f64 Clock::tick() {
    const TimePoint now = SteadyClock::now();
    const std::chrono::duration<f64> delta = now - m_last;
    m_last = now;
    return delta.count();
}

f64 Clock::elapsed() const {
    const std::chrono::duration<f64> total = SteadyClock::now() - m_start;
    return total.count();
}

} // namespace enigma
