#include "renderer/UpscalerSettings.h"

#include "renderer/UpscalerFactory.h"

namespace enigma {

UpscalerBackend UpscalerSettings::effectiveBackend(const VkPhysicalDeviceProperties& props,
                                                   gfx::GpuTier tier) const {
    if (autoSelect || backend == UpscalerBackend::None) {
        return UpscalerFactory::autoSelect(props, tier);
    }
    return backend;
}

} // namespace enigma
