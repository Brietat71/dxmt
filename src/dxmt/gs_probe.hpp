// Instrumentation jetable (branche gs-probe2, ne pas fusionner) : sorties dans GS_PROBE_DIR, chemin codé en dur
// parce que la bouteille ne laisse pas passer de variable d'environnement sans toucher à l'Info.plist.
#pragma once
#include <atomic>
#include <cstdint>
#include <string>

#define GS_PROBE_DIR std::string("Z:\\Users\\brintet\\code\\dxmt-gs-data\\probe2\\")

namespace dxmt {
inline std::atomic<uint64_t> gs_probe_frames{0}; // Present
inline std::atomic<uint64_t> gs_probe_draws{0};  // tous les draws (PreDraw)
} // namespace dxmt
