// What every problem header provides, and the helpers shared by the simulators.
//
// A problem P is a struct with
//   NAME, D, T, C (level channels), CR (measured rate channels, 0 if none),
//   N_PHI, N_EXTRA, N_NOISE, USES_PHI_HAT, CAN_CRASH, HAS_SUCCESS,
//   PHI / EXTRA (ParamSpec tables), NOISE (NoiseSpec table), THETA_LO / THETA_HI / THETA_NAMES,
//   CHANNELS / RATE_CHANNELS, a per-instance `Cache` with prepare(), and
//   simulate(theta, instance, cache, rng, opts, J, x, xdot) -> Outcome.
// simulate() writes the full horizon: J[T] with +inf from the crash step on, x[T*C] and
// xdot[T*CR] row-major, rows after the last simulated step repeating it.
#pragma once

#include <cmath>
#include <cstring>
#include <limits>

#include "prior.hpp"

namespace esbo {

struct Outcome {
    int crash_step = 0;      // 1-based step of the crash, 0 if the trial did not crash
    bool success = true;     // not crashed and the success criterion (if any) holds
};

struct SimOpts {
    // Start from the nominal reset state without random jitter or random kick sign. Used with
    // zero-noise instances for deterministic rollouts (parity tests).
    bool deterministic = false;
    // Optional start state (Cartpole: the reset state, Furuta: the braked state before the
    // kick). Validation only.
    const double* x0 = nullptr;
};

// sin and cos through separate libm calls. Compilers fuse sin(a) and cos(a) into one sincos call,
// which can differ from the separate calls in the last bit. The simulators use these so their
// trajectories match the Python simulators (which call math.sin / math.cos) step by step.
inline double libm_sin(double a) {
    static double (*volatile fn)(double) = static_cast<double (*)(double)>(std::sin);
    return fn(a);
}
inline double libm_cos(double a) {
    static double (*volatile fn)(double) = static_cast<double (*)(double)>(std::cos);
    return fn(a);
}

enum Category { CAT_SUCCESS = 0, CAT_CRASH = 1, CAT_UNSUCCESSFUL = 2 };

inline int category(const Outcome& o) {
    if (o.crash_step > 0) return CAT_CRASH;
    return o.success ? CAT_SUCCESS : CAT_UNSUCCESSFUL;
}

// Set J to +inf from the crash step on and repeat the last simulated row of x / xdot.
inline void finish_trajectory(int T, int C, int CR, int n_sim, int crash_step, float* J, float* x,
                              float* xdot) {
    if (crash_step > 0)
        for (int t = crash_step - 1; t < T; ++t) J[t] = std::numeric_limits<float>::infinity();
    for (int t = n_sim; t < T; ++t) {
        J[t] = crash_step > 0 ? std::numeric_limits<float>::infinity() : J[n_sim - 1];
        if (C > 0) std::memcpy(x + std::size_t(t) * C, x + std::size_t(n_sim - 1) * C, sizeof(float) * C);
        if (CR > 0 && xdot != nullptr)
            std::memcpy(xdot + std::size_t(t) * CR, xdot + std::size_t(n_sim - 1) * CR,
                        sizeof(float) * CR);
    }
}

}  // namespace esbo
