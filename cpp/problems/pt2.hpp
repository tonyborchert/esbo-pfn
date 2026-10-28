// PT2 dead-time: G(s) = K w0^2 / (s^2 + 2 D w0 s + w0^2) with an input dead time, a parallel PID
// with derivative filter (N = 100) and log10 gains, unit step reference, forward Euler at 0.01 s,
// logged every 0.1 s (T = 501 logged steps). The dead-time line holds 11 base steps as in the
// original implementation. Crash: measured overshoot y - w > 2 at a logged step. Stage cost ITAE
// tau_t |w - y_t| with tau_t = (t - 1) 0.1 s.
#pragma once

#include <cmath>

#include "esbo/problem.hpp"

namespace esbo {
namespace pt2 {

constexpr double DT = 0.01;
constexpr double T_OBJ = 0.1;
constexpr int N_STEPS = 5000;
constexpr int LOG_EVERY = 10;
constexpr int DEAD_LINE = 10;
constexpr double PID_N = 100.0;
constexpr double REFERENCE = 1.0;
constexpr double OVERSHOOT_LIMIT = 2.0;

inline constexpr ParamSpec PHI_TABLE[3] = {
    scale("K", 4.0, 5.0, 5.0),
    scale("w0", 0.5, 5.0, 5.0),
    scale("D", 0.5, 5.0, 5.0),
};
inline constexpr NoiseSpec NOISE_TABLE[1] = {{"y", 4e-5}};
inline constexpr double THETA_LO_T[3] = {-1.0, -1.0, -1.0};
inline constexpr double THETA_HI_T[3] = {1.0, 1.0, 1.0};
inline const char* const THETA_NAMES_T[3] = {"log10_K_P", "log10_K_I", "log10_K_D"};
inline const char* const CHANNELS_T[1] = {"y"};

struct Problem {
    static constexpr const char* NAME = "pt2_deadtime";
    static constexpr int D = 3, T = 501, C = 1, CR = 0;
    static constexpr int N_PHI = 3, N_EXTRA = 0, N_NOISE = 1;
    static constexpr bool USES_PHI_HAT = false, CAN_CRASH = true, HAS_SUCCESS = false;
    static constexpr const ParamSpec* PHI = PHI_TABLE;
    static constexpr const ParamSpec* EXTRA = nullptr;
    static constexpr const NoiseSpec* NOISE = NOISE_TABLE;
    static constexpr const double* THETA_LO = THETA_LO_T;
    static constexpr const double* THETA_HI = THETA_HI_T;
    static const char* const* theta_names() { return THETA_NAMES_T; }
    static const char* const* channels() { return CHANNELS_T; }
    static const char* const* rate_channels() { return nullptr; }
    static double phi_nominal(int i) { return PHI_TABLE[i].a; }
    static double extra_nominal(int) { return 0.0; }
    static bool extra_active(int, const PriorOpts&) { return true; }

    struct Cache {};
    static void prepare(const Instance&, Cache&) {}

    template <class R>
    static Outcome simulate(const double* theta, const Instance& in, const Cache&, R& rng,
                            const SimOpts&, float* J, float* x, float*) {
        const double K = in.phi[0], w0 = in.phi[1], Dm = in.phi[2];
        const double kp = std::pow(10.0, theta[0]);
        const double ki = std::pow(10.0, theta[1]);
        const double kd = std::pow(10.0, theta[2]);
        const double sigma = in.noise[0];

        double integ = 0.0, dstate = 0.0, y = 0.0, ydot = 0.0;
        double ubuf[DEAD_LINE] = {0.0};
        int head = 0, crash = 0, n = 0;
        double acc = 0.0;
        for (int k = 0; k <= N_STEPS; ++k) {
            const double y_meas = y + (sigma > 0.0 ? sigma * rng.normal() : 0.0);
            if (k % LOG_EVERY == 0) {
                const int i = k / LOG_EVERY;
                const float y32 = static_cast<float>(y_meas);
                const float err = std::fabs(static_cast<float>(REFERENCE) - y32);
                acc += static_cast<double>(
                    static_cast<float>(static_cast<double>(err) * (static_cast<double>(i) * T_OBJ)));
                J[i] = static_cast<float>(acc);
                x[i] = y32;
                n = i + 1;
                if (y_meas - REFERENCE > OVERSHOOT_LIMIT || !std::isfinite(y_meas)) {
                    crash = i + 1;
                    break;
                }
            }
            if (k == N_STEPS) break;
            const double e = REFERENCE - y_meas;
            const double d_out = kd * PID_N * (e - dstate);
            const double u = kp * e + ki * integ + d_out;
            integ += DT * e;
            dstate += DT * PID_N * (e - dstate);
            const double u_delayed = ubuf[head];
            ubuf[head] = u;
            head = (head + 1) % DEAD_LINE;
            ydot += DT * (K * w0 * w0 * u_delayed - 2.0 * Dm * w0 * ydot - w0 * w0 * y);
            y += DT * ydot;
        }
        Outcome out;
        out.crash_step = crash;
        out.success = crash == 0;
        finish_trajectory(T, C, CR, n, crash, J, x, nullptr);
        return out;
    }
};

}  // namespace pt2
}  // namespace esbo
