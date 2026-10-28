// Three-Tank: port of RWTH's three-tank rig (B102 -> B103 -> B104), Torricelli throttles,
// identified pump and valve maps, outer PI on the B104 level error feeding a fixed inner pump-flow
// PI loop, forward Euler at 0.1 s logged every 0.2 s from the 5 s set-point step on (T = 476).
// Tank levels saturate, so it never crashes. Stage cost (V_ref - V4_meas)^2.
#pragma once

#include <cmath>

#include "esbo/problem.hpp"

namespace esbo {
namespace threetank {

// np.interp with clamped ends.
inline double interp1(double x, const double* xp, const double* fp, int n) {
    if (!(x > xp[0])) return fp[0];
    if (x >= xp[n - 1]) return fp[n - 1];
    int lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (xp[mid] <= x) lo = mid; else hi = mid;
    }
    if (xp[lo] == x) return fp[lo];
    const double slope = (fp[lo + 1] - fp[lo]) / (xp[lo + 1] - xp[lo]);
    return slope * (x - xp[lo]) + fp[lo];
}

inline double clipv(double x, double lo, double hi) { return x < lo ? lo : (x > hi ? hi : x); }

// ---- constants (experiments/threetank_pi/system.py:19-80) -----------------------------------
constexpr double DT = 0.1;                    // fixed step [s]
constexpr int N_STEPS = 1000;                 // int(round(T_SIM / DT)), T_SIM = 100 s
constexpr int LOG_EVERY = 2;                  // int(round(T_OBJ / DT)),  T_OBJ = 0.2 s
constexpr int REF_START = 50;                 // int(round(T_REF_JUMP / DT)), 5 s
constexpr int MAX_EPISODE_LENGTH = 476;       // logged samples in [5 s, 100 s]

constexpr double A_TANK = 0.029;
constexpr double A_PIPE_B102_B103 = 600e-6;
constexpr double A_PIPE_B103_B104 = 600e-6;
constexpr double V2_INIT = 1.48e-3;
constexpr double V3_INIT = 1.42e-3;
constexpr double V4_INIT = 1.255e-3;
constexpr double V_SAT_HIGH = 10e-3;
constexpr double V_SAT_LOW = 0.0;
constexpr double G = 9.81;

constexpr double VALVE_B103_B104 = 100.0;
constexpr double PUMP_LAG_TAU = 1.2341;
constexpr int DELAY_STEPS = 11;               // int(round(PUMP_DELAY / DT)), 1.1 s

constexpr double INNER_P = 30e3;
constexpr double INNER_I = 0.6e3;
constexpr double PUMP_OFFSET = 50.0;
constexpr double REFERENCE_AFTER = 4e-3 * 0.75;   // = 3e-3
constexpr double REFERENCE_BEFORE = 0.0;

// pump map: Q_P101 [m^3/s] = f(U_P101 [%], B101 volume [m^3])
inline const double PUMP_MAP_U[11] = {50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100};
inline const double PUMP_MAP_VOL[2] = {40e-3, 130e-3};
inline const double PUMP_MAP_Q0[11] = {0, 0.0606e-3, 0.2450e-3, 0.3378e-3, 0.4184e-3, 0.5002e-3,
                                       0.5801e-3, 0.6551e-3, 0.7138e-3, 0.7746e-3, 0.8206e-3};
inline const double PUMP_MAP_Q1[11] = {0, 0.2799e-3, 0.3497e-3, 0.4278e-3, 0.5053e-3, 0.5810e-3,
                                       0.6481e-3, 0.7168e-3, 0.7695e-3, 0.8350e-3, 0.8786e-3};

// valve V101 map: effective % = f(U_V101 %)
inline const double VALVE_V101_IN[17] = {0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80,
                                         85, 90, 100};
inline const double VALVE_V101_OUT[17] = {100, 96, 90, 88, 84, 76, 68, 60, 49, 41, 34, 27, 22, 19,
                                          16, 12, 0};

// valve V106 outflow map: Q_out [m^3/s] = f(U_V106 [%], V4 [m^3]). The V4 = 2e-3 column is ZERO,
// the 2.8e-3 and 10e-3 columns are the same curve (system.py:65-68).
inline const double VALVE_V106_U[15] = {15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85};
inline const double VALVE_V106_V4[3] = {2e-3, 2.8e-3, 10e-3};
inline const double VALVE_V106_COL[15] = {0, 0.032e-3, 0.097e-3, 0.179e-3, 0.262e-3, 0.361e-3,
                                          0.475e-3, 0.604e-3, 0.902e-3, 1.089e-3, 1.318e-3,
                                          1.536e-3, 1.685e-3, 1.853e-3, 2.029e-3};
inline const double VALVE_V106_ZERO[15] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

inline double signed_sqrt(double x) {
    if (x > 0.0) return std::sqrt(x);
    if (x < 0.0) return -std::sqrt(-x);
    return 0.0;                                  // np.sign(0) * sqrt(0)
}

inline double throttle_flow(double valve_pct, double v_up, double v_down, double a_pipe) {
    const double dh = (v_up - v_down) / A_TANK;
    return (valve_pct / 100.0) * a_pipe * signed_sqrt(2.0 * G * dh);
}

inline double pump_flow(double u_p101, double b101_volume) {
    const double u = clipv(u_p101, PUMP_MAP_U[0], PUMP_MAP_U[10]);
    const double q_cols[2] = {interp1(u, PUMP_MAP_U, PUMP_MAP_Q0, 11),
                              interp1(u, PUMP_MAP_U, PUMP_MAP_Q1, 11)};
    // b101_volume is NOT clipped in the Python -- np.interp clamps it for free.
    return interp1(b101_volume, PUMP_MAP_VOL, q_cols, 2);
}

inline double v106_outflow(double v4, double u_v106) {
    const double v4c = clipv(v4, VALVE_V106_V4[0], VALVE_V106_V4[2]);
    const double u = clipv(u_v106, VALVE_V106_U[0], VALVE_V106_U[14]);
    const double q_cols[3] = {interp1(u, VALVE_V106_U, VALVE_V106_ZERO, 15),
                              interp1(u, VALVE_V106_U, VALVE_V106_COL, 15),
                              interp1(u, VALVE_V106_U, VALVE_V106_COL, 15)};
    return interp1(v4c, VALVE_V106_V4, q_cols, 3);
}

inline constexpr ParamSpec PHI_TABLE[3] = {
    bounded("V101_opening", 10.0, 85.0),
    bounded("V106_opening", 10.0, 90.0),
    bounded("B101_volume", 0.040, 0.130),
};
inline constexpr NoiseSpec NOISE_TABLE[1] = {{"V4", 1.7e-4}};
inline constexpr double THETA_LO_T[2] = {0.0, 0.0};
inline constexpr double THETA_HI_T[2] = {3.0, 0.1};
inline const char* const THETA_NAMES_T[2] = {"K_P", "K_I"};
inline const char* const CHANNELS_T[3] = {"V2", "V3", "V4"};

struct Problem {
    static constexpr const char* NAME = "threetank_pi";
    static constexpr int D = 2, T = 476, C = 3, CR = 0;
    static constexpr int N_PHI = 3, N_EXTRA = 0, N_NOISE = 1;
    static constexpr bool USES_PHI_HAT = false, CAN_CRASH = false, HAS_SUCCESS = false;
    static constexpr const ParamSpec* PHI = PHI_TABLE;
    static constexpr const ParamSpec* EXTRA = nullptr;
    static constexpr const NoiseSpec* NOISE = NOISE_TABLE;
    static constexpr const double* THETA_LO = THETA_LO_T;
    static constexpr const double* THETA_HI = THETA_HI_T;
    static const char* const* theta_names() { return THETA_NAMES_T; }
    static const char* const* channels() { return CHANNELS_T; }
    static const char* const* rate_channels() { return nullptr; }
    // The nominal plant uses the identified rig values rather than the domain centers.
    static double phi_nominal(int i) { return NOMINAL_PHI[i]; }
    static double extra_nominal(int) { return 0.0; }
    static bool extra_active(int, const PriorOpts&) { return true; }
    static constexpr double NOMINAL_PHI[3] = {43.3, 44.7, 80e-3};

    struct Cache {};
    static void prepare(const Instance&, Cache&) {}

    template <class R>
    static Outcome simulate(const double* theta, const Instance& in, const Cache&, R& rng,
                            const SimOpts&, float* J, float* x, float*) {
        const double kp = theta[0], ki = theta[1];
        const double u_v101 = in.phi[0], u_v106 = in.phi[1], b101_volume = in.phi[2];
        const double valve_v101_eff = interp1(u_v101, VALVE_V101_IN, VALVE_V101_OUT, 17);
        const double sigma = in.noise[0];
        double outer_integ = 0.0, inner_integ = 0.0;
        double v2 = V2_INIT, v3 = V3_INIT, v4 = V4_INIT;
        double pump_line[DELAY_STEPS] = {0.0};
        int head = 0, n = 0;
        double pump_lag = 0.0, q_pump = 0.0, acc = 0.0;
        for (int k = 0; k <= N_STEPS; ++k) {
            const double ref = (k >= REF_START) ? REFERENCE_AFTER : REFERENCE_BEFORE;
            const double v4_meas = v4 + (sigma > 0.0 ? sigma * rng.normal() : 0.0);
            if (k >= REF_START && ((k - REF_START) % LOG_EVERY == 0)) {
                const float v4_32 = static_cast<float>(v4_meas);
                const float d = static_cast<float>(REFERENCE_AFTER) - v4_32;
                acc += static_cast<double>(d * d);
                J[n] = static_cast<float>(acc);
                x[n * C + 0] = static_cast<float>(v2);
                x[n * C + 1] = static_cast<float>(v3);
                x[n * C + 2] = v4_32;
                ++n;
            }
            if (k == N_STEPS) break;
            const double e_out = ref - v4_meas;
            const double o = kp * e_out + ki * outer_integ;
            outer_integ += DT * e_out;
            const double e_in = o - q_pump;
            const double u_p101 = PUMP_OFFSET + (INNER_P * e_in + INNER_I * inner_integ);
            inner_integ += DT * e_in;
            pump_lag += DT * (pump_flow(u_p101, b101_volume) - pump_lag) / PUMP_LAG_TAU;
            q_pump = pump_line[head];
            pump_line[head] = pump_lag;
            head = (head + 1) % DELAY_STEPS;
            const double q_23 = throttle_flow(valve_v101_eff, v2, v3, A_PIPE_B102_B103);
            const double q_34 = throttle_flow(VALVE_B103_B104, v3, v4, A_PIPE_B103_B104);
            const double q_out = v106_outflow(v4, u_v106);
            v2 = clipv(v2 + DT * (q_pump - q_23), V_SAT_LOW, V_SAT_HIGH);
            v3 = clipv(v3 + DT * (q_23 - q_34), V_SAT_LOW, V_SAT_HIGH);
            v4 = clipv(v4 + DT * (q_34 - q_out), V_SAT_LOW, V_SAT_HIGH);
        }
        Outcome out;
        finish_trajectory(T, C, CR, n, 0, J, x, nullptr);
        return out;
    }
};

}  // namespace threetank
}  // namespace esbo
