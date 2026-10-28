// Cartpole: Gymnasium cart-pole integrated with explicit Euler steps of 0.02 s, two parallel PD
// controllers on cart position error and pole angle with per-step error differences, bang-bang
// force of magnitude `max_force`. The position goal steps 0, +2, -2, +2, -2 m. The controller sees
// the previous step's measurement. Crash: cart beyond +-3.5 m or pole beyond +-30 degrees. Stage
// cost 10 tanh^2(e) + xdot^2 + phi^2 + phidot^2 on the measured channels, e = goal - x.
// Float widths follow the thesis-era Python: float64 physics, float32 measurements, float32
// controller errors with float64 gains (the controller is a sign test, so these matter).
#pragma once

#include <cmath>
#include <cstddef>

#include "esbo/problem.hpp"

namespace esbo {
namespace cartpole {

constexpr double GRAVITY = 9.8;
constexpr double TAU = 0.02;
constexpr double THETA_THRESHOLD = 30.0 * 2.0 * M_PI / 360.0;
constexpr double X_THRESHOLD = 3.5;
constexpr double RESET_HALF_WIDTH = 0.05;

inline constexpr ParamSpec PHI_TABLE[4] = {
    scale("cart_mass", 1.0, 8.0, 8.0),
    scale("pole_mass", 0.1, 8.0, 8.0),
    scale("pole_half_length", 0.5, 3.0, 3.0),
    scale("max_force", 10.0, 3.0, 3.0),
};
inline constexpr NoiseSpec NOISE_TABLE[4] = {
    {"x_cart", 0.004}, {"xdot_cart", 0.02}, {"phi_pole", 0.002}, {"phidot_pole", 0.02}};
inline constexpr double THETA_LO_T[4] = {0.3, 30.0, 10.0, 150.0};
inline constexpr double THETA_HI_T[4] = {3.0, 300.0, 100.0, 1500.0};
inline const char* const THETA_NAMES_T[4] = {"K_P_cart", "K_D_cart", "K_P_pole", "K_D_pole"};
inline const char* const CHANNELS_T[2] = {"x_cart", "phi_pole"};
inline const char* const RATE_CHANNELS_T[2] = {"xdot_cart", "phidot_pole"};

inline float goal_at(int i) {
    if (i < 100) return 0.0f;
    if (i < 300) return 2.0f;
    if (i < 600) return -2.0f;
    if (i < 800) return 2.0f;
    return -2.0f;
}

struct Env {
    double masscart, masspole, length, force_mag, total_mass, polemass_length;
    double s[4] = {0, 0, 0, 0};   // [x, x_dot, phi, phi_dot]

    explicit Env(const double* phi)
        : masscart(phi[0]), masspole(phi[1]), length(phi[2]), force_mag(phi[3]),
          total_mass(phi[1] + phi[0]), polemass_length(phi[1] * phi[2]) {}

    void step(int action) {
        const double xx = s[0], x_dot = s[1], th = s[2], th_dot = s[3];
        const double force = force_mag * action;
        const double costheta = libm_cos(th), sintheta = libm_sin(th);
        const double temp = (force + polemass_length * (th_dot * th_dot) * sintheta) / total_mass;
        const double thetaacc = (GRAVITY * sintheta - costheta * temp) /
                                (length * (4.0 / 3.0 - masspole * (costheta * costheta) / total_mass));
        const double xacc = temp - polemass_length * thetaacc * costheta / total_mass;
        s[0] = xx + TAU * x_dot;
        s[1] = x_dot + TAU * xacc;
        s[2] = th + TAU * th_dot;
        s[3] = th_dot + TAU * thetaacc;
    }

    bool terminated() const {
        return s[0] < -X_THRESHOLD || s[0] > X_THRESHOLD || s[2] < -THETA_THRESHOLD ||
               s[2] > THETA_THRESHOLD;
    }
};

// Errors are formed in float32 from the float32 measurements, gains multiply in float64.
struct PD {
    double kp, kd;
    float last_error = 0.0f;
    PD(double p, double d) : kp(p), kd(d) {}
    double observe(float v) {
        const float error = 0.0f - v;
        const float d_error = error - last_error;
        last_error = error;
        return kp * static_cast<double>(error) + kd * static_cast<double>(d_error);
    }
};

struct Problem {
    static constexpr const char* NAME = "cartpole";
    static constexpr int D = 4, T = 1000, C = 2, CR = 2;
    static constexpr int N_PHI = 4, N_EXTRA = 0, N_NOISE = 4;
    static constexpr bool USES_PHI_HAT = false, CAN_CRASH = true, HAS_SUCCESS = false;
    static constexpr const ParamSpec* PHI = PHI_TABLE;
    static constexpr const ParamSpec* EXTRA = nullptr;
    static constexpr const NoiseSpec* NOISE = NOISE_TABLE;
    static constexpr const double* THETA_LO = THETA_LO_T;
    static constexpr const double* THETA_HI = THETA_HI_T;
    static const char* const* theta_names() { return THETA_NAMES_T; }
    static const char* const* channels() { return CHANNELS_T; }
    static const char* const* rate_channels() { return RATE_CHANNELS_T; }
    static double phi_nominal(int i) { return PHI_TABLE[i].a; }
    static double extra_nominal(int) { return 0.0; }
    static bool extra_active(int, const PriorOpts&) { return true; }

    struct Cache {};
    static void prepare(const Instance&, Cache&) {}

    template <class R>
    static Outcome simulate(const double* theta, const Instance& in, const Cache&, R& rng,
                            const SimOpts& opts, float* J, float* x, float* xdot) {
        Env env(in.phi.data());
        PD cart(theta[0], theta[1]);
        PD pole(theta[2], theta[3]);
        const float sig[4] = {static_cast<float>(in.noise[0]), static_cast<float>(in.noise[1]),
                              static_cast<float>(in.noise[2]), static_cast<float>(in.noise[3])};
        for (int c = 0; c < 4; ++c) {
            const double u = rng.uniform(-RESET_HALF_WIDTH, RESET_HALF_WIDTH);
            env.s[c] = opts.x0 != nullptr ? opts.x0[c] : (opts.deterministic ? 0.0 : u);
        }

        bool terminated = false;
        int crash = 0, n = 0;
        float prev_err = 0.0f, prev_angle = 0.0f;
        double acc = 0.0;
        for (int i = 0; i < T; ++i) {
            if (i != 0) {
                const double u = cart.observe(prev_err);
                const double up = pole.observe(prev_angle);
                const int action = (up + u < 0.0) ? 1 : -1;
                env.step(action);
                terminated = terminated || env.terminated();
            }
            float o[4];
            for (int c = 0; c < 4; ++c)
                o[c] = static_cast<float>(rng.normal() * static_cast<double>(sig[c]));
            o[0] -= goal_at(i);
            for (int c = 0; c < 4; ++c) o[c] += static_cast<float>(env.s[c]);
            const float err = o[0];   // measured position relative to the goal
            x[i * C + 0] = err + goal_at(i);
            x[i * C + 1] = o[2];
            xdot[i * CR + 0] = o[1];
            xdot[i * CR + 1] = o[3];
            const double s0 = static_cast<double>(std::tanh(err));
            const double v = o[1], a = o[2], w = o[3];
            acc += static_cast<double>(static_cast<float>(10.0 * s0 * s0 + v * v + a * a + w * w));
            J[i] = static_cast<float>(acc);
            prev_err = err;
            prev_angle = o[2];
            n = i + 1;
            if (terminated) { crash = i + 1; break; }
        }
        Outcome out;
        out.crash_step = crash;
        out.success = crash == 0;
        finish_trajectory(T, C, CR, n, crash, J, x, xdot);
        return out;
    }
};

}  // namespace cartpole
}  // namespace esbo
