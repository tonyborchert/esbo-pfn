// Furuta pendulum (QUBE-Servo 2 matched): energy swing-up with an arm barrier, LQR hold within
// 20 degrees of upright designed from the estimate p_hat, actuation dead time and a brake-then-kick
// reset drawn per instance. Plant: the FurutaEnv equations integrated with 10 semi-implicit Euler
// substeps per 120 Hz control step. Crash: the continuous arm angle leaves +-90 degrees, the state
// becomes non-finite, or the LQR cannot be designed (crash at step 1). Success: no crash and mean
// |phi_pend| over the final two seconds below 5 degrees. Stage cost
// |phi_pend| (1 + 20 (1 - min(|phi_pend|, 5 deg) / 5 deg)) on the measured pendulum angle.
#pragma once

#include <algorithm>
#include <cmath>

#include "esbo/linalg.hpp"
#include "esbo/problem.hpp"

namespace esbo {
namespace furuta {

constexpr double G = 9.81;
constexpr double FREQ = 120.0;
constexpr int INTEGRATION_STEPS = 10;
constexpr double RESIDUAL_STD = 0.01;          // braked state jitter before the kick
constexpr double RESET_MAX_STATE = 1e6;        // a kick beyond this is treated as not taken
constexpr double KICK_SECONDS = 0.03;
constexpr double THETA_SAFE = M_PI / 2.0;
constexpr double ENV_MAX_VOLTAGE = 18.0;
constexpr double QUBE_MAX_VOLTAGE = 6.0;
constexpr double BALANCE_THRESHOLD = 20.0 * M_PI / 180.0;
constexpr double BARRIER_POWER = 2.0;          // fixed at control rates of 120 Hz and above
constexpr double PHI_UP = 5.0 * M_PI / 180.0;
constexpr double HOLD_WEIGHT = 20.0;
constexpr double SUCCESS_SECONDS = 2.0;

// p = [Rm, kt, km, mr, Lr, Dr, mp, Lp, Dp]: motor resistance, torque constant, back-EMF
// constant (tied to kt), arm mass, length and damping, pendulum mass, length and damping
inline constexpr ParamSpec PHI_TABLE[9] = {
    scale("Rm", 8.4, 4.0, 5.0),
    scale("kt", 0.042, 4.0, 5.0),
    tied("km", 1, 1.5),   // nominal: equal to the torque constant (SI units)
    scale("mr", 0.095, 4.0, 8.0),
    scale("Lr", 0.085, 3.0, 4.0),
    scale("Dr", 0.000275, 10.0, 3.0),
    scale("mp", 0.024, 4.0, 10.0),
    scale("Lp", 0.129, 3.0, 4.0),
    scale("Dp", 0.0000505, 10.0, 3.0),
};
// dead time in seconds (8.33 to 25 ms, one to three control periods), kick amplitude in volts
inline constexpr ParamSpec EXTRA_TABLE[2] = {
    bounded("dead_time", 1.0 / FREQ, 3.0 / FREQ),
    bounded("kick", 0.0, 0.6),
};
inline constexpr NoiseSpec NOISE_TABLE[3] = {{"angle", 0.002}, {"velocity", 0.07}, {"voltage", 0.1}};
inline constexpr double THETA_LO_T[8] = {40.0, 0.70, 11.5, 4.5, 0.5, 0.5, 0.5, 1.5};
inline constexpr double THETA_HI_T[8] = {90.0, 1.20, 12.5, 5.5, 1.5, 1.5, 1.5, 3.5};
inline const char* const THETA_NAMES_T[8] = {"mu", "E_bar", "q1", "q2", "q3", "q4", "r", "K_P_barrier"};
inline const char* const CHANNELS_T[2] = {"phi_arm", "phi_pend"};
inline const char* const RATE_CHANNELS_T[2] = {"phidot_arm", "phidot_pend"};

struct Params {
    double Rm, kt, km, mr, Lr, Dr, Jr, mp, Lp, Dp, Jp;
    explicit Params(const double* p)
        : Rm(p[0]), kt(p[1]), km(p[2]), mr(p[3]), Lr(p[4]), Dr(p[5]),
          Jr(p[3] * p[4] * p[4] / 12.0), mp(p[6]), Lp(p[7]), Dp(p[8]),
          Jp(p[6] * p[7] * p[7] / 12.0) {}
};

// ((a + pi) mod 2 pi) - pi with numpy modulo semantics.
inline double wrap(double a) {
    const double two_pi = 2.0 * M_PI;
    double r = std::fmod(a + M_PI, two_pi);
    if (r < 0.0) r += two_pi;
    return r - M_PI;
}

inline void accel(const Params& p, double alpha, double theta_dot, double alpha_dot, double tau,
                  double& theta_dd, double& alpha_dd) {
    const double sin_a = libm_sin(alpha), cos_a = libm_cos(alpha), sin_2a = libm_sin(2.0 * alpha);
    const double Lp2 = p.Lp * p.Lp, Lr2 = p.Lr * p.Lr;
    const double denom = 4.0 * Lp2 * Lr2 * p.mp * p.mp * cos_a * cos_a -
                         (4.0 * p.Jp + Lp2 * p.mp) *
                             (4.0 * p.Jr + Lp2 * p.mp * sin_a * sin_a + 4.0 * Lr2 * p.mp);
    const double term_p = -8.0 * p.Dp * alpha_dot + Lp2 * p.mp * theta_dot * theta_dot * sin_2a +
                          4.0 * p.Lp * G * p.mp * sin_a;
    const double term_r = 4.0 * p.Dr * theta_dot + Lp2 * alpha_dot * p.mp * theta_dot * sin_2a +
                          2.0 * p.Lp * p.Lr * alpha_dot * alpha_dot * p.mp * sin_a - 4.0 * tau;
    theta_dd = (-p.Lp * p.Lr * p.mp * term_p * cos_a + (4.0 * p.Jp + Lp2 * p.mp) * term_r) / denom;
    alpha_dd = (2.0 * p.Lp * p.Lr * p.mp * term_r * cos_a -
                0.5 * (4.0 * p.Jr + Lp2 * p.mp * sin_a * sin_a + 4.0 * Lr2 * p.mp) * term_p) /
               denom;
}

struct Env {
    Params p;
    double dt_sub;
    double s[4] = {0, 0, 0, 0};   // [theta_arm, alpha_pend (0 = upright), theta_dot, alpha_dot]

    explicit Env(const double* phi) : p(phi), dt_sub(1.0 / FREQ / INTEGRATION_STEPS) {}

    void step(double voltage) {
        const double Vm = std::min(ENV_MAX_VOLTAGE, std::max(-ENV_MAX_VOLTAGE, voltage));
        double th = s[0], al = s[1], thd = s[2], ald = s[3];
        for (int k = 0; k < INTEGRATION_STEPS; ++k) {
            const double tau = -(p.kt * (Vm - p.km * thd)) / p.Rm;
            double thdd, aldd;
            accel(p, al, thd, ald, tau, thdd, aldd);
            thd += thdd * dt_sub;
            ald += aldd * dt_sub;
            th = wrap(th + thd * dt_sub);
            al = wrap(al + ald * dt_sub);
        }
        s[0] = th; s[1] = al; s[2] = thd; s[3] = ald;
    }
};

// Continuous (A, B) about upright from a central-difference Jacobian of `accel`.
inline void linearize_upright(const double* phi, la::Mat& A, la::Mat& B) {
    const Params p(phi);
    auto fdot = [&](const double* x, double Vm, double* out) {
        const double tau = -(p.kt * (Vm - p.km * x[2])) / p.Rm;
        double tdd, add;
        accel(p, x[1], x[2], x[3], tau, tdd, add);
        out[0] = x[2]; out[1] = x[3]; out[2] = tdd; out[3] = add;
    };
    const double eps = 1e-6;
    A = la::Mat(4, 4);
    B = la::Mat(4, 1);
    double xp[4], xm[4], fp[4], fm[4];
    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) { xp[i] = 0.0; xm[i] = 0.0; }
        xp[j] = eps; xm[j] = -eps;
        fdot(xp, 0.0, fp);
        fdot(xm, 0.0, fm);
        for (int i = 0; i < 4; ++i) A(i, j) = (fp[i] - fm[i]) / (2.0 * eps);
    }
    const double x0[4] = {0, 0, 0, 0};
    fdot(x0, eps, fp);
    fdot(x0, -eps, fm);
    for (int i = 0; i < 4; ++i) B(i, 0) = -(fp[i] - fm[i]) / (2.0 * eps);
}

// Zero-order-hold discretization, expm([[A, B], [0, 0]] dt).
inline bool discretize(const la::Mat& A, const la::Mat& B, la::Mat& Ad, la::Mat& Bd) {
    const double dt = 1.0 / FREQ;
    la::Mat em(5, 5);
    for (int j = 0; j < 4; ++j)
        for (int i = 0; i < 4; ++i) em(i, j) = A(i, j) * dt;
    for (int i = 0; i < 4; ++i) em(i, 4) = B(i, 0) * dt;
    const la::Mat ms = la::expm(em);
    if (ms.r != 5 || !la::all_finite(ms)) return false;
    Ad = la::Mat(4, 4);
    Bd = la::Mat(4, 1);
    for (int j = 0; j < 4; ++j)
        for (int i = 0; i < 4; ++i) Ad(i, j) = ms(i, j);
    for (int i = 0; i < 4; ++i) Bd(i, 0) = ms(i, 4);
    return true;
}

// Discrete LQR gains [kp_theta, kp_alpha, kd_theta, kd_alpha], false if undesignable.
inline bool lqr_gains(const la::Mat& Ad, const la::Mat& Bd, const double* q, double r,
                      double* gains) {
    la::Mat Q(4, 4), R(1, 1), P;
    for (int i = 0; i < 4; ++i) Q(i, i) = q[i];
    R(0, 0) = r;
    if (!la::solve_discrete_are(Ad, Bd, Q, R, P)) return false;
    const la::Mat PB = la::mul(P, Bd);
    double denom = r;
    for (int i = 0; i < 4; ++i) denom += Bd(i, 0) * PB(i, 0);
    if (!(std::fabs(denom) > 0.0) || !std::isfinite(denom)) return false;
    const la::Mat PA = la::mul(P, Ad);
    for (int j = 0; j < 4; ++j) {
        double v = 0.0;
        for (int i = 0; i < 4; ++i) v += Bd(i, 0) * PA(i, j);
        gains[j] = v / denom;
        if (!std::isfinite(gains[j])) return false;
    }
    return true;
}

struct Controller {
    Params p;   // the estimate p_hat
    double mu, ref_energy, barrier_kp;
    double kp_theta = 0, kp_alpha = 0, kd_theta = 0, kd_alpha = 0;
    bool has_cont = false;
    double theta_cont = 0.0, prev_theta = 0.0;

    Controller(const double* phi_hat, const double* theta)
        : p(phi_hat), mu(theta[0]), ref_energy(theta[1] * phi_hat[6] * G * phi_hat[7]),
          barrier_kp(theta[7]) {}

    static double sgn(double v) { return v > 0.0 ? 1.0 : (v < 0.0 ? -1.0 : 0.0); }

    double flip_up(double alpha, double alpha_dot) const {
        const double pend_torque = 0.5 * p.mp * G * p.Lp * (1.0 + libm_cos(alpha));
        const double energy = pend_torque + (p.Jp / 2.0) * alpha_dot * alpha_dot;
        double u = mu * (energy - ref_energy) * sgn(-libm_cos(alpha) * alpha_dot);
        u = std::min(QUBE_MAX_VOLTAGE, std::max(-QUBE_MAX_VOLTAGE, u));
        return -((p.Rm / p.kt) * ((p.mr * p.Lr) * u));
    }

    double action(const double* obs) {
        const double th = obs[0], al = obs[1], thd = obs[2], ald = obs[3];
        if (!has_cont) {
            theta_cont = prev_theta = th;
            has_cont = true;
        } else {
            theta_cont += wrap(th - prev_theta);
            prev_theta = th;
        }
        double a;
        if (std::fabs(al) <= BALANCE_THRESHOLD)
            a = th * kp_theta + al * kp_alpha + thd * kd_theta + ald * kd_alpha;
        else
            a = flip_up(al, ald) +
                barrier_kp * sgn(theta_cont) * std::pow(std::fabs(theta_cont), BARRIER_POWER);
        return std::min(QUBE_MAX_VOLTAGE, std::max(-QUBE_MAX_VOLTAGE, a));
    }
};

// Fractional-sample transport delay on the voltage command, starting from zeros.
struct DeadTime {
    static constexpr int MAX_TAPS = 16;
    bool active = false;
    int n = 0, len = 0;
    double frac = 0.0;
    double buf[MAX_TAPS]{};

    explicit DeadTime(double samples) {
        if (!(samples > 0.0)) return;
        active = true;
        n = static_cast<int>(std::floor(samples));
        frac = samples - static_cast<double>(n);
        len = std::min(n + 2, MAX_TAPS);
    }
    double operator()(double v) {
        if (!active) return v;
        for (int i = 0; i < len - 1; ++i) buf[i] = buf[i + 1];
        buf[len - 1] = v;
        return (1.0 - frac) * buf[len - 1 - n] + frac * buf[len - 2 - n];
    }
};

inline float stage_cost(float alpha) {
    const float thr = static_cast<float>(PHI_UP);
    const float aabs = std::fabs(alpha);
    const float pr = std::min(aabs, thr) / thr;
    return aabs * (1.0f + static_cast<float>(HOLD_WEIGHT) * (1.0f - pr));
}

struct Problem {
    static constexpr const char* NAME = "furuta";
    static constexpr int D = 8, T = 600, C = 2, CR = 2;
    static constexpr int N_PHI = 9, N_EXTRA = 2, N_NOISE = 3;
    static constexpr bool USES_PHI_HAT = true, CAN_CRASH = true, HAS_SUCCESS = true;
    static constexpr const ParamSpec* PHI = PHI_TABLE;
    static constexpr const ParamSpec* EXTRA = EXTRA_TABLE;
    static constexpr const NoiseSpec* NOISE = NOISE_TABLE;
    static constexpr const double* THETA_LO = THETA_LO_T;
    static constexpr const double* THETA_HI = THETA_HI_T;
    static const char* const* theta_names() { return THETA_NAMES_T; }
    static const char* const* channels() { return CHANNELS_T; }
    static const char* const* rate_channels() { return RATE_CHANNELS_T; }

    static double phi_nominal(int i) { return PHI_TABLE[i].a; }
    static double extra_nominal(int i) { return i == 0 ? 2.0 / FREQ : 0.1; }
    static bool extra_active(int i, const PriorOpts& o) { return !(i == 0 && !o.dead_time); }

    struct Cache {
        bool designable = false;
        la::Mat Ad, Bd;
    };

    static void prepare(const Instance& in, Cache& c) {
        la::Mat A, B;
        linearize_upright(in.phi_hat.data(), A, B);
        c.designable = la::all_finite(A) && la::all_finite(B) && discretize(A, B, c.Ad, c.Bd);
    }

    template <class R>
    static Outcome simulate(const double* theta, const Instance& in, const Cache& cache, R& rng,
                            const SimOpts& opts, float* J, float* x, float* xdot) {
        const double s_angle = in.noise[0], s_vel = in.noise[1], s_volt = in.noise[2];
        const double dead_time = in.extras[0], kick = in.extras[1];

        Controller ctrl(in.phi_hat.data(), theta);
        bool have_ctrl = cache.designable;
        if (have_ctrl) {
            double g[4];
            have_ctrl = lqr_gains(cache.Ad, cache.Bd, theta + 2, theta[6], g);
            ctrl.kp_theta = g[0]; ctrl.kp_alpha = g[1]; ctrl.kd_theta = g[2]; ctrl.kd_alpha = g[3];
        }

        Env env(in.phi.data());
        DeadTime dl(dead_time * FREQ);

        // Brake-then-kick reset: braked state with a small residual, then a 30 ms pulse of
        // amplitude `kick` and random sign issued through the delay line.
        const double res = opts.deterministic ? 0.0 : RESIDUAL_STD;
        env.s[0] = wrap(rng.normal() * res);
        env.s[1] = wrap(M_PI + rng.normal() * res);
        env.s[2] = rng.normal() * res;
        env.s[3] = rng.normal() * res;
        if (opts.x0 != nullptr)
            for (int i = 0; i < 4; ++i) env.s[i] = opts.x0[i];
        const double braked[4] = {env.s[0], env.s[1], env.s[2], env.s[3]};
        const double u_sign = rng.uniform();
        const double sign = opts.deterministic ? 1.0 : (u_sign < 0.5 ? -1.0 : 1.0);
        const int n_kick = std::max(1, static_cast<int>(std::lround(KICK_SECONDS * FREQ)));
        for (int s = 0; s < n_kick; ++s) env.step(dl(sign * kick));
        bool sane = true;
        for (int i = 0; i < 4; ++i)
            if (!std::isfinite(env.s[i]) || std::fabs(env.s[i]) >= RESET_MAX_STATE) sane = false;
        if (!sane)
            for (int i = 0; i < 4; ++i) env.s[i] = braked[i];

        double theta_cont = env.s[0], prev_theta = env.s[0];
        int crash = 0, n = 0;
        double acc = 0.0;
        for (int i = 0; i < T; ++i) {
            const double obs[4] = {env.s[0] + rng.normal() * s_angle,
                                   env.s[1] + rng.normal() * s_angle,
                                   env.s[2] + rng.normal() * s_vel,
                                   env.s[3] + rng.normal() * s_vel};
            x[i * C + 0] = static_cast<float>(obs[0]);
            x[i * C + 1] = static_cast<float>(obs[1]);
            xdot[i * CR + 0] = static_cast<float>(obs[2]);
            xdot[i * CR + 1] = static_cast<float>(obs[3]);
            acc += static_cast<double>(stage_cost(x[i * C + 1]));
            J[i] = static_cast<float>(acc);
            n = i + 1;
            if (!have_ctrl) { crash = i + 1; break; }
            const double voltage = dl(ctrl.action(obs)) + rng.normal() * s_volt;
            env.step(voltage);
            if (!(std::isfinite(env.s[0]) && std::isfinite(env.s[1]) && std::isfinite(env.s[2]) &&
                  std::isfinite(env.s[3]))) {
                crash = i + 1;
                break;
            }
            theta_cont += wrap(env.s[0] - prev_theta);
            prev_theta = env.s[0];
            if (std::fabs(theta_cont) > THETA_SAFE) { crash = i + 1; break; }
        }
        Outcome out;
        out.crash_step = crash;
        if (crash == 0) {
            const int hold_n = static_cast<int>(std::lround(SUCCESS_SECONDS * FREQ));
            double s = 0.0;
            for (int i = T - hold_n; i < T; ++i) s += std::fabs(static_cast<double>(x[i * C + 1]));
            out.success = s / hold_n < PHI_UP;
        } else {
            out.success = false;
        }
        finish_trajectory(T, C, CR, n, crash, J, x, xdot);
        return out;
    }
};

}  // namespace furuta
}  // namespace esbo
