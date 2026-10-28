// The universal prior over system instances (thesis Section "Domain-Randomized Prior
// Instantiation" and Appendix tables tab:app:prior-phi / tab:app:prior-noise).
//
//   Scale(f_lo, f_hi):  p ~ LogUniform(p0 / f_lo, p0 * f_hi)
//   Bounded(lo, hi):    p ~ Uniform(lo, hi)
//   Tied(src, zeta):    p = p_src * LogUniform(1 / zeta, zeta), drawn after every other entry
//   sensor noise:       sigma_i = base_i * S * eta_i,  S ~ LogUniform(1.5/4, 4/1.5) once per
//                       instance, eta_i ~ LogUniform(1/1.5, 1.5) per noise channel
//   estimate:           p_hat_i = max(0, N(p_i, (p_i / 10)^2))  (only for problems whose
//                       controller is designed from an estimate)
// Entries marked "extra" (Furuta dead time and kick amplitude) are Bounded, drawn per instance and
// never part of the estimate.
#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "rng.hpp"

namespace esbo {

enum class Rule { Scale, Bounded, Tied };

struct ParamSpec {
    const char* name;
    Rule rule;
    double a;      // Scale: nominal p0 | Bounded: lo | Tied: unused
    double b;      // Scale: f_lo       | Bounded: hi | Tied: zeta
    double c;      // Scale: f_hi       | Bounded: unused
    int tie_src;   // Tied: index of the source entry, else -1
};

constexpr ParamSpec scale(const char* n, double nominal, double f_lo, double f_hi) {
    return ParamSpec{n, Rule::Scale, nominal, f_lo, f_hi, -1};
}
constexpr ParamSpec bounded(const char* n, double lo, double hi) {
    return ParamSpec{n, Rule::Bounded, lo, hi, 0.0, -1};
}
constexpr ParamSpec tied(const char* n, int src, double zeta) {
    return ParamSpec{n, Rule::Tied, 0.0, zeta, 0.0, src};
}

// Problem-level prior switches (only the Furuta dead time has one).
struct PriorOpts {
    bool dead_time = true;
};

struct NoiseSpec {
    const char* name;
    double base;
};

constexpr double NOISE_SCALE_LO = 1.5 / 4.0;
constexpr double NOISE_SCALE_HI = 4.0 / 1.5;
constexpr double NOISE_JITTER = 1.5;
constexpr double ESTIMATE_SIGMA_REL = 0.10;

// One sampled system instance. `phi_hat` is empty for problems without an estimate.
struct Instance {
    std::vector<double> phi, phi_hat, noise, extras;
    std::uint64_t seed = 0;
};

template <class P>
Instance sample_instance(Rng& rng, const PriorOpts& opts) {
    Instance in;
    in.phi.resize(P::N_PHI);
    for (int i = 0; i < P::N_PHI; ++i) {
        const ParamSpec& p = P::PHI[i];
        if (p.rule == Rule::Scale) in.phi[i] = rng.loguniform(p.a / p.b, p.a * p.c);
        else if (p.rule == Rule::Bounded) in.phi[i] = rng.uniform(p.a, p.b);
    }
    for (int i = 0; i < P::N_PHI; ++i) {
        const ParamSpec& p = P::PHI[i];
        if (p.rule == Rule::Tied) in.phi[i] = in.phi[p.tie_src] * rng.loguniform(1.0 / p.b, p.b);
    }
    in.extras.assign(P::N_EXTRA, 0.0);
    for (int i = 0; i < P::N_EXTRA; ++i) {
        const ParamSpec& p = P::EXTRA[i];
        const double v = rng.uniform(p.a, p.b);
        if (P::extra_active(i, opts)) in.extras[i] = v;
    }
    in.noise.resize(P::N_NOISE);
    const double S = rng.loguniform(NOISE_SCALE_LO, NOISE_SCALE_HI);
    for (int i = 0; i < P::N_NOISE; ++i)
        in.noise[i] = P::NOISE[i].base * S * rng.loguniform(1.0 / NOISE_JITTER, NOISE_JITTER);
    if (P::USES_PHI_HAT) {
        in.phi_hat.resize(P::N_PHI);
        for (int i = 0; i < P::N_PHI; ++i) {
            const double p = in.phi[i];
            in.phi_hat[i] = std::max(0.0, p + ESTIMATE_SIGMA_REL * p * rng.normal());
        }
    }
    return in;
}

// The default plant: nominal parameters, an exact estimate, base noise levels (or none).
template <class P>
Instance nominal_instance(bool noise, const PriorOpts& opts) {
    Instance in;
    in.phi.resize(P::N_PHI);
    for (int i = 0; i < P::N_PHI; ++i)
        if (P::PHI[i].rule != Rule::Tied) in.phi[i] = P::phi_nominal(i);
    for (int i = 0; i < P::N_PHI; ++i)
        if (P::PHI[i].rule == Rule::Tied) in.phi[i] = in.phi[P::PHI[i].tie_src];
    in.extras.resize(P::N_EXTRA);
    for (int i = 0; i < P::N_EXTRA; ++i)
        in.extras[i] = P::extra_active(i, opts) ? P::extra_nominal(i) : 0.0;
    in.noise.resize(P::N_NOISE);
    for (int i = 0; i < P::N_NOISE; ++i) in.noise[i] = noise ? P::NOISE[i].base : 0.0;
    if (P::USES_PHI_HAT) in.phi_hat = in.phi;
    return in;
}

template <class P>
void sample_theta(Rng& rng, double* theta) {
    for (int i = 0; i < P::D; ++i) theta[i] = rng.uniform(P::THETA_LO[i], P::THETA_HI[i]);
}

}  // namespace esbo
