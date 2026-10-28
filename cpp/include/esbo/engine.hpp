// The priors and generators shared by every problem: degeneracy screen, single-trajectory prior,
// trajectory-set prior, pool layout and row packing (thesis Section "Priors", M:386-466).
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "problem.hpp"
#include "rng.hpp"
#include "sobol.hpp"
#include "util.hpp"

namespace esbo {

constexpr std::uint64_t PILOT_ROOT_SEED = 20261028;
constexpr long long PILOT_SAMPLES = 50000;
constexpr long long PILOT_BLOCK = 1000;
constexpr float F_INF = std::numeric_limits<float>::infinity();

// ------------------------------------------------------------------ grids (esbopfn.grids)
constexpr int REFERENCE_N_GRID = 67;   // ceil(1000 / 15)

inline int decoder_stride(int T) {
    int best = 1, best_d = std::numeric_limits<int>::max();
    for (int s = 1; s <= T; ++s) {
        const int n = (T + s - 1) / s;
        const int d = std::abs(n - REFERENCE_N_GRID);
        if (d < best_d) {
            best_d = d;
            best = s;
        }
    }
    return best;
}

// Decoder token steps l * stride (l = 1 .. N_grid - 1) followed by the horizon T.
inline std::vector<int> decoder_steps_with_horizon(int T) {
    const int s = decoder_stride(T);
    const int n = (T + s - 1) / s;
    std::vector<int> out;
    for (int l = 1; l < n; ++l) out.push_back(l * s);
    out.push_back(T);
    return out;
}

// t_l = floor(l T / N), l = 1 .. N.
inline std::vector<int> curve_steps(int T, int N) {
    std::vector<int> out;
    for (int l = 1; l <= N; ++l) out.push_back(static_cast<int>((static_cast<long long>(l) * T) / N));
    return out;
}

template <class P>
int n_max_trials() {
    return 3 * 15 * std::min(P::D, 4);
}

// ------------------------------------------------------------------ simulation workspace
template <class P>
struct Sim {
    std::vector<float> J, x, xdot;
    typename P::Cache cache;

    Sim() : J(P::T), x(std::size_t(P::T) * P::C), xdot(std::size_t(P::T) * std::max(P::CR, 1)) {}

    void prepare(const Instance& in) {
        cache = typename P::Cache{};
        P::prepare(in, cache);
    }
    Outcome run(const double* theta, const Instance& in, Rng& rng, const SimOpts& o = SimOpts{}) {
        return P::simulate(theta, in, cache, rng, o, J.data(), x.data(), xdot.data());
    }
};

// ------------------------------------------------------------------ degeneracy screen
struct ScreenCfg {
    int n_theta = 100;
    int repeats = 10;
    int min_pass = 25;
    long long max_candidates = 1000000;
};

struct ScreenResult {
    bool passed = false;
    bool vacuous = false;      // the problem has no unsuccessful trials, nothing to screen
    int passes = 0;
    int first_total = 0, first_crash = 0, first_unsuccessful = 0;
    long long rollouts = 0;

    // Crash ratio and unsuccessful-given-not-crashed ratio of the screen's first repeats.
    double rho_crash() const { return first_total > 0 ? double(first_crash) / first_total : 0.0; }
    double rho_unsuccessful_given_ok() const {
        const int ok = first_total - first_crash;
        return ok > 0 ? double(first_unsuccessful) / ok : 0.0;
    }
};

// Run n_theta controllers from a digitally shifted Sobol' design over Theta, each up to `repeats`
// times on the instance. A controller passes if all its repeats succeed. The instance is kept if
// at least `min_pass` controllers pass. The loop stops early once that is out of reach.
template <class P>
ScreenResult screen(Sim<P>& sim, const Instance& in, Rng& rng, const ScreenCfg& c) {
    ScreenResult r;
    if (!(P::CAN_CRASH || P::HAS_SUCCESS)) {
        r.passed = true;
        r.vacuous = true;
        r.passes = c.n_theta;
        return r;
    }
    std::uint32_t shift[SOBOL_MAX_DIM] = {0};
    for (int k = 0; k < P::D; ++k) shift[k] = static_cast<std::uint32_t>(rng.bits() >> 32);
    Sobol sob(P::D, shift);
    double u[SOBOL_MAX_DIM], th[SOBOL_MAX_DIM];
    for (int m = 0; m < c.n_theta; ++m) {
        sob.next(u);
        for (int k = 0; k < P::D; ++k) th[k] = P::THETA_LO[k] + u[k] * (P::THETA_HI[k] - P::THETA_LO[k]);
        bool ok = true;
        for (int rep = 0; rep < c.repeats; ++rep) {
            const Outcome o = sim.run(th, in, rng);
            ++r.rollouts;
            if (rep == 0) {
                ++r.first_total;
                const int cat = category(o);
                if (cat == CAT_CRASH) ++r.first_crash;
                if (cat == CAT_UNSUCCESSFUL) ++r.first_unsuccessful;
            }
            if (!o.success) {
                ok = false;
                break;
            }
        }
        if (ok) ++r.passes;
        if (r.passes + (c.n_theta - m - 1) < c.min_pass) break;
    }
    r.passed = r.passes >= c.min_pass;
    return r;
}

template <class P>
Instance draw_screened_instance(Sim<P>& sim, Rng& rng, const ScreenCfg& c, const PriorOpts& po,
                                ScreenResult& sr, long long& candidates) {
    for (candidates = 1;; ++candidates) {
        Instance in = sample_instance<P>(rng, po);
        sim.prepare(in);
        sr = screen(sim, in, rng, c);
        if (sr.passed) return in;
        if (candidates >= c.max_candidates)
            throw std::runtime_error(std::string(P::NAME) + ": no instance passed the screen in " +
                                     std::to_string(candidates) + " candidates");
    }
}

// ------------------------------------------------------------------ single-trajectory prior
struct PilotResult {
    long long n = 0, crash = 0, unsuccessful = 0;   // unsuccessful counts crashed trials too
    double rho_unsuccessful() const { return n > 0 ? double(unsuccessful) / n : 0.0; }
    double rho_crash() const { return n > 0 ? double(crash) / n : 0.0; }
};

// Rate of unsuccessful trajectories under the unscreened prior, from PILOT_SAMPLES draws in blocks
// seeded by derive_seed(root, block), so the estimate does not depend on the thread count.
template <class P>
PilotResult pilot(std::uint64_t root, long long n, int workers, const PriorOpts& po) {
    const long long blocks = (n + PILOT_BLOCK - 1) / PILOT_BLOCK;
    std::vector<PilotResult> parts(static_cast<std::size_t>(blocks));
    std::vector<long long> ids(static_cast<std::size_t>(blocks));
    std::iota(ids.begin(), ids.end(), 0LL);
    run_chunks(ids, workers, [&](long long b) {
        Rng rng(derive_seed(root, static_cast<std::uint64_t>(b)));
        Sim<P> sim;
        PilotResult& pr = parts[static_cast<std::size_t>(b)];
        const long long m = std::min(PILOT_BLOCK, n - b * PILOT_BLOCK);
        double theta[16];
        for (long long i = 0; i < m; ++i) {
            Instance in = sample_instance<P>(rng, po);
            sim.prepare(in);
            sample_theta<P>(rng, theta);
            const Outcome o = sim.run(theta, in, rng);
            ++pr.n;
            if (o.crash_step > 0) ++pr.crash;
            if (!o.success) ++pr.unsuccessful;
        }
    });
    PilotResult tot;
    for (const auto& p : parts) {
        tot.n += p.n;
        tot.crash += p.crash;
        tot.unsuccessful += p.unsuccessful;
    }
    return tot;
}

// Rejection probability max(0, 1 - (1 - rho) / rho) for unsuccessful trajectories, so that at
// most half of the accepted trajectories are unsuccessful.
inline double rejection_probability(double rho) {
    if (!(rho > 0.0)) return 0.0;
    return std::max(0.0, 1.0 - (1.0 - rho) / rho);
}

// ------------------------------------------------------------------ trajectory-set prior
struct SetCounts {
    int n = 0, n_crash = 0, n_unsuccessful = 0, n_success = 0;
    double rho_crash_prime = 0.0, rho_unsuccessful_prime = 0.0;
    int forced = 0;          // trajectories accepted outside their quota (rejection gave up)
    long long draws = 0;
};

// rho'_c ~ U(0, rho_c), rho'_s|c ~ U(0, rho_s|c), n_c ~ Bin(n, rho'_c),
// n_s&c ~ Bin(n - n_c, rho'_s|c), n_s = rest.
inline SetCounts draw_counts(int n, double rho_c, double rho_u, Rng& rng) {
    SetCounts sc;
    sc.n = n;
    sc.rho_crash_prime = rng.uniform(0.0, rho_c);
    sc.rho_unsuccessful_prime = rng.uniform(0.0, rho_u);
    sc.n_crash = rng.binomial(n, sc.rho_crash_prime);
    sc.n_unsuccessful = rng.binomial(n - sc.n_crash, sc.rho_unsuccessful_prime);
    sc.n_success = n - sc.n_crash - sc.n_unsuccessful;
    return sc;
}

// Draw theta ~ U(Theta) and simulate until the exact counts are reached. on_accept(theta,
// outcome, slot) sees the simulator buffers of each accepted trajectory. After 1000 + 1000 n
// draws, remaining slots take any trajectory (counted in `forced`, counts updated).
template <class P, class F>
void fill_set(Sim<P>& sim, const Instance& in, Rng& rng, SetCounts& sc, F&& on_accept) {
    int need[3] = {sc.n_success, sc.n_crash, sc.n_unsuccessful};
    const long long max_draws = 1000 + 1000LL * sc.n;
    double theta[16];
    int filled = 0;
    while (filled < sc.n) {
        sample_theta<P>(rng, theta);
        const Outcome o = sim.run(theta, in, rng);
        ++sc.draws;
        const int cat = category(o);
        if (need[cat] > 0) {
            --need[cat];
        } else if (sc.draws > max_draws) {
            const int give = need[0] > 0 ? 0 : (need[1] > 0 ? 1 : 2);
            --need[give];
            ++sc.forced;
            if (give == CAT_SUCCESS) --sc.n_success;
            if (give == CAT_CRASH) --sc.n_crash;
            if (give == CAT_UNSUCCESSFUL) --sc.n_unsuccessful;
            if (cat == CAT_SUCCESS) ++sc.n_success;
            if (cat == CAT_CRASH) ++sc.n_crash;
            if (cat == CAT_UNSUCCESSFUL) ++sc.n_unsuccessful;
        } else {
            continue;
        }
        on_accept(theta, o, filled);
        ++filled;
    }
}

// ------------------------------------------------------------------ pool layout
struct Column {
    std::string name;
    int offset = 0, length = 0;
    std::vector<int> shape;
    std::vector<std::string> names;
};

struct Layout {
    std::string kind;              // traj | set | labels
    std::string resolution;        // full | grid
    std::vector<Column> cols;
    std::vector<int> steps;        // stored 1-based steps
    std::string xdot_mode;         // per_step | interval_sum | none
    int R = 0, S = 0, CX = 0;

    void add(const std::string& name, std::vector<int> shape, std::vector<std::string> names = {}) {
        Column c;
        c.name = name;
        c.offset = R;
        c.length = 1;
        for (int s : shape) c.length *= s;
        c.shape = std::move(shape);
        c.names = std::move(names);
        R += c.length;
        cols.push_back(std::move(c));
    }
    int off(const std::string& name) const {
        for (const auto& c : cols)
            if (c.name == name) return c.offset;
        throw std::runtime_error("no column " + name);
    }
};

template <class P>
std::vector<std::string> names_of(const ParamSpec* t, int n) {
    std::vector<std::string> out;
    for (int i = 0; i < n; ++i) out.emplace_back(t[i].name);
    return out;
}
template <class P>
std::vector<std::string> theta_names() {
    std::vector<std::string> out;
    for (int i = 0; i < P::D; ++i) out.emplace_back(P::theta_names()[i]);
    return out;
}
template <class P>
std::vector<std::string> noise_names() {
    std::vector<std::string> out;
    for (int i = 0; i < P::N_NOISE; ++i) out.emplace_back(P::NOISE[i].name);
    return out;
}
template <class P>
std::vector<std::string> channel_names(bool rates) {
    std::vector<std::string> out;
    const int n = rates ? P::CR : P::C;
    const char* const* t = rates ? P::rate_channels() : P::channels();
    for (int i = 0; i < n; ++i) out.emplace_back(t[i]);
    return out;
}

template <class P>
Layout make_layout(const std::string& kind, const std::string& resolution) {
    Layout L;
    L.kind = kind;
    L.resolution = kind == "labels" ? "horizon" : resolution;
    if (kind == "labels") {
        L.steps = {P::T};
    } else if (resolution == "grid") {
        L.steps = decoder_steps_with_horizon(P::T);
    } else {
        for (int t = 1; t <= P::T; ++t) L.steps.push_back(t);
    }
    L.S = static_cast<int>(L.steps.size());
    if (kind != "traj") L.add("instance", {1});
    L.add("theta", {P::D}, theta_names<P>());
    if (kind == "traj") {
        L.add("phi", {P::N_PHI}, names_of<P>(P::PHI, P::N_PHI));
        if (P::USES_PHI_HAT) L.add("phi_hat", {P::N_PHI}, names_of<P>(P::PHI, P::N_PHI));
        L.add("noise_std", {P::N_NOISE}, noise_names<P>());
        if (P::N_EXTRA > 0) L.add("extras", {P::N_EXTRA}, names_of<P>(P::EXTRA, P::N_EXTRA));
    }
    L.add("crash_step", {1});
    L.add("success", {1});
    L.add("J", {L.S});
    if (kind == "labels") {
        L.xdot_mode = "none";
        return L;
    }
    L.add("x", {L.S, P::C}, channel_names<P>(false));
    if (L.resolution == "grid") {
        L.CX = P::CR > 0 ? P::CR : P::C;
        L.xdot_mode = "interval_sum";
        L.add("xdot", {L.S, L.CX}, P::CR > 0 ? channel_names<P>(true) : channel_names<P>(false));
    } else if (P::CR > 0) {
        L.CX = P::CR;
        L.xdot_mode = "per_step";
        L.add("xdot", {L.S, P::CR}, channel_names<P>(true));
    } else {
        L.xdot_mode = "none";
    }
    return L;
}

template <class P>
std::string layout_json(const Layout& L, const std::string& binary, const PriorOpts& po,
                        int rows_per_unit = 1) {
    std::string o = "{\n";
    o += "  \"format\": \"esbopfn-pool\",\n  \"format_version\": 1,\n";
    o += "  \"kind\": " + jstr(L.kind) + ",\n";
    o += "  \"resolution\": " + jstr(L.resolution) + ",\n";
    o += "  \"problem\": " + jstr(P::NAME) + ",\n";
    o += "  \"dtype\": \"float32\",\n";
    o += "  \"row_floats\": " + std::to_string(L.R) + ",\n";
    o += "  \"T\": " + std::to_string(P::T) + ",\n";
    o += "  \"d\": " + std::to_string(P::D) + ",\n";
    o += "  \"steps\": " + (L.resolution == "full" ? std::string("\"all\"") : jarr(L.steps.data(), L.S)) + ",\n";
    o += "  \"channels\": " + jstrs(channel_names<P>(false)) + ",\n";
    o += "  \"rate_channels\": " + jstrs(channel_names<P>(true)) + ",\n";
    o += "  \"xdot_mode\": " + jstr(L.xdot_mode) + ",\n";
    o += "  \"rows_per_unit\": " + std::to_string(rows_per_unit) + ",\n";
    o += "  \"crash_step_convention\": \"1-based step of the crash, 0 if none\",\n";
    o += "  \"generator\": " + jstr(binary) + ",\n";
    o += "  \"generator_version\": " + jstr(generator_version()) + ",\n";
    o += "  \"prior_options\": {\"dead_time\": " + std::string(po.dead_time ? "true" : "false") + "},\n";
    o += "  \"columns\": [\n";
    for (std::size_t i = 0; i < L.cols.size(); ++i) {
        const Column& c = L.cols[i];
        o += "    {\"name\": " + jstr(c.name) + ", \"offset\": " + std::to_string(c.offset) +
             ", \"length\": " + std::to_string(c.length) + ", \"shape\": " +
             jarr(c.shape.data(), static_cast<int>(c.shape.size()));
        if (!c.names.empty()) o += ", \"names\": " + jstrs(c.names);
        o += "}";
        o += (i + 1 < L.cols.size()) ? ",\n" : "\n";
    }
    o += "  ]\n}\n";
    return o;
}

// ------------------------------------------------------------------ row packing
template <class P>
void pack_instance_fields(const Layout& L, const Instance& in, float* row) {
    int o = L.off("phi");
    for (int i = 0; i < P::N_PHI; ++i) row[o + i] = static_cast<float>(in.phi[i]);
    if (P::USES_PHI_HAT) {
        o = L.off("phi_hat");
        for (int i = 0; i < P::N_PHI; ++i) row[o + i] = static_cast<float>(in.phi_hat[i]);
    }
    o = L.off("noise_std");
    for (int i = 0; i < P::N_NOISE; ++i) row[o + i] = static_cast<float>(in.noise[i]);
    if (P::N_EXTRA > 0) {
        o = L.off("extras");
        for (int i = 0; i < P::N_EXTRA; ++i) row[o + i] = static_cast<float>(in.extras[i]);
    }
}

// Writes theta, crash_step, success and the trajectory part of a row (any layout kind).
template <class P>
void pack_trajectory(const Layout& L, const Sim<P>& sim, const double* theta, const Outcome& o,
                     float* row) {
    int off = L.off("theta");
    for (int i = 0; i < P::D; ++i) row[off + i] = static_cast<float>(theta[i]);
    row[L.off("crash_step")] = static_cast<float>(o.crash_step);
    row[L.off("success")] = o.success ? 1.0f : 0.0f;
    float* J = row + L.off("J");
    for (int l = 0; l < L.S; ++l) J[l] = sim.J[L.steps[l] - 1];
    if (L.kind == "labels") return;
    float* x = row + L.off("x");
    for (int l = 0; l < L.S; ++l)
        for (int c = 0; c < P::C; ++c)
            x[l * P::C + c] = sim.x[std::size_t(L.steps[l] - 1) * P::C + c];
    if (L.xdot_mode == "per_step") {
        float* xd = row + L.off("xdot");
        std::memcpy(xd, sim.xdot.data(), sizeof(float) * std::size_t(P::T) * P::CR);
    } else if (L.xdot_mode == "interval_sum") {
        // Rates summed over (t_{l-1}, t_l] at step resolution: measured rates if the problem has
        // them, else level differences x_t - x_{t-1} with zero at step 1 (esbopfn.grids.tokens).
        float* xd = row + L.off("xdot");
        const int CX = L.CX;
        std::vector<double> acc(static_cast<std::size_t>(CX), 0.0);
        int l = 0;
        for (int t = 1; t <= P::T && l < L.S; ++t) {
            for (int c = 0; c < CX; ++c) {
                double r;
                if (P::CR > 0) {
                    r = sim.xdot[std::size_t(t - 1) * P::CR + c];
                } else {
                    r = t == 1 ? 0.0
                               : double(sim.x[std::size_t(t - 1) * P::C + c]) -
                                     double(sim.x[std::size_t(t - 2) * P::C + c]);
                }
                acc[c] += r;
            }
            if (t == L.steps[l]) {
                for (int c = 0; c < CX; ++c) {
                    xd[l * CX + c] = static_cast<float>(acc[c]);
                    acc[c] = 0.0;
                }
                ++l;
            }
        }
    }
}

// ------------------------------------------------------------------ instance JSON
template <class P>
std::string named_obj(const std::vector<double>& v, const std::vector<std::string>& names) {
    std::string o = "{";
    for (std::size_t i = 0; i < v.size() && i < names.size(); ++i) {
        if (i) o += ", ";
        o += jstr(names[i]) + ": " + jnum(v[i]);
    }
    return o + "}";
}

template <class P>
std::string instance_json(const Instance& in) {
    std::string o = "{\"phi\": " + named_obj<P>(in.phi, names_of<P>(P::PHI, P::N_PHI));
    o += ", \"phi_hat\": " + named_obj<P>(in.phi_hat, names_of<P>(P::PHI, P::N_PHI));
    o += ", \"noise_std\": " + named_obj<P>(in.noise, noise_names<P>());
    o += ", \"extras\": " + named_obj<P>(in.extras, names_of<P>(P::EXTRA, P::N_EXTRA));
    o += ", \"seed\": " + std::to_string(in.seed) + "}";
    return o;
}

inline std::vector<double> read_named(const JValue* v, const std::vector<std::string>& names,
                                      const std::string& what, bool required) {
    std::vector<double> out(names.size(), 0.0);
    if (v == nullptr || v->kind == JValue::Null) {
        if (required && !names.empty()) throw std::runtime_error("instance JSON lacks " + what);
        return {};
    }
    if (v->kind == JValue::Arr) {
        if (v->arr.size() != names.size())
            throw std::runtime_error(what + " must have " + std::to_string(names.size()) + " entries");
        for (std::size_t i = 0; i < names.size(); ++i) out[i] = v->arr[i].num;
        return out;
    }
    if (v->kind != JValue::Obj) throw std::runtime_error(what + " must be an object or a list");
    if (v->obj.empty() && !required) return {};
    for (std::size_t i = 0; i < names.size(); ++i) {
        const JValue* e = v->find(names[i]);
        if (e == nullptr) {
            std::string all;
            for (const auto& n : names) all += " " + n;
            throw std::runtime_error(what + " lacks entry '" + names[i] + "' (expected:" + all + ")");
        }
        out[i] = e->num;
    }
    return out;
}

template <class P>
Instance instance_from_json(const std::string& text) {
    const JValue v = JParser(text).parse();
    Instance in;
    in.phi = read_named(v.find("phi"), names_of<P>(P::PHI, P::N_PHI), "phi", true);
    in.phi_hat = read_named(v.find("phi_hat"), names_of<P>(P::PHI, P::N_PHI), "phi_hat", false);
    if (P::USES_PHI_HAT && in.phi_hat.empty()) in.phi_hat = in.phi;
    if (!P::USES_PHI_HAT) in.phi_hat.clear();
    in.noise = read_named(v.find("noise_std"), noise_names<P>(), "noise_std", true);
    in.extras = read_named(v.find("extras"), names_of<P>(P::EXTRA, P::N_EXTRA), "extras", false);
    if (in.extras.empty()) {
        in.extras.resize(P::N_EXTRA);
        for (int i = 0; i < P::N_EXTRA; ++i) in.extras[i] = P::extra_nominal(i);
    }
    if (const JValue* s = v.find("seed")) in.seed = static_cast<std::uint64_t>(s->num);
    return in;
}

}  // namespace esbo
