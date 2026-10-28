// The subcommands shared by the three generator binaries.
//
//   traj      single-trajectory prior -> pool (rows.f32, spec.json, chunks.log)
//   sets      trajectory-set prior, N_max full trajectories per screened instance -> set pool
//             (+ labels/ with unrun query trials, instances.json)
//   contexts  FT-PFN examples, a fresh screened instance and trajectory set per example, stored
//             pre-tokenized on the 8-point curve grid in one shard per chunk
//   screen    screened instances only -> instances.json
//   pilot     rate of unsuccessful trajectories under the unscreened prior
//   rollout   one deterministic rollout as JSON (parity tests)
//   speed     generation-speed benchmark on one core
//   info, version, derive-seed, sobol
//
// Every chunked command takes --out DIR --count N --seed S --workers W --chunk K [--chunks LIST].
// Chunk k is generated from Rng(derive_seed(S, k)), so the output does not depend on W. Finished
// chunks are listed in DIR/chunks.log (contexts: one shard directory per chunk) and skipped when
// the command is run again.
#pragma once

#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "engine.hpp"

namespace esbo {

using Clock = std::chrono::steady_clock;

inline double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

struct Common {
    std::string out;
    long long count = 0, chunk = 0;
    std::uint64_t seed = 0;
    int workers = 1;
    std::vector<long long> chunks;
    PriorOpts po;
    ScreenCfg sc;
    bool quiet = false;
    long long n_chunks() const { return (count + chunk - 1) / chunk; }
    long long first(long long c) const { return c * chunk; }
    long long units(long long c) const { return std::min(chunk, count - c * chunk); }
};

inline Common parse_common(const Args& a, long long default_chunk) {
    Common c;
    c.out = a.require("out");
    c.count = a.geti("count", 0);
    if (c.count <= 0) throw std::runtime_error("--count must be positive");
    c.chunk = a.geti("chunk", default_chunk);
    if (c.chunk <= 0) throw std::runtime_error("--chunk must be positive");
    c.seed = a.getu("seed", 0);
    c.workers = static_cast<int>(a.geti("workers", 1));
    c.quiet = a.has("quiet");
    c.po.dead_time = !a.has("no_dead_time");
    c.sc.n_theta = static_cast<int>(a.geti("screen_thetas", 100));
    c.sc.repeats = static_cast<int>(a.geti("screen_repeats", 10));
    c.sc.min_pass = static_cast<int>(a.geti("screen_min_pass", 25));
    c.sc.max_candidates = a.geti("screen_max_candidates", 1000000);
    if (a.has("chunks")) {
        for (long long k : parse_index_list(a.get("chunks")))
            if (k >= 0 && k < c.n_chunks()) c.chunks.push_back(k);
    } else {
        for (long long k = 0; k < c.n_chunks(); ++k) c.chunks.push_back(k);
    }
    return c;
}

inline std::string screen_json(const ScreenCfg& s) {
    return "{\"n_theta\": " + std::to_string(s.n_theta) + ", \"repeats\": " +
           std::to_string(s.repeats) + ", \"min_pass\": " + std::to_string(s.min_pass) +
           ", \"design\": \"sobol\"}";
}

// Fails if DIR already holds a dataset with a different identity, writes it otherwise.
inline void claim_directory(const std::string& dir, const std::string& identity) {
    mkdirs(dir);
    const std::string path = dir + "/identity.json";
    if (file_exists(path)) {
        const std::string have = read_file(path);
        if (have != identity)
            throw std::runtime_error(dir + " holds a different dataset (identity.json differs):\n" +
                                     have + "\nrequested:\n" + identity);
        return;
    }
    write_atomic(path, identity);
}

inline void write_spec(const std::string& dir, const std::string& spec) {
    const std::string path = dir + "/spec.json";
    if (file_exists(path) && read_file(path) != spec)
        throw std::runtime_error(path + " has a different column layout");
    if (!file_exists(path)) write_atomic(path, spec);
}

inline std::string chunk_name(long long c) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "chunk_%06lld", c);
    return buf;
}

// Chunks of `log` whose recorded unit count matches the current plan.
inline std::set<long long> done_chunks(const std::string& log_path, const Common& c) {
    std::set<long long> out;
    std::ifstream f(log_path);
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        long long k = 0, n = 0;
        if (ss >> k >> n)
            if (k >= 0 && k < c.n_chunks() && n == c.units(k)) out.insert(k);
    }
    return out;
}

// Merge instances.d/chunk_*.jsonl into instances.json once every chunk is present.
inline void merge_instances(const std::string& dir, const Common& c) {
    std::string o = "[\n";
    bool firstline = true;
    for (long long k = 0; k < c.n_chunks(); ++k) {
        const std::string p = dir + "/instances.d/" + chunk_name(k) + ".jsonl";
        std::ifstream f(p);
        if (!f) throw std::runtime_error("missing " + p);
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            if (!firstline) o += ",\n";
            o += line;
            firstline = false;
        }
    }
    o += "\n]\n";
    write_atomic(dir + "/instances.json", o);
}

inline void log_chunk(const Common& c, const char* what, long long k, long long n, double secs) {
    if (c.quiet) return;
    std::fprintf(stderr, "%s: chunk %lld/%lld done (%lld units, %.2f s)\n", what, k + 1,
                 c.n_chunks(), n, secs);
}

template <class P>
std::string instance_line(long long index, const Instance& in, const ScreenResult& sr,
                          long long candidates, const SetCounts* set) {
    std::string o = "{\"index\": " + std::to_string(index) + ", \"instance\": " + instance_json<P>(in);
    o += ", \"screen\": {\"passes\": " + std::to_string(sr.passes) +
         ", \"candidates\": " + std::to_string(candidates) +
         ", \"rollouts\": " + std::to_string(sr.rollouts) +
         ", \"vacuous\": " + (sr.vacuous ? "true" : "false") +
         ", \"rho_crash\": " + jnum(sr.rho_crash()) +
         ", \"rho_unsuccessful_given_ok\": " + jnum(sr.rho_unsuccessful_given_ok()) + "}";
    if (set != nullptr) {
        o += ", \"set\": {\"n\": " + std::to_string(set->n) +
             ", \"n_crash\": " + std::to_string(set->n_crash) +
             ", \"n_unsuccessful\": " + std::to_string(set->n_unsuccessful) +
             ", \"n_success\": " + std::to_string(set->n_success) +
             ", \"rho_crash_prime\": " + jnum(set->rho_crash_prime) +
             ", \"rho_unsuccessful_prime\": " + jnum(set->rho_unsuccessful_prime) +
             ", \"forced\": " + std::to_string(set->forced) +
             ", \"draws\": " + std::to_string(set->draws) + "}";
    }
    return o + "}";
}

// ------------------------------------------------------------------ pilot
template <class P>
int cmd_pilot(const Args& a) {
    PriorOpts po;
    po.dead_time = !a.has("no_dead_time");
    const long long n = a.geti("n", PILOT_SAMPLES);
    const std::uint64_t root = a.getu("pilot_seed", PILOT_ROOT_SEED);
    const auto t0 = Clock::now();
    const PilotResult pr = pilot<P>(root, n, static_cast<int>(a.geti("workers", 1)), po);
    std::printf("{\"problem\": %s, \"n\": %lld, \"pilot_seed\": %llu, \"rho_unsuccessful\": %s, "
                "\"rho_crash\": %s, \"seconds\": %s}\n",
                jstr(P::NAME).c_str(), pr.n, static_cast<unsigned long long>(root),
                jnum(pr.rho_unsuccessful()).c_str(), jnum(pr.rho_crash()).c_str(),
                jnum(seconds_since(t0)).c_str());
    return 0;
}

template <class P>
double rho_from_args(const Args& a, const Common& c) {
    if (a.has("no_reject")) return 0.0;
    if (a.has("rho")) return a.getd("rho", 0.0);
    const PilotResult pr = pilot<P>(PILOT_ROOT_SEED, PILOT_SAMPLES, c.workers, c.po);
    if (!c.quiet)
        std::fprintf(stderr, "pilot: rho_unsuccessful = %.6f over %lld draws\n",
                     pr.rho_unsuccessful(), pr.n);
    return pr.rho_unsuccessful();
}

// ------------------------------------------------------------------ traj
template <class P>
int cmd_traj(const Args& a, const std::string& binary) {
    const Common c = parse_common(a, 10000);
    const double rho = rho_from_args<P>(a, c);
    const double rp = rejection_probability(rho);
    const Layout L = make_layout<P>("traj", "full");
    const std::string identity =
        "{\"kind\": \"traj\", \"problem\": " + jstr(P::NAME) + ", \"seed\": " +
        std::to_string(c.seed) + ", \"chunk\": " + std::to_string(c.chunk) +
        ", \"rho_unsuccessful\": " + jnum(rho) + ", \"dead_time\": " +
        (c.po.dead_time ? "true" : "false") + "}\n";
    claim_directory(c.out, identity);
    write_spec(c.out, layout_json<P>(L, binary, c.po));
    const std::string logp = c.out + "/chunks.log";
    const std::set<long long> done = done_chunks(logp, c);
    ChunkLog log(logp);
    BinFile rows(c.out + "/rows.f32");
    std::vector<long long> todo;
    for (long long k : c.chunks)
        if (!done.count(k)) todo.push_back(k);
    const long long BLOCK = 256;
    run_chunks(todo, c.workers, [&](long long k) {
        const auto t0 = Clock::now();
        Rng rng(derive_seed(c.seed, static_cast<std::uint64_t>(k)));
        Sim<P> sim;
        const long long n = c.units(k), first = c.first(k);
        std::vector<float> buf(static_cast<std::size_t>(BLOCK * L.R));
        double theta[16];
        long long attempts = 0, in_block = 0, block_start = 0;
        for (long long r = 0; r < n; ++r) {
            Instance in;
            Outcome o;
            for (;;) {
                in = sample_instance<P>(rng, c.po);
                sim.prepare(in);
                sample_theta<P>(rng, theta);
                o = sim.run(theta, in, rng);
                ++attempts;
                if (category(o) != CAT_SUCCESS && rp > 0.0 && rng.uniform() < rp) continue;
                break;
            }
            float* row = buf.data() + in_block * L.R;
            std::fill(row, row + L.R, 0.0f);
            pack_instance_fields<P>(L, in, row);
            pack_trajectory<P>(L, sim, theta, o, row);
            ++in_block;
            if (in_block == BLOCK || r + 1 == n) {
                rows.pwrite_all(buf.data(), sizeof(float) * std::size_t(in_block * L.R),
                                sizeof(float) * std::uint64_t(first + block_start) * L.R);
                block_start += in_block;
                in_block = 0;
            }
        }
        rows.sync();
        log.mark(k, n);
        log_chunk(c, "traj", k, n, seconds_since(t0));
        (void)attempts;
    });
    if (done_chunks(logp, c).size() == static_cast<std::size_t>(c.n_chunks()))
        write_atomic(c.out + "/pool.finished", "{\"rows\": " + std::to_string(c.count) + "}\n");
    return 0;
}

// ------------------------------------------------------------------ sets
template <class P>
int cmd_sets(const Args& a, const std::string& binary) {
    const Common c = parse_common(a, 100);
    const std::string resolution = a.get("resolution", "grid");
    if (resolution != "grid" && resolution != "full")
        throw std::runtime_error("--resolution must be grid or full");
    const int n_labels = static_cast<int>(a.geti("labels", 128));
    const int nmax = static_cast<int>(a.geti("n_max", n_max_trials<P>()));
    const Layout L = make_layout<P>("set", resolution);
    const Layout LL = make_layout<P>("labels", resolution);
    const std::string identity =
        "{\"kind\": \"sets\", \"problem\": " + jstr(P::NAME) + ", \"seed\": " +
        std::to_string(c.seed) + ", \"chunk\": " + std::to_string(c.chunk) +
        ", \"resolution\": " + jstr(resolution) + ", \"n_max\": " + std::to_string(nmax) +
        ", \"labels\": " + std::to_string(n_labels) + ", \"screen\": " + screen_json(c.sc) +
        ", \"dead_time\": " + (c.po.dead_time ? "true" : "false") + "}\n";
    claim_directory(c.out, identity);
    mkdirs(c.out + "/labels");
    mkdirs(c.out + "/instances.d");
    write_spec(c.out, layout_json<P>(L, binary, c.po, nmax));
    write_spec(c.out + "/labels", layout_json<P>(LL, binary, c.po, n_labels));
    const std::string logp = c.out + "/chunks.log";
    const std::set<long long> done = done_chunks(logp, c);
    ChunkLog log(logp);
    BinFile rows(c.out + "/rows.f32");
    BinFile lrows(c.out + "/labels/rows.f32");
    std::vector<long long> todo;
    for (long long k : c.chunks)
        if (!done.count(k)) todo.push_back(k);
    run_chunks(todo, c.workers, [&](long long k) {
        const auto t0 = Clock::now();
        const std::uint64_t cs = derive_seed(c.seed, static_cast<std::uint64_t>(k));
        const long long n = c.units(k), first = c.first(k);
        std::vector<float> pool(std::size_t(nmax) * L.R), lab(std::size_t(n_labels) * LL.R);
        std::string lines;
        Sim<P> sim;
        double theta[16];
        for (long long j = 0; j < n; ++j) {
            const long long i = first + j;
            const std::uint64_t iseed = sub_seed(cs, static_cast<std::uint64_t>(j));
            Rng rng(iseed);
            ScreenResult sr;
            long long cand = 0;
            Instance in = draw_screened_instance(sim, rng, c.sc, c.po, sr, cand);
            in.seed = iseed;
            SetCounts sc = draw_counts(nmax, sr.rho_crash(), sr.rho_unsuccessful_given_ok(), rng);
            std::vector<int> perm(static_cast<std::size_t>(nmax));
            std::iota(perm.begin(), perm.end(), 0);
            std::shuffle(perm.begin(), perm.end(), rng.gen);
            std::fill(pool.begin(), pool.end(), 0.0f);
            fill_set(sim, in, rng, sc, [&](const double* th, const Outcome& o, int slot) {
                float* row = pool.data() + std::size_t(perm[static_cast<std::size_t>(slot)]) * L.R;
                row[L.off("instance")] = static_cast<float>(i);
                pack_trajectory<P>(L, sim, th, o, row);
            });
            rows.pwrite_all(pool.data(), sizeof(float) * pool.size(),
                            sizeof(float) * std::uint64_t(i) * nmax * L.R);
            std::fill(lab.begin(), lab.end(), 0.0f);
            for (int l = 0; l < n_labels; ++l) {
                sample_theta<P>(rng, theta);
                const Outcome o = sim.run(theta, in, rng);
                float* row = lab.data() + std::size_t(l) * LL.R;
                row[LL.off("instance")] = static_cast<float>(i);
                pack_trajectory<P>(LL, sim, theta, o, row);
            }
            if (n_labels > 0)
                lrows.pwrite_all(lab.data(), sizeof(float) * lab.size(),
                                 sizeof(float) * std::uint64_t(i) * n_labels * LL.R);
            lines += instance_line<P>(i, in, sr, cand, &sc) + "\n";
        }
        rows.sync();
        lrows.sync();
        write_atomic(c.out + "/instances.d/" + chunk_name(k) + ".jsonl", lines);
        log.mark(k, n);
        log_chunk(c, "sets", k, n, seconds_since(t0));
    });
    if (done_chunks(logp, c).size() == static_cast<std::size_t>(c.n_chunks())) {
        merge_instances(c.out, c);
        write_atomic(c.out + "/pool.finished",
                     "{\"instances\": " + std::to_string(c.count) + ", \"rows\": " +
                         std::to_string(c.count * nmax) + "}\n");
    }
    return 0;
}

// ------------------------------------------------------------------ screen
template <class P>
int cmd_screen(const Args& a) {
    const Common c = parse_common(a, 100);
    const std::string identity =
        "{\"kind\": \"screen\", \"problem\": " + jstr(P::NAME) + ", \"seed\": " +
        std::to_string(c.seed) + ", \"chunk\": " + std::to_string(c.chunk) +
        ", \"screen\": " + screen_json(c.sc) + ", \"dead_time\": " +
        (c.po.dead_time ? "true" : "false") + "}\n";
    claim_directory(c.out, identity);
    mkdirs(c.out + "/instances.d");
    const std::string logp = c.out + "/chunks.log";
    const std::set<long long> done = done_chunks(logp, c);
    ChunkLog log(logp);
    std::vector<long long> todo;
    for (long long k : c.chunks)
        if (!done.count(k)) todo.push_back(k);
    run_chunks(todo, c.workers, [&](long long k) {
        const auto t0 = Clock::now();
        const std::uint64_t cs = derive_seed(c.seed, static_cast<std::uint64_t>(k));
        const long long n = c.units(k), first = c.first(k);
        std::string lines;
        Sim<P> sim;
        for (long long j = 0; j < n; ++j) {
            const std::uint64_t iseed = sub_seed(cs, static_cast<std::uint64_t>(j));
            Rng rng(iseed);
            ScreenResult sr;
            long long cand = 0;
            Instance in = draw_screened_instance(sim, rng, c.sc, c.po, sr, cand);
            in.seed = iseed;
            lines += instance_line<P>(first + j, in, sr, cand, nullptr) + "\n";
        }
        write_atomic(c.out + "/instances.d/" + chunk_name(k) + ".jsonl", lines);
        log.mark(k, n);
        log_chunk(c, "screen", k, n, seconds_since(t0));
    });
    if (done_chunks(logp, c).size() == static_cast<std::size_t>(c.n_chunks())) {
        merge_instances(c.out, c);
        write_atomic(c.out + "/pool.finished", "{\"instances\": " + std::to_string(c.count) + "}\n");
    }
    return 0;
}

// ------------------------------------------------------------------ contexts (FT-PFN)
constexpr int FTPFN_GRID = 8;
constexpr int CTX_TOKENS = FTPFN_GRID + 1;
enum TrialStatus : std::int8_t { ST_COMPLETE = 0, ST_TRUNCATED = 1, ST_CRASHED = 2 };

struct ContextChunk {
    std::vector<std::int64_t> ex_ptr{0};
    std::vector<std::int32_t> ex_n;
    std::vector<float> ex_meta;      // p_cut, rho'_c, rho'_s|c per example
    std::vector<std::int64_t> ex_instance_seed;
    std::vector<float> tr_theta, tr_target, tok_J, un_theta, un_target;
    std::vector<std::int32_t> tr_len;
    std::vector<std::int8_t> tr_status;
    std::vector<std::int16_t> tok_t;
};

struct CtxTrial {
    std::vector<float> theta;
    std::int32_t len = 0;
    std::int8_t status = 0;
    float target = 0.0f;
    std::int16_t tok_t[CTX_TOKENS] = {0};
    float tok_J[CTX_TOKENS] = {0};
};

// One FT-PFN training example: a fresh screened instance, n ~ U{1..N_max} trajectories drawn with
// the trajectory-set prior, one cut probability per example, a cutoff ~ U{1..T_i} for each cut
// trial, the tokens ((theta_i, t_l), J_{t_l}) for t_l <= T_i on the 8-point grid plus the last
// observed step if T_i < T, and n_unrun unrun controllers with their total cost.
template <class P>
void context_example(Sim<P>& sim, Rng& rng, const Common& c, int nmax, int n_unrun,
                     const std::vector<int>& grid, ContextChunk& out, std::uint64_t iseed) {
    ScreenResult sr;
    long long cand = 0;
    Instance in = draw_screened_instance(sim, rng, c.sc, c.po, sr, cand);
    const int n = static_cast<int>(rng.integer(1, nmax));
    SetCounts sc = draw_counts(n, sr.rho_crash(), sr.rho_unsuccessful_given_ok(), rng);
    const double p_cut = rng.uniform();
    std::vector<CtxTrial> trials(static_cast<std::size_t>(n));
    fill_set(sim, in, rng, sc, [&](const double* th, const Outcome& o, int slot) {
        CtxTrial& t = trials[static_cast<std::size_t>(slot)];
        t.theta.assign(th, th + P::D);
        const int Tlen = o.crash_step > 0 ? o.crash_step : P::T;
        const bool cut = rng.uniform() < p_cut;
        const int Lobs = cut ? static_cast<int>(rng.integer(1, Tlen)) : Tlen;
        t.len = Lobs;
        if (o.crash_step > 0 && Lobs == o.crash_step) t.status = ST_CRASHED;
        else if (Lobs == P::T) t.status = ST_COMPLETE;
        else t.status = ST_TRUNCATED;
        t.target = o.crash_step > 0 ? F_INF : sim.J[P::T - 1];
        int m = 0;
        for (int g : grid)
            if (g <= Lobs) {
                t.tok_t[m] = static_cast<std::int16_t>(g);
                t.tok_J[m] = sim.J[g - 1];
                ++m;
            }
        if (Lobs < P::T) {
            t.tok_t[m] = static_cast<std::int16_t>(Lobs);
            t.tok_J[m] = sim.J[Lobs - 1];
        }
    });
    std::shuffle(trials.begin(), trials.end(), rng.gen);
    for (const CtxTrial& t : trials) {
        out.tr_theta.insert(out.tr_theta.end(), t.theta.begin(), t.theta.end());
        out.tr_len.push_back(t.len);
        out.tr_status.push_back(t.status);
        out.tr_target.push_back(t.target);
        out.tok_t.insert(out.tok_t.end(), t.tok_t, t.tok_t + CTX_TOKENS);
        out.tok_J.insert(out.tok_J.end(), t.tok_J, t.tok_J + CTX_TOKENS);
    }
    double theta[16];
    for (int u = 0; u < n_unrun; ++u) {
        sample_theta<P>(rng, theta);
        const Outcome o = sim.run(theta, in, rng);
        for (int k = 0; k < P::D; ++k) out.un_theta.push_back(static_cast<float>(theta[k]));
        out.un_target.push_back(o.crash_step > 0 ? F_INF : sim.J[P::T - 1]);
    }
    out.ex_n.push_back(n);
    out.ex_meta.push_back(static_cast<float>(p_cut));
    out.ex_meta.push_back(static_cast<float>(sc.rho_crash_prime));
    out.ex_meta.push_back(static_cast<float>(sc.rho_unsuccessful_prime));
    out.ex_instance_seed.push_back(static_cast<std::int64_t>(iseed));
    out.ex_ptr.push_back(out.ex_ptr.back() + n);
}

template <class T>
void write_array(const std::string& path, const std::vector<T>& v) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("cannot write " + path);
    f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(sizeof(T) * v.size()));
}

template <class P>
int cmd_contexts(const Args& a) {
    const Common c = parse_common(a, 1000);
    const int n_unrun = static_cast<int>(a.geti("n_unrun", 16));
    const int nmax = static_cast<int>(a.geti("n_max", n_max_trials<P>()));
    const std::vector<int> grid = curve_steps(P::T, FTPFN_GRID);
    const std::string identity =
        "{\"kind\": \"contexts\", \"problem\": " + jstr(P::NAME) + ", \"seed\": " +
        std::to_string(c.seed) + ", \"chunk\": " + std::to_string(c.chunk) +
        ", \"n_max\": " + std::to_string(nmax) + ", \"n_unrun\": " + std::to_string(n_unrun) +
        ", \"grid\": " + jarr(grid.data(), FTPFN_GRID) + ", \"screen\": " + screen_json(c.sc) +
        ", \"dead_time\": " + (c.po.dead_time ? "true" : "false") + "}\n";
    claim_directory(c.out, identity);
    mkdirs(c.out + "/shards");
    auto shard_done = [&](long long k) {
        return file_exists(c.out + "/shards/" + chunk_name(k) + "/shard.json");
    };
    std::vector<long long> todo;
    for (long long k : c.chunks)
        if (!shard_done(k)) todo.push_back(k);
    run_chunks(todo, c.workers, [&](long long k) {
        const auto t0 = Clock::now();
        const std::uint64_t cs = derive_seed(c.seed, static_cast<std::uint64_t>(k));
        const long long n = c.units(k), first = c.first(k);
        ContextChunk ch;
        Sim<P> sim;
        for (long long j = 0; j < n; ++j) {
            const std::uint64_t iseed = sub_seed(cs, static_cast<std::uint64_t>(j));
            Rng rng(iseed);
            context_example<P>(sim, rng, c, nmax, n_unrun, grid, ch, iseed);
        }
        const std::string tmp = c.out + "/shards/.tmp_" + chunk_name(k) + "_" + std::to_string(::getpid());
        mkdirs(tmp);
        write_array(tmp + "/example_ptr.i64", ch.ex_ptr);
        write_array(tmp + "/example_n.i32", ch.ex_n);
        write_array(tmp + "/example_meta.f32", ch.ex_meta);
        write_array(tmp + "/example_seed.i64", ch.ex_instance_seed);
        write_array(tmp + "/trial_theta.f32", ch.tr_theta);
        write_array(tmp + "/trial_len.i32", ch.tr_len);
        write_array(tmp + "/trial_status.i8", ch.tr_status);
        write_array(tmp + "/trial_target.f32", ch.tr_target);
        write_array(tmp + "/trial_tok_t.i16", ch.tok_t);
        write_array(tmp + "/trial_tok_J.f32", ch.tok_J);
        write_array(tmp + "/unrun_theta.f32", ch.un_theta);
        write_array(tmp + "/unrun_target.f32", ch.un_target);
        const long long n_trials = ch.ex_ptr.back();
        std::string meta = "{\"format\": \"esbopfn-contexts\", \"format_version\": 1, \"problem\": " +
                           jstr(P::NAME) + ", \"chunk\": " + std::to_string(k) +
                           ", \"first_example\": " + std::to_string(first) +
                           ", \"examples\": " + std::to_string(n) +
                           ", \"trials\": " + std::to_string(n_trials) +
                           ", \"d\": " + std::to_string(P::D) + ", \"T\": " + std::to_string(P::T) +
                           ", \"n_max\": " + std::to_string(nmax) +
                           ", \"n_tokens\": " + std::to_string(CTX_TOKENS) +
                           ", \"n_unrun\": " + std::to_string(n_unrun) +
                           ", \"grid\": " + jarr(grid.data(), FTPFN_GRID) +
                           ", \"status_codes\": {\"complete\": 0, \"truncated\": 1, \"crashed\": 2}" +
                           ", \"generator_version\": " + jstr(generator_version()) + "}\n";
        write_atomic(tmp + "/shard.json", meta);
        const std::string dst = c.out + "/shards/" + chunk_name(k);
        if (std::rename(tmp.c_str(), dst.c_str()) != 0)
            throw std::runtime_error("cannot move shard into place: " + dst);
        log_chunk(c, "contexts", k, n, seconds_since(t0));
    });
    bool all = true;
    for (long long k = 0; k < c.n_chunks(); ++k) all = all && shard_done(k);
    if (all)
        write_atomic(c.out + "/pool.finished", "{\"examples\": " + std::to_string(c.count) + "}\n");
    return 0;
}

// ------------------------------------------------------------------ screen one instance
// Runs the degeneracy screen on one given instance (--instance-json or --nominal) and prints the
// result as JSON.
template <class P>
int cmd_screen_one(const Args& a) {
    PriorOpts po;
    po.dead_time = !a.has("no_dead_time");
    ScreenCfg sc;
    sc.n_theta = static_cast<int>(a.geti("screen_thetas", 100));
    sc.repeats = static_cast<int>(a.geti("screen_repeats", 10));
    sc.min_pass = static_cast<int>(a.geti("screen_min_pass", 25));
    Instance in;
    if (a.has("instance_json")) {
        const std::string v = a.get("instance_json");
        in = instance_from_json<P>(!v.empty() && v[0] == '{' ? v : read_file(v));
    } else {
        in = nominal_instance<P>(!a.has("no_noise"), po);
    }
    Sim<P> sim;
    sim.prepare(in);
    Rng rng(derive_seed(a.getu("seed", 0), 3));
    // full design: no early stop, so `passes` counts every passing controller
    ScreenCfg full = sc;
    full.min_pass = a.has("full") ? 0 : sc.min_pass;
    ScreenResult r = screen(sim, in, rng, full);
    r.passed = r.passes >= sc.min_pass;
    std::printf("{\"problem\": %s, \"passed\": %s, \"passes\": %d, \"n_theta\": %d, "
                "\"min_pass\": %d, \"rollouts\": %lld, \"rho_crash\": %s, "
                "\"rho_unsuccessful_given_ok\": %s}\n",
                jstr(P::NAME).c_str(), r.passed ? "true" : "false", r.passes, sc.n_theta,
                sc.min_pass, r.rollouts, jnum(r.rho_crash()).c_str(),
                jnum(r.rho_unsuccessful_given_ok()).c_str());
    return 0;
}

// ------------------------------------------------------------------ rollout
template <class P>
int cmd_rollout(const Args& a) {
    PriorOpts po;
    po.dead_time = !a.has("no_dead_time");
    const std::uint64_t seed = a.getu("seed", 0);
    Instance in;
    if (a.has("instance_json")) {
        const std::string v = a.get("instance_json");
        in = instance_from_json<P>(!v.empty() && v[0] == '{' ? v : read_file(v));
    } else if (a.has("nominal")) {
        in = nominal_instance<P>(!a.has("no_noise"), po);
    } else {
        Rng irng(derive_seed(seed, 1));
        in = sample_instance<P>(irng, po);
        in.seed = seed;
    }
    if (a.has("no_noise"))
        std::fill(in.noise.begin(), in.noise.end(), 0.0);
    std::vector<double> theta;
    if (a.has("theta")) {
        theta = parse_doubles(a.get("theta"));
    } else {
        Rng trng(derive_seed(seed, 2));
        theta.resize(P::D);
        sample_theta<P>(trng, theta.data());
    }
    if (static_cast<int>(theta.size()) != P::D)
        throw std::runtime_error("--theta needs " + std::to_string(P::D) + " values");
    SimOpts so;
    so.deterministic = a.has("deterministic");
    std::vector<double> x0;
    if (a.has("x0")) {
        x0 = parse_doubles(a.get("x0"));
        if (x0.size() != 4) throw std::runtime_error("--x0 needs 4 values");
        so.x0 = x0.data();
    }
    Sim<P> sim;
    sim.prepare(in);
    Rng rng(seed);
    const Outcome o = sim.run(theta.data(), in, rng, so);
    std::string out = "{\"problem\": " + jstr(P::NAME) + ", \"theta\": " + jarr(theta.data(), P::D);
    out += ", \"instance\": " + instance_json<P>(in);
    out += ", \"crash_step\": " + (o.crash_step > 0 ? std::to_string(o.crash_step) : std::string("null"));
    out += ", \"success\": " + std::string(o.success ? "true" : "false");
    out += ", \"J\": " + jarr(sim.J.data(), P::T);
    out += ", \"x\": [";
    for (int t = 0; t < P::T; ++t) out += (t ? ", " : "") + jarr(sim.x.data() + std::size_t(t) * P::C, P::C);
    out += "], \"xdot\": ";
    if (P::CR > 0) {
        out += "[";
        for (int t = 0; t < P::T; ++t)
            out += (t ? ", " : "") + jarr(sim.xdot.data() + std::size_t(t) * P::CR, P::CR);
        out += "]";
    } else {
        out += "null";
    }
    out += "}\n";
    std::fwrite(out.data(), 1, out.size(), stdout);
    return 0;
}

// ------------------------------------------------------------------ speed
// Single-core generation speed: seconds per 10,000 single trajectories, trajectory acceptance,
// system acceptance (screen pass rate), seconds per 10,000 FT-PFN and XDecPFN trajectory sets.
template <class P>
int cmd_speed(const Args& a) {
    PriorOpts po;
    po.dead_time = !a.has("no_dead_time");
    ScreenCfg scfg;
    const std::uint64_t seed = a.getu("seed", 0);
    const long long n_traj = a.geti("count", 2000);
    const long long n_sets = a.geti("sets", 10);
    const double budget = a.getd("budget", 120.0);
    Common c;
    c.po = po;
    c.sc = scfg;
    c.quiet = true;
    c.workers = 1;
    const auto tp = Clock::now();
    const PilotResult pr = pilot<P>(PILOT_ROOT_SEED, a.geti("pilot", PILOT_SAMPLES), 1, po);
    const double pilot_s = seconds_since(tp);
    const double rho = a.has("rho") ? a.getd("rho", 0.0) : pr.rho_unsuccessful();
    const double rp = rejection_probability(rho);

    Sim<P> sim;
    double theta[16];
    // single trajectories
    Rng rng(derive_seed(seed, 0));
    long long rows = 0, attempts = 0;
    auto t0 = Clock::now();
    while (rows < n_traj && seconds_since(t0) < budget) {
        Instance in = sample_instance<P>(rng, po);
        sim.prepare(in);
        sample_theta<P>(rng, theta);
        const Outcome o = sim.run(theta, in, rng);
        ++attempts;
        if (category(o) != CAT_SUCCESS && rp > 0.0 && rng.uniform() < rp) continue;
        ++rows;
    }
    const double traj_s = seconds_since(t0);

    // XDecPFN set pools (screened instance, N_max trajectories, labels)
    const int nmax = n_max_trials<P>();
    const int n_labels = static_cast<int>(a.geti("labels", 128));
    long long xsets = 0, cand_total = 0, accepted = 0;
    t0 = Clock::now();
    while (xsets < n_sets && seconds_since(t0) < budget) {
        Rng r2(derive_seed(seed, std::vector<std::uint64_t>{1, static_cast<std::uint64_t>(xsets)}));
        ScreenResult sr;
        long long cand = 0;
        Instance in = draw_screened_instance(sim, r2, scfg, po, sr, cand);
        cand_total += cand;
        ++accepted;
        SetCounts sc = draw_counts(nmax, sr.rho_crash(), sr.rho_unsuccessful_given_ok(), r2);
        fill_set(sim, in, r2, sc, [](const double*, const Outcome&, int) {});
        for (int l = 0; l < n_labels; ++l) {
            sample_theta<P>(r2, theta);
            sim.run(theta, in, r2);
        }
        ++xsets;
    }
    const double xdec_s = seconds_since(t0);

    // FT-PFN examples
    const std::vector<int> grid = curve_steps(P::T, FTPFN_GRID);
    long long fsets = 0;
    t0 = Clock::now();
    ContextChunk ch;
    while (fsets < n_sets && seconds_since(t0) < budget) {
        Rng r3(derive_seed(seed, std::vector<std::uint64_t>{2, static_cast<std::uint64_t>(fsets)}));
        const long long before = ch.ex_ptr.size();
        context_example<P>(sim, r3, c, nmax, static_cast<int>(a.geti("n_unrun", 16)), grid, ch, 0);
        (void)before;
        ++fsets;
    }
    const double ft_s = seconds_since(t0);
    // the FT-PFN screens also count toward the system acceptance
    std::printf(
        "{\"problem\": %s, \"single_trajectories\": %lld, \"traj_seconds\": %s, "
        "\"traj_s_per_10k\": %s, \"traj_acceptance\": %s, \"rho_unsuccessful\": %s, "
        "\"pilot_seconds\": %s, \"xdecpfn_sets\": %lld, \"xdecpfn_s_per_10k\": %s, "
        "\"system_candidates\": %lld, \"system_acceptance\": %s, \"ftpfn_sets\": %lld, "
        "\"ftpfn_s_per_10k\": %s, \"n_max\": %d, \"labels\": %d, \"version\": %s}\n",
        jstr(P::NAME).c_str(), rows, jnum(traj_s).c_str(),
        jnum(rows > 0 ? traj_s / rows * 1e4 : NAN).c_str(),
        jnum(attempts > 0 ? double(rows) / attempts : NAN).c_str(), jnum(rho).c_str(),
        jnum(pilot_s).c_str(), xsets, jnum(xsets > 0 ? xdec_s / xsets * 1e4 : NAN).c_str(),
        cand_total, jnum(cand_total > 0 ? double(accepted) / cand_total : NAN).c_str(), fsets,
        jnum(fsets > 0 ? ft_s / fsets * 1e4 : NAN).c_str(), nmax, n_labels,
        jstr(generator_version()).c_str());
    return 0;
}

// ------------------------------------------------------------------ info and debugging
template <class P>
int cmd_info(const std::string& binary) {
    PriorOpts po;
    const Layout L = make_layout<P>("traj", "full");
    const Layout G = make_layout<P>("set", "grid");
    std::string o = "{\"problem\": " + jstr(P::NAME) + ", \"binary\": " + jstr(binary);
    o += ", \"version\": " + jstr(generator_version());
    o += ", \"d\": " + std::to_string(P::D) + ", \"T\": " + std::to_string(P::T);
    o += ", \"theta_lo\": " + jarr(P::THETA_LO, P::D) + ", \"theta_hi\": " + jarr(P::THETA_HI, P::D);
    o += ", \"theta_names\": " + jstrs(theta_names<P>());
    o += ", \"channels\": " + jstrs(channel_names<P>(false));
    o += ", \"rate_channels\": " + jstrs(channel_names<P>(true));
    o += ", \"phi_names\": " + jstrs(names_of<P>(P::PHI, P::N_PHI));
    o += ", \"extra_names\": " + jstrs(names_of<P>(P::EXTRA, P::N_EXTRA));
    o += ", \"noise_names\": " + jstrs(noise_names<P>());
    o += ", \"uses_phi_hat\": " + std::string(P::USES_PHI_HAT ? "true" : "false");
    o += ", \"can_crash\": " + std::string(P::CAN_CRASH ? "true" : "false");
    o += ", \"has_success\": " + std::string(P::HAS_SUCCESS ? "true" : "false");
    o += ", \"n_max\": " + std::to_string(n_max_trials<P>());
    const std::vector<int> ds = decoder_steps_with_horizon(P::T);
    o += ", \"decoder_steps_with_horizon\": " + jarr(ds.data(), static_cast<int>(ds.size()));
    const std::vector<int> cg = curve_steps(P::T, FTPFN_GRID);
    o += ", \"ftpfn_grid\": " + jarr(cg.data(), FTPFN_GRID);
    o += ", \"traj_row_floats\": " + std::to_string(L.R) + ", \"set_grid_row_floats\": " + std::to_string(G.R);
    o += ", \"nominal_instance\": " + instance_json<P>(nominal_instance<P>(true, po));
    o += "}\n";
    std::fwrite(o.data(), 1, o.size(), stdout);
    return 0;
}

inline int cmd_derive_seed(const Args& a) {
    std::vector<std::uint64_t> labels;
    std::stringstream ss(a.get("label", ""));
    std::string part;
    while (std::getline(ss, part, ','))
        if (!part.empty()) labels.push_back(std::stoull(part));
    std::printf("%llu\n", static_cast<unsigned long long>(derive_seed(a.getu("seed", 0), labels)));
    return 0;
}

inline int cmd_sobol(const Args& a) {
    const int d = static_cast<int>(a.geti("dim", 2));
    const int n = static_cast<int>(a.geti("n", 8));
    Sobol s(d, nullptr);
    double u[SOBOL_MAX_DIM];
    for (int i = 0; i < n; ++i) {
        s.next(u);
        for (int j = 0; j < d; ++j) std::printf("%s%.0f", j ? " " : "", u[j] * 4294967296.0);
        std::printf("\n");
    }
    return 0;
}

inline const char* USAGE =
    "subcommands: traj | sets | contexts | screen | screen-one | pilot | rollout | speed | info |\n"
    "             version | derive-seed | sobol\n"
    "chunked commands: --out DIR --count N --seed S --workers W --chunk K [--chunks LIST]\n"
    "  traj      [--rho R | --no-reject]\n"
    "  sets      [--resolution grid|full] [--labels 128] [--n-max N]\n"
    "  contexts  [--n-unrun 16] [--n-max N]\n"
    "  screen    [--screen-thetas 100 --screen-repeats 10 --screen-min-pass 25]\n"
    "  screen-one [--instance-json FILE|JSON | --nominal] [--seed S] [--full]\n"
    "  pilot     [--n 50000] [--pilot-seed S] [--workers W]\n"
    "  rollout   [--theta a,b,..] [--instance-json FILE|JSON | --nominal] [--seed S]\n"
    "            [--deterministic] [--no-noise] [--x0 a,b,c,d]\n"
    "  speed     [--count 2000] [--sets 10] [--budget 120]\n"
    "Furuta only: --no-dead-time (prior without actuation dead time).\n";

template <class P>
int dispatch(const std::string& cmd, const Args& a, const std::string& binary) {
    if (cmd == "traj") return cmd_traj<P>(a, binary);
    if (cmd == "sets") return cmd_sets<P>(a, binary);
    if (cmd == "contexts") return cmd_contexts<P>(a);
    if (cmd == "screen") return cmd_screen<P>(a);
    if (cmd == "pilot") return cmd_pilot<P>(a);
    if (cmd == "rollout") return cmd_rollout<P>(a);
    if (cmd == "screen-one") return cmd_screen_one<P>(a);
    if (cmd == "speed") return cmd_speed<P>(a);
    if (cmd == "info") return cmd_info<P>(binary);
    throw std::runtime_error("unknown subcommand '" + cmd + "'\n" + USAGE);
}

// Common entry: handles version / derive-seed / sobol, then runs `body(cmd, args)`.
template <class F>
int guarded_main(int argc, char** argv, const char* binary, F&& body) {
    try {
        if (argc < 2) {
            std::fprintf(stderr, "%s %s\n%s", binary, generator_version(), USAGE);
            return 2;
        }
        const std::string cmd = argv[1];
        if (cmd == "version" || cmd == "--version") {
            std::printf("%s %s\n", binary, generator_version());
            return 0;
        }
        if (cmd == "-h" || cmd == "--help" || cmd == "help") {
            std::printf("%s %s\n%s", binary, generator_version(), USAGE);
            return 0;
        }
        const Args a = parse_args(argc, argv, 2);
        if (cmd == "derive-seed") return cmd_derive_seed(a);
        if (cmd == "sobol") return cmd_sobol(a);
        return body(cmd, a);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s: error: %s\n", binary, e.what());
        return 1;
    }
}

}  // namespace esbo
