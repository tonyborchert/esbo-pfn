// Random numbers and seed derivation.
//
// derive_seed() is a port of numpy's SeedSequence entropy mixing, so that
//     derive_seed(root, label) == esbopfn.util.seeds.derive_seed(root, label)
// for non-negative integer roots and labels below 2**64. Chunk k of a dataset with seed S is
// generated from Rng(derive_seed(S, k)), which makes every pool a pure function of (S, chunk size)
// and independent of the number of worker threads or processes.
#pragma once

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace esbo {

// ------------------------------------------------------------------ numpy SeedSequence port
namespace seedseq {

constexpr std::uint32_t INIT_A = 0x43b0d7e5u;
constexpr std::uint32_t MULT_A = 0x931e8875u;
constexpr std::uint32_t INIT_B = 0x8b51f9ddu;
constexpr std::uint32_t MULT_B = 0x58f38dedu;
constexpr std::uint32_t MIX_MULT_L = 0xca01f9ddu;
constexpr std::uint32_t MIX_MULT_R = 0x4973f715u;
constexpr int XSHIFT = 16;
constexpr int POOL_SIZE = 4;

inline std::uint32_t hashmix(std::uint32_t value, std::uint32_t& hash_const) {
    value ^= hash_const;
    hash_const *= MULT_A;
    value *= hash_const;
    value ^= value >> XSHIFT;
    return value;
}

inline std::uint32_t mix(std::uint32_t x, std::uint32_t y) {
    std::uint32_t r = MIX_MULT_L * x - MIX_MULT_R * y;
    r ^= r >> XSHIFT;
    return r;
}

// numpy `_int_to_uint32_array`: little-endian 32-bit words, [0] for zero.
inline void push_words(std::uint64_t v, std::vector<std::uint32_t>& out) {
    if (v == 0) {
        out.push_back(0u);
        return;
    }
    while (v != 0) {
        out.push_back(static_cast<std::uint32_t>(v & 0xffffffffu));
        v >>= 32;
    }
}

// Pool of SeedSequence(entropy=root, spawn_key=labels).
inline void pool_of(std::uint64_t root, const std::vector<std::uint64_t>& labels,
                    std::uint32_t pool[POOL_SIZE]) {
    std::vector<std::uint32_t> entropy;
    push_words(root, entropy);
    std::vector<std::uint32_t> spawn;
    for (std::uint64_t lb : labels) push_words(lb, spawn);
    if (!spawn.empty() && entropy.size() < static_cast<std::size_t>(POOL_SIZE))
        entropy.resize(POOL_SIZE, 0u);
    entropy.insert(entropy.end(), spawn.begin(), spawn.end());

    std::uint32_t hash_const = INIT_A;
    for (int i = 0; i < POOL_SIZE; ++i)
        pool[i] = hashmix(i < static_cast<int>(entropy.size()) ? entropy[i] : 0u, hash_const);
    for (int src = 0; src < POOL_SIZE; ++src)
        for (int dst = 0; dst < POOL_SIZE; ++dst)
            if (src != dst) pool[dst] = mix(pool[dst], hashmix(pool[src], hash_const));
    for (std::size_t src = POOL_SIZE; src < entropy.size(); ++src)
        for (int dst = 0; dst < POOL_SIZE; ++dst)
            pool[dst] = mix(pool[dst], hashmix(entropy[src], hash_const));
}

inline void generate_state(const std::uint32_t pool[POOL_SIZE], std::uint32_t* out, int n_words) {
    std::uint32_t hash_const = INIT_B;
    for (int i = 0; i < n_words; ++i) {
        std::uint32_t v = pool[i % POOL_SIZE];
        v ^= hash_const;
        hash_const *= MULT_B;
        v *= hash_const;
        v ^= v >> XSHIFT;
        out[i] = v;
    }
}

}  // namespace seedseq

// 63-bit seed of the substream (root, *labels), equal to esbopfn.util.seeds.derive_seed.
inline std::uint64_t derive_seed(std::uint64_t root, const std::vector<std::uint64_t>& labels) {
    std::uint32_t pool[seedseq::POOL_SIZE];
    seedseq::pool_of(root, labels, pool);
    std::uint32_t words[2];
    seedseq::generate_state(pool, words, 2);
    const std::uint64_t v = static_cast<std::uint64_t>(words[0]) |
                            (static_cast<std::uint64_t>(words[1]) << 32);
    return v & ((std::uint64_t(1) << 63) - 1);
}

inline std::uint64_t derive_seed(std::uint64_t root, std::uint64_t label) {
    return derive_seed(root, std::vector<std::uint64_t>{label});
}

inline std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// Sub-seed of item `index` inside a stream seeded with `seed` (63 bits).
inline std::uint64_t sub_seed(std::uint64_t seed, std::uint64_t index) {
    return splitmix64(seed ^ splitmix64(index + 0x632BE59BD9B4E019ULL)) &
           ((std::uint64_t(1) << 63) - 1);
}

// ------------------------------------------------------------------ generator
struct Rng {
    std::mt19937_64 gen;
    std::normal_distribution<double> nd{0.0, 1.0};
    std::uniform_real_distribution<double> ud{0.0, 1.0};

    explicit Rng(std::uint64_t seed) : gen(splitmix64(seed)) {}

    double normal() { return nd(gen); }
    double uniform() { return ud(gen); }
    double uniform(double lo, double hi) { return lo + (hi - lo) * ud(gen); }
    double loguniform(double lo, double hi) {
        return std::exp(uniform(std::log(lo), std::log(hi)));
    }
    // Uniform integer in [lo, hi] inclusive.
    long long integer(long long lo, long long hi) {
        std::uniform_int_distribution<long long> d(lo, hi);
        return d(gen);
    }
    int binomial(int n, double p) {
        if (n <= 0 || p <= 0.0) return 0;
        if (p >= 1.0) return n;
        std::binomial_distribution<int> d(n, p);
        return d(gen);
    }
    std::uint64_t bits() { return gen(); }
};

// A generator that draws nothing random: normal() = 0, used for deterministic parity rollouts.
struct ZeroNoise {
    double normal() { return 0.0; }
};

}  // namespace esbo
