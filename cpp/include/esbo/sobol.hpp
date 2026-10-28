// Sobol' points with a random digital shift, used only for the theta design of the degeneracy
// screen. Direction numbers are the Joe-Kuo new-joe-kuo-6.21201 table for dimensions 1..8, so with
// a zero shift the points equal scipy.stats.qmc.Sobol(d, scramble=False). The random digital shift
// (one uniform 32-bit word per dimension, XORed into every point) keeps every point marginally
// uniform. Each screened candidate instance gets its own shift.
#pragma once

#include <cstdint>

namespace esbo {

constexpr int SOBOL_MAX_DIM = 8;
constexpr int SOBOL_BITS = 32;

// Joe-Kuo rows for dimensions 2..8 (dimension 1 is van der Corput): degree s, coefficient a and
// initial direction numbers m_1..m_s.
inline const int SOBOL_S[SOBOL_MAX_DIM] = {0, 1, 2, 3, 3, 4, 4, 5};
inline const std::uint32_t SOBOL_A[SOBOL_MAX_DIM] = {0, 0, 1, 1, 2, 1, 4, 2};
inline const std::uint32_t SOBOL_M[SOBOL_MAX_DIM][5] = {
    {0, 0, 0, 0, 0}, {1, 0, 0, 0, 0}, {1, 3, 0, 0, 0}, {1, 3, 1, 0, 0},
    {1, 1, 1, 0, 0}, {1, 1, 3, 3, 0}, {1, 3, 5, 13, 0}, {1, 1, 5, 5, 17}};

struct SobolDirections {
    std::uint32_t v[SOBOL_MAX_DIM][SOBOL_BITS];

    SobolDirections() {
        for (int k = 0; k < SOBOL_BITS; ++k) v[0][k] = 1u << (SOBOL_BITS - 1 - k);
        for (int j = 1; j < SOBOL_MAX_DIM; ++j) {
            const int s = SOBOL_S[j];
            const std::uint32_t a = SOBOL_A[j];
            for (int k = 0; k < s; ++k) v[j][k] = SOBOL_M[j][k] << (SOBOL_BITS - 1 - k);
            for (int k = s; k < SOBOL_BITS; ++k) {
                std::uint32_t x = v[j][k - s] ^ (v[j][k - s] >> s);
                for (int i = 1; i < s; ++i) x ^= (((a >> (s - 1 - i)) & 1u) * v[j][k - i]);
                v[j][k] = x;
            }
        }
    }
};

inline const SobolDirections& sobol_directions() {
    static const SobolDirections d;
    return d;
}

// Gray-code order Sobol' block. next() fills `dim` values in [0, 1).
class Sobol {
public:
    Sobol(int dim, const std::uint32_t* shift) : dim_(dim) {
        for (int j = 0; j < SOBOL_MAX_DIM; ++j) {
            x_[j] = 0u;
            shift_[j] = (shift != nullptr && j < dim) ? shift[j] : 0u;
        }
    }

    void next(double* u) {
        if (i_ != 0) {
            int c = 0;
            std::uint64_t t = i_;
            while ((t & 1ULL) == 0ULL) {
                ++c;
                t >>= 1;
            }
            const SobolDirections& dd = sobol_directions();
            for (int j = 0; j < dim_; ++j) x_[j] ^= dd.v[j][c];
        }
        for (int j = 0; j < dim_; ++j)
            u[j] = static_cast<double>(x_[j] ^ shift_[j]) * (1.0 / 4294967296.0);
        ++i_;
    }

private:
    int dim_;
    std::uint64_t i_ = 0;
    std::uint32_t x_[SOBOL_MAX_DIM];
    std::uint32_t shift_[SOBOL_MAX_DIM];
};

}  // namespace esbo
