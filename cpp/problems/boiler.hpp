// Boiler: port of MathWorks' sldemo_boiler. Bang-bang burner (heating +rate/25, cooling
// -rate/25 per second) from 15 degC, 8-bit ADC thermometer (step 25/64 degC, bias -15 degC),
// hysteresis controller with tunable turn-on delay and a 100 s safety off time, set point 20 degC,
// 1 s Euler steps logged every 2 s (T = 701). Never crashes. Stage cost
// 0.5 min(|20 - meas|, 2)^2 + u + 8 [u rose], with no switching penalty at the first step.
#pragma once

#include <algorithm>
#include <cmath>

#include "esbo/problem.hpp"

namespace esbo {
namespace boiler {

constexpr int N_STEPS = 1400;
constexpr int LOG_EVERY = 2;
constexpr double CTRL_DT = 1.0;
constexpr double DT = 1.0;
constexpr double PLANT_GAIN = 1.0 / 25.0;
constexpr double T_INIT = 15.0;
constexpr double ADC_STEP = 0.390625;
constexpr double ADC_BIAS = -15.0;
constexpr double SET_POINT = 20.0;
constexpr double SAFETY_OFF_TIME = 100.0;
constexpr float COMFORT_SCALING = 0.5f;
constexpr float COMFORT_CLIP = 2.0f;
constexpr float SWITCH_PENALTY = 8.0f;

inline double adc_quantize(double temp) {
    return ADC_BIAS + ADC_STEP * std::floor((temp - ADC_BIAS) / ADC_STEP);
}

inline constexpr ParamSpec PHI_TABLE[2] = {
    scale("heating_rate", 1.0, 5.0, 5.0),
    scale("cooling_rate", 0.1, 5.0, 5.0),
};
inline constexpr NoiseSpec NOISE_TABLE[1] = {{"temperature", 0.03}};
inline constexpr double THETA_LO_T[1] = {5.0};
inline constexpr double THETA_HI_T[1] = {500.0};
inline const char* const THETA_NAMES_T[1] = {"turn_on_delay"};
inline const char* const CHANNELS_T[1] = {"temperature"};

struct Problem {
    static constexpr const char* NAME = "boiler";
    static constexpr int D = 1, T = 701, C = 1, CR = 0;
    static constexpr int N_PHI = 2, N_EXTRA = 0, N_NOISE = 1;
    static constexpr bool USES_PHI_HAT = false, CAN_CRASH = false, HAS_SUCCESS = false;
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
        const double heating = in.phi[0], cooling = in.phi[1];
        const double turn_on_delay = theta[0];
        const double sigma = in.noise[0];
        bool on = false;
        double time_in_state = 0.0;
        int u = 0;
        auto observe = [&](double digital_temp) {
            const bool cold = digital_temp <= SET_POINT;
            if (!on) {
                if (time_in_state >= turn_on_delay && cold) { on = true; time_in_state = 0.0; }
            } else if (!cold || time_in_state >= SAFETY_OFF_TIME) {
                on = false;
                time_in_state = 0.0;
            }
            u = on ? 1 : 0;
            time_in_state += CTRL_DT;
            return u;
        };
        auto sense = [&](double temp) {
            return adc_quantize(temp + (sigma > 0.0 ? sigma * rng.normal() : 0.0));
        };
        int n = 0;
        float u_prev = 0.0f;
        double acc = 0.0;
        auto log_step = [&](double meas, int uu) {
            const float m32 = static_cast<float>(meas);
            const float u32 = static_cast<float>(uu);
            float e = std::fabs(static_cast<float>(SET_POINT) - m32);
            e = std::min(e, COMFORT_CLIP);
            const float comfort = COMFORT_SCALING * (e * e);
            const float turned_on = (n > 0 && u32 > u_prev) ? 1.0f : 0.0f;
            const float ji = comfort + u32 + SWITCH_PENALTY * turned_on;
            u_prev = u32;
            acc += static_cast<double>(ji);
            J[n] = static_cast<float>(acc);
            x[n] = m32;
            ++n;
        };
        double temp = T_INIT;
        double meas = sense(temp);
        log_step(meas, observe(meas));
        for (int k = 1; k <= N_STEPS; ++k) {
            temp += DT * PLANT_GAIN * (u == 1 ? heating : -cooling);
            meas = sense(temp);
            const int uu = observe(meas);
            if (k % LOG_EVERY == 0) log_step(meas, uu);
        }
        Outcome out;
        finish_trajectory(T, C, CR, n, 0, J, x, nullptr);
        return out;
    }
};

}  // namespace boiler
}  // namespace esbo
