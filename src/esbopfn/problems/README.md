# Problems

Five simulated control problems, each a subpackage with `SPEC` (search box, horizon, channels),
`PRIOR` (universal prior) and `simulate(theta, instance, rng, t_max=None) -> Trajectory`.
`get_problem(name)` bundles the three. The numbers live in `configs/problems.yaml`, which mirrors
the thesis appendix tables: `tab:app:problems` (plant, channels, cost), `tab:app:grid` (horizons),
`tab:app:prior-phi` (physical parameters), `tab:app:prior-noise` (noise bases) and
`tab:app:theta-box` (search boxes). The numerics are ports of the Python simulators the thesis
used, checked against reference rollouts in `tests/golden`.

All step indices are 1-based. `J` is `+inf` from the crash step on. `success` is `not crashed`
except on Furuta, which also needs the pendulum held upright.

## Universal prior (`prior.py`)

- Scale entries: `LogUniform(p0 / f_lo, p0 * f_hi)`. Bounded entries: `Uniform(lo, hi)`. Tied
  entries: `p_src * LogUniform(1/zeta, zeta)`.
- Noise per group: `sigma = base * S * eta` with `S ~ LogUniform(1.5/4, 4/1.5)` per instance and
  `eta ~ LogUniform(1/1.5, 1.5)` per group, so `sigma` lies in `[base/4, 4 base]`. Gaussian noise
  is added at every simulation step.
- Estimate (Furuta only): `p_hat_i = max(0, N(p_i, (p_i/10)^2))`.
- `screen`: 100 random controllers, keep the instance if at least 25 succeed in all 10 repeats.
- `single_trajectory_prior`: instance and controller per draw, unsuccessful trajectories rejected
  with probability `max(0, 1 - (1 - rho)/rho)`, `rho` from a seeded pilot.
- `trajectory_set_prior`: exact counts of crashed, unsuccessful and successful trials from
  `rho'_c ~ U(0, rho_c)` and `rho'_{sbar|cbar} ~ U(0, rho_{sbar|cbar})`, then one cut probability
  `~U(0, 1)` per example and cutoffs `~U{1..T_i}`.

## Boiler (`boiler`, T = 701, d = 1)

Port of MathWorks' `sldemo_boiler`. The temperature rises at `heating_rate / 25` per second while
the burner is on and falls at `cooling_rate / 25` while it is off, from 15 degC, with 1 s Euler
steps. The thermometer adds sensor noise and quantizes with an 8-bit ADC (0.390625 degC steps).
The bang-bang chart turns on after the turn-on delay `theta` once the reading is at or below
20 degC, and turns off above 20 degC or after 100 s on. Logged every 2 s for 1400 s. Cost per step
`0.5 min(|20 - T_meas|, 2)^2 + u + 8 [u turned on]`. Channel `temperature` (measured). No crash.
Prior: heating rate 1.0 and cooling rate 0.1 (`tab:app:prior-phi`, Scale(5, 5)), noise base
0.03 degC.

## Three-Tank (`threetank_pi`, T = 476, d = 2)

Port of the RWTH three-tank rig. Three tanks in series (B102, B103, B104) with signed Torricelli
throttles, the identified V101 valve map between B102 and B103, the identified pump map, a 1.2341 s
pump lag and 1.1 s transport delay, and the identified V106 outflow map, Euler at 0.1 s with
volumes saturating at 0 and 10e-3 m^3. The tuned outer PI `(K_P, K_I)` on the measured B104 level
feeds a fixed inner PI on the pump flow. The set point steps to 3e-3 m^3 at 5 s, where logging
starts (every 0.2 s up to 100 s). Cost `(3e-3 - V4_meas)^2`. Channels `V2`, `V3` (true levels) and
`V4` (measured, noisy). No crash. Prior: V101 opening U[10, 85] %, V106 opening U[10, 90] %, B101
volume U[0.040, 0.130] m^3 (nominal 43.3 %, 44.7 %, 0.080 m^3), noise base 1.7e-4 m^3.

## PT2 dead-time (`pt2_deadtime`, T = 501, d = 3)

`G(s) = K w0^2 / (s^2 + 2 D w0 s + w0^2)` with a 0.1 s input dead time, Euler at 0.01 s. Parallel
PID with derivative filter N = 100 and gains `10**theta`, tracking `w = 1`. Logged every 0.1 s for
50 s. Crash when the measured output exceeds `w + 2` at a logged step. Cost (ITAE)
`(t - 1) 0.1 |1 - y_meas|`. Channel `y` (measured). Prior: K = 4, w0 = 0.5, D = 0.5, each
Scale(5, 5), noise base 4e-5.

## Cartpole (`cartpole`, T = 1000, d = 4)

Gymnasium cart-pole, explicit Euler at 0.02 s, start state `U(-0.05, 0.05)^4`. Two PD loops on the
measured cart position error and pole angle (derivatives are per-step differences) act on the
previous measurement, and the force is `+max_force` if the summed output is negative and
`-max_force` otherwise. The goal is 0 for 100 steps, then +2, -2, +2, -2 m for 200, 300, 200 and
200 steps. Crash when the cart leaves +-3.5 m or the pole +-30 deg. Cost
`10 tanh^2(x_ref - x) + xdot^2 + phi^2 + phidot^2` on the measurements. Channels `x_cart`,
`phi_pole`, rates `xdot_cart`, `phidot_pole`, all noisy. Prior: cart mass 1.0 and pole mass 0.1
(Scale(8, 8)), pole half-length 0.5 and maximum force 10 (Scale(3, 3)), noise bases 0.004 m,
0.02 m/s, 0.002 rad, 0.02 rad/s.

## Furuta (`furuta`, T = 600, d = 8)

QUBE-Servo 2 model (Quanser workbook equations, rod inertias `m L^2/12`, motor torque
`-kt (V - km theta_dot) / Rm`), ten semi-implicit Euler sub-steps per 1/120 s step, voltage
clipped to 18 V at the plant and 6 V at the controller. Outside 20 deg of upright the energy pump
`mu (E - E_bar m_p g L_p) sign(-cos(alpha) alpha_dot)` plus the barrier `K_P^barrier sign(theta)
theta^2` drives the arm, inside it the discrete LQR with `Q = diag(q1..q4)`, `R = r` designed from
`p_hat` holds the pendulum. The command passes the per-instance dead time (fractional delay line),
then voltage noise is added. Each trial starts from the rig's reset: rest hanging down with a 0.01
residual, then a 30 ms kick of the per-instance amplitude with a random sign. Crash when the arm's
continuous angle leaves +-90 deg (or the state diverges or the LQR cannot be designed). Success
when not crashed and the mean measured `|phi_pend|` over the last 2 s is below 5 deg. Cost
`|phi_pend| (1 + 20 (1 - min(|phi_pend|, 5 deg) / 5 deg))`. Channels `phi_arm`, `phi_pend`, rates
`phidot_arm`, `phidot_pend`, all noisy. Prior (`tab:app:prior-phi`): Rm 8.4 and kt 0.042 Scale(4, 5),
km tied to kt (1.5), mr 0.095 Scale(4, 8), Lr 0.085 Scale(3, 4), Dr 2.75e-4 Scale(10, 3), mp 0.024
Scale(4, 10), Lp 0.129 Scale(3, 4), Dp 5.05e-5 Scale(10, 3). Extras outside `p_hat`: dead time
U[1/120, 3/120] s and kick U[0, 0.6] V. Noise bases 0.002 rad (both angles, one draw), 0.07 rad/s
(both rates, one draw), 0.1 V (motor voltage, actuation). `get_problem("furuta",
dead_time=False)` fixes the dead time at zero.
