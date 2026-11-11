# Synthetic generative domain (chain pilot)

## Valid static parameters

Registered protocols validate `scenario_params` before runs:

| Field | Constraint |
|-------|------------|
| `layout` | `chain` only |
| `n_links` | strictly finite integral count ≥ 2 (no fractional truncation; bool/string rejected) |
| `n_steps` | strictly finite integral count ≥ 2 |
| `demand_vph` | finite, ≥ 0 |
| `capacity_vehicles` | finite, > 0 |
| `noise_std` | finite, ≥ 0 |
| `missing_rate` | finite, in `[0, 1)` |

RNG-dependent stresses (e.g. `shifted_demand` with `boundary_shift=0.25`) remain in the declared stress registry; the integrator throttles flux rather than widening capacity.

## Flux and CFL policy

Discrete balance: `N[t+1] = N[t] + dt (B z[t] + R u[t])` with vehicles and vph.

Each step applies **start-of-step** supply/demand caps (explicit Euler, not full CTM/LWR):

- Send from cell `i`: `z_i ≤ N_i / dt`
- Receive at cell `i`: inflow ≤ `(cap_i - N_i) / dt`
- Inlet `u[:,0]` and outlet magnitude `u[:,1]` are independent nonnegative channels on `R[:,0]` and `R[:,1]`.
- Excess boundary demand above start-of-step send/receive caps is **rejected at that step** (throttled to feasible flux); there is **no upstream external queue** storing deferred boundary inflow.

Corridor vehicle change over the horizon equals net external boundary injection `sum_t dt (u_in - u_out)`.

## Amendment (2026-11-11)

Final holdout v3 (`n_links=7` with development demand/capacity) failed because the legacy generator sampled flux independently then integrated. The feasible flux integrator applies supply caps first; **v3 seeds remain archived** (`holdout_status: retired_failed`). A new generator enables future evaluation; it does not revise the preregistered failed final attempt (`results/research_final_v3/final_attempt_failure.json`).
