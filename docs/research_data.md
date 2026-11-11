# Measured-Data Feasibility and Provenance for CREST

This document establishes the scientific provenance, access feasibility, physical limitations, and mathematical identifiability of real-world measured freeway datasets for the **CREST** (*Constrained Reconstruction and Estimation of Spatiotemporal Traffic*) research implementation.

---

## Executive Summary

The CREST system was developed and validated on the synthetic *TrafficFlowBench* release (IEEE Big Data Cup 2026, Challenge 06, Track 2). While the synthetic benchmark enforces exact macroscopic conservation and consistent Fundamental Diagram relationships, **real-world measured traffic observations present fundamental observational gaps, measurement noise, and identifiability limits**.

To make real-data follow-through concrete, honest, and reviewable:
1. **Caltrans PeMS** is confirmed as **blocked** for automated anonymous access: registration and account approval are required; secondary redistribution terms remain unverified from primary Caltrans text.
2. **I-24 MOTION full INCEPTION portal** (`i24motion.org/data`) is confirmed as **blocked** for direct automated downloading: automated requests trigger security appliance (WAF) rejection; web registration requires manual approval ("New accounts will gain access after site approval within one business day").
3. **USDOT ITS DataHub NGSIM Trajectories** were **verified, safely fetched, and cryptographically verified** via the official public SODA API with zero credentials:
   - A bounded slice of 25,000 deterministic records (4.37 MB) from Interstate 80 Eastbound was downloaded into `data/research_external/ngsim_i80_slice.csv` (SHA-256: `38173aad33806922b4f22f516363f8a4fce53038554e6d85744e151f05870d1b`).
   - Licensed under U.S. Federal Government Open Data (Public Domain).
4. **I-24 MOTION Feasibility Trajectory Sample** was **verified and fetched** from the official open-source release (Derek Gloudemans / Vanderbilt University TDOT project):
   - A 60-second multi-camera trajectory tracking sample (5.53 MB, 1,801 frames) was downloaded into `data/research_external/i24_motion_sample_3.json` (SHA-256: `857ee641cbcff3b56974bde323fcb8c126dddccca9ae0f336b559a4027e8cc17`).
   - **Software** on the GitHub examples repo is MIT-licensed; **trajectory sample redistribution/commercial repackaging terms are unverified** in this manifest except citation and no-reidentification intent for the full portal (registration required).
5. **Generic speed-only datasets (METR-LA, PEMS-BAY)** are **explicitly excluded** from conservation ground-truth evaluation because they record speed only without vehicle counts, volume, or ramp fluxes. Substituting speed-only benchmarks for physical conservation is mathematically invalid.

Total measured data fetched is **9.9 MB**, adhering strictly to the safety budget (< 100 MB).

---

## Data Provenance and Verified Sources

The machine-readable catalog is maintained in [`configs/research_data_manifest.json`](../configs/research_data_manifest.json).

### 1. USDOT NGSIM I-80 Freeway Trajectories (Verified Fetched)

- **Official Catalog Entry**: [Data.gov NGSIM Entry](https://catalog.data.gov/dataset/next-generation-simulation-ngsim-vehicle-trajectories-and-supporting-data)
- **SODA API Endpoint**: `https://data.transportation.gov/resource/8ect-6jqj.csv`
- **Query URL**: `https://data.transportation.gov/resource/8ect-6jqj.csv?$where=location='i-80'&$order=global_time,vehicle_id&$limit=25000`
- **Local Path**: [`data/research_external/ngsim_i80_slice.csv`](../data/research_external/ngsim_i80_slice.csv)
- **File Size**: 4,369,161 bytes (~4.17 MB)
- **SHA-256**: `38173aad33806922b4f22f516363f8a4fce53038554e6d85744e151f05870d1b`
- **Retrieved UTC**: `2026-11-11T06:37:07.346784+00:00`
- **Record Count**: 25,000 rows raw on disk (SHA-256 unchanged). Adapter retains **24,999** rows over **~65.5 s** after conservative **edge-only** trim (sparse first timestamp only; interior peaks retained).
- **Raw `global_time` range (ms)**: `1113433135300` … `1113433200900` (657 unique timestamps, 86 vehicles; UTC 2005-04-13 22:58:55.300–23:00:00.900).
- **Geography & Temporal Context**:
  - Location: Eastbound Interstate 80 in Emeryville, California (San Francisco Bay Area).
  - Geometry: 1,650 ft (500 m) corridor, 6 mainline lanes + Powell Street on-ramp and auxiliary lane.
  - Recording Period: April 13, 2005, 10 Hz sampling frequency (0.1 s per frame).
- **License & Permission**:
  - U.S. Federal Government Open Data (Public Domain / CCO-compatible).
  - Citation: Federal Highway Administration (FHWA), *Next Generation Simulation (NGSIM) Program*, U.S. Department of Transportation.
- **Fields & Units**:
  - `vehicle_id` (int): Vehicle tracking index.
  - `frame_id` (int): Frame index (0.1 s increments).
  - `global_time` (int): Epoch timestamp in milliseconds.
  - `local_x`, `local_y` (float): Lateral and longitudinal coordinates (feet).
  - `global_x`, `global_y` (float): California State Plane NAD83 coordinates (feet).
  - `v_length`, `v_width` (float): Vehicle physical dimensions (feet).
  - `v_class` (int): 1 = Motorcycle, 2 = Passenger Car, 3 = Heavy Truck/Bus.
  - `v_vel` (float): Instantaneous velocity ($\text{ft/s}$).
  - `v_acc` (float): Instantaneous acceleration ($\text{ft/s}^2$).
  - `lane_id` (int): Lane index (1 = leftmost HOV lane, 2–5 = mainline, 6 = auxiliary, 7 = on-ramp).
  - `preceding`, `following` (int): Leader/follower vehicle IDs.
  - `space_headway` (float): Front-to-front distance (feet).
  - `time_headway` (float): Headway clearance (seconds).
  - `o_zone`, `d_zone`, `int_id`, `section_id`, `direction`, `movement`: Present in SODA schema but `null` for freeway records.

### 2. I-24 MOTION Feasibility Trajectory Sample (Verified Fetched)

- **Source Repository**: `DerekGloudemans/I24-MOTION-examples` (lead author of the I-24 MOTION system)
- **Download URL**: `https://raw.githubusercontent.com/DerekGloudemans/I24-MOTION-examples/master/example_data/trajectory_data_3.json`
- **Local Path**: [`data/research_external/i24_motion_sample_3.json`](../data/research_external/i24_motion_sample_3.json)
- **File Size**: 5,532,215 bytes (~5.28 MB)
- **SHA-256**: `857ee641cbcff3b56974bde323fcb8c126dddccca9ae0f336b559a4027e8cc17`
- **Retrieved UTC**: `2026-11-11T06:37:15.722830+00:00`
- **Record Count**: 1,801 frames (60 seconds at 30 fps)
- **Geography & Context**:
  - Interstate 24 near Nashville, Tennessee (Pole 1, Camera 3).
  - 4K video vehicle detections mapped to local world coordinates and GPS coordinates.
- **License & Permission Evidence**:
  - **Repository software**: MIT License on `DerekGloudemans/I24-MOTION-examples` (code only).
  - **Trajectory JSON blob**: redistribution/commercial terms **unverified** from primary portal DUA in this review; do not redistribute derived coordinates without signed agreement.
  - Citation: Gloudemans et al., 2023, TR-C 155, 104311.
- **Coordinates**: `world_position` in the open sample is **not** loaded as metric longitudinal trajectories (unverified axis basis; adapter fails closed).

---

## Blocked Sources and Missing Prerequisites Status

| Source | Access Endpoint | Technical / Policy Blocker | Missing Prerequisites |
|---|---|---|---|
| **Caltrans PeMS** | `https://pems.dot.ca.gov/` | Registration and approval gate; session authentication; unattended export blocked | Approved account credentials; verified redistribution authorization |
| **I-24 MOTION INCEPTION Portal** | `https://i24motion.org/data` | Security appliance (WAF) rejection for non-browser clients; web portal requires manual administrator approval ("1 business day approval") | Registered `i24motion.org` user; signed Data Use Agreement compliance; manual administrator activation |
| **METR-LA / PEMS-BAY** | `https://github.com/liyaguang/DCRNN` | Speed-only measurement; no vehicle counts, no volume, no ramp measurements, no density | Scientifically excluded: cannot evaluate conservation law ($S_{\mathrm{LWR}}$) or OD demand ($Ax = y$) |

---

## Identifiability Matrix across CREST Tasks

The CREST architecture addresses four evaluation targets. Below is the strict scientific assessment of what measured freeway data actually identifies versus what remains **unidentifiable**:

```
+----------------------------------------------------------------------------------------------------+
| Task                   | Synthetic Benchmark              | Measured Trajectory Reality (NGSIM/I-24)|
+----------------------------------------------------------------------------------------------------+
| Task 1: State          | Masked cells (R1, R2, R3) on     | SUPPORTED via Edie's generalized       |
| Reconstruction         | 5-min link speeds & flows        | macro aggregation (dx, dt bins).       |
|                        |                                  | Limited to micro/meso corridor lengths.|
+----------------------------------------------------------------------------------------------------+
| Task 2: Queue          | Binary is_queue indicator        | PROXY ONLY. Breakdown thresholds       |
| Prediction             | withheld on all splits           | (v < 25 mph) can be observed, but      |
|                        |                                  | official queue labels are unobservable|
|                        |                                  | without arbitrary human definitions.   |
+----------------------------------------------------------------------------------------------------+
| Task 3: Conservation   | Evaluates discrete LWR:          | UNIDENTIFIABLE AS EXACT GROUND TRUTH.  |
| & Fundamental Diagram  | Storage delta = Net Inflow       | Occlusion, track dropouts, and boundary|
|                        | S_physics = 0.5(S_FD + S_LWR)    | clipping cause false mass destruction. |
|                        |                                  | Loop pulse drift causes runaway storage|
+----------------------------------------------------------------------------------------------------+
| Task 4: Origin-        | Solves regularized Ax = y on     | UNIDENTIFIABLE. Corridor is a single   |
| Destination Estimation | network paths with weak prior    | linear route (no alternative paths).   |
|                        |                                  | Freeway o_zone/d_zone fields are null. |
+----------------------------------------------------------------------------------------------------+
```

### Deep Dive: Why Discrete Conservation Cannot Be Proved on Measured Trajectories

In synthetic simulations, the continuity equation is an algebraic identity:
$$\Delta N_i(t) = N_i(t) - N_i(t - \Delta t) = [q_{\mathrm{in}, i}(t) - q_{\mathrm{out}, i}(t) + r_{\mathrm{on}, i}(t) - r_{\mathrm{off}, i}(t)] \Delta t$$

In real computer-vision trajectory tracking (NGSIM and I-24):
1. **Camera Occlusion**: High-profile vehicles (Class 3 heavy trucks) occlude smaller passenger cars. A vehicle may enter upstream, disappear behind a truck at $y = 800\text{ ft}$, and fail to be recorded at the downstream line. This creates an apparent "loss of mass".
2. **Track Fragmentation**: When tracking algorithms drop a track and re-initialize it several frames later under a new `vehicle_id`, a vehicle appears inside the box without an entry crossing. This creates an apparent "spontaneous creation of mass".
3. **Boundary Clipping**: Vehicles in weaving sections change lanes across camera stitching boundaries or near ramp tapers.

Similarly, in inductive loop detector networks (such as Caltrans PeMS):
- Loop counters experience 2% to 10% counting error due to multi-axle trucks, tailgating, or detector sensitivity drift.
- Integrating $(q_{\mathrm{in}} - q_{\mathrm{out}})\Delta t$ over hours produces unbounded cumulative drift, leading to fictitious negative densities or millions of phantom queued vehicles.

> [!IMPORTANT]
> **Scientific Finding**: Real measured freeway data can support **approximate conservation bounds** or **bounded residual estimation**, but cannot be claimed as "exact ground-truth conservation" without introducing unmeasured boundary slack variables.

---

## Suggested Adapter Contract for Core Pipeline

To bridge measured trajectory data with the macroscopic CREST pipeline, we propose the following **Trajectory-to-Macro Adapter Contract** based on Edie's generalized definitions (Edie, 1963):

For any road link $l$ of length $\Delta x$ over a time interval $\Delta t$, defining space-time domain $A = \Delta x \times \Delta t$:
- **Total travel time**: $T(A) = \sum_{i} t_i$ (where $t_i$ is time spent by vehicle $i$ in $A$)
- **Total distance traveled**: $D(A) = \sum_{i} d_i$ (where $d_i$ is distance traveled by vehicle $i$ in $A$)
- **Generalized Flow**: $q(A) = \frac{D(A)}{|A|} = \frac{\sum d_i}{\Delta x \Delta t}$
- **Generalized Density**: $k(A) = \frac{T(A)}{|A|} = \frac{\sum t_i}{\Delta x \Delta t}$
- **Space Mean (Harmonic) Speed**: $v(A) = \frac{q(A)}{k(A)} = \frac{D(A)}{T(A)}$

This satisfies the fundamental relationship $q = k \cdot v$ mathematically on any space-time region without division-by-zero singularities.

### Python Reference Adapter Implementation

The reference contract is implemented and tested in [`tests/test_research_data_sources.py`](../tests/test_research_data_sources.py):

```python
def trajectory_to_link_state(
    df_trajectories: pd.DataFrame,
    x_start_ft: float,
    x_end_ft: float,
    t_start_ms: int,
    t_end_ms: int,
    dt_sample_sec: float = 0.1,
) -> dict:
    """Aggregate micro trajectory records into CREST macro link states."""
    dx_ft = x_end_ft - x_start_ft
    dt_sec = (t_end_ms - t_start_ms) / 1000.0
    area_ft_s = dx_ft * dt_sec

    # Filter trajectory samples inside space-time cell
    mask = (
        (df_trajectories["global_time"] >= t_start_ms)
        & (df_trajectories["global_time"] < t_end_ms)
        & (df_trajectories["local_y"] >= x_start_ft)
        & (df_trajectories["local_y"] < x_end_ft)
    )
    cell_pts = df_trajectories[mask]
    if len(cell_pts) == 0:
        return {"speed_mph": 0.0, "flow_vph": 0.0, "density_vpm": 0.0, "pct_observed": 0.0}

    total_time_sec = len(cell_pts) * dt_sample_sec
    total_dist_ft = (cell_pts["v_vel"] * dt_sample_sec).sum()

    # Flow (veh / hour)
    flow_vph = (total_dist_ft / area_ft_s) * 3600.0
    # Density (veh / mile)
    density_vpm = (total_time_sec / area_ft_s) * 5280.0
    # Space mean speed (mph)
    speed_mph = (total_dist_ft / total_time_sec) / 1.46667 if total_time_sec > 0 else 0.0

    return {
        "speed_mph": float(speed_mph),
        "flow_vph": float(flow_vph),
        "density_vpm": float(density_vpm),
        "pct_observed": 100.0,
    }
```

---

## Gitignore Recommendation

Inspection of [`.gitignore`](../.gitignore) confirms:
```gitignore
data/*
!data/.gitkeep
```
Because `data/*` ignores all subdirectories and files under `data/`, the raw research downloads in `data/research_external/` are **already completely ignored by git**. No edits to `.gitignore` are required or made.

---

## Development measured pilot (implemented)

A bounded NGSIM development path is implemented in [`src/research/trajectory_adapter.py`](../src/research/trajectory_adapter.py) and [`src/research/measured_runner.py`](../src/research/measured_runner.py). Run instructions and artifact layout are in [`docs/research_measured.md`](research_measured.md). Outputs land under `results/research_measured/` with raw SHA-256, `adapter_source_hash`, and explicit sample-limit / conservation caveats.

---

## Remaining Unrun Measured Experiments

Due to the identified credential and coverage blockers, the following measured empirical experiments remain unrun and are cataloged for future work:

1. **Regional Multi-Corridor State Reconstruction**:
   - Blocked by Caltrans PeMS credential approval. Full network reconstruction across California District 12 (I-5, SR-55, SR-22) cannot be evaluated until automated or bulk credentialed export is authorized.
2. **corridor-Length Shockwave Propagation (4.2 miles)**:
   - Blocked by I-24 MOTION INCEPTION portal account approval. The fetched sample verifies single-pole tracking (60 s), but 4.2-mile wave dynamics require the full multi-camera dataset.
3. **Network Route-Choice OD Estimation**:
   - Linear corridors (NGSIM, I-24) have zero route alternatives. True $Ax = y$ evaluation on measured data requires regional probe traces (e.g. connected vehicle GPS breadcrumbs across an urban network) with multiple path alternatives between origins and destinations.

---

## Verification and Reproducibility

To re-fetch and verify the measured data artifacts from scratch:

```powershell
# 1. Fetch deterministic NGSIM slice
python scripts/fetch_research_data.py --source ngsim_i80

# 2. Fetch I-24 MOTION feasibility sample
python scripts/fetch_research_data.py --source i24_sample

# 3. Verify existing downloads against manifest hashes
python scripts/fetch_research_data.py --verify-only

# 4. Run targeted test suite
python -m pytest tests/test_research_data_sources.py
```
