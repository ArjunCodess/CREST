# Running the CREST roadmap

Use the `roadmap` package extra for the new solver and the Windows time-zone database. The competition replay and its checkpoint requirements are documented separately in the [reproduction guide](reproducibility.md).

```powershell
python -m pip install -e ".[roadmap]"
python -m src.research.roadmap fixture
python -m src.research.roadmap export-benchmark
```

The offline CPU fixture writes `artifacts/crest/fixture.json`. It includes masked speed and flow comparisons, chain-to-junction transfer, synthetic storage estimates and conditional LP bounds, a causal queue episode, a route-identifiability test, and a check that incremental replay matches full recomputation. All fixture truth comes from simulation. The route test masks the same counters for both point estimators, and the historical v4 artifacts still record the original invalid comparator.

## Measured data and shadow operation

```powershell
python -m src.research.pems --poll-seconds 300
python -m src.research.ngsim_recordings
python -m src.research.roadmap measured
python -m src.research.roadmap shadow --poll-seconds 300
```

Run the polling commands in separate processes. PeMS files are stored under `data/crest/pems` as content-addressed snapshots with arrival times, raw hashes, and a daily metadata version. The current compact station rows differ from the generic PeMS export. The parser checks the observed compact layout and reads station count, occupancy, speed, and reported percentage observed. The development benchmark scores only fully observed detector readings. Keep the raw data local while redistribution and compact feed semantics are being confirmed. [Caltrans](https://pems.dot.ca.gov/)

The FHWA downloader retrieves all rows of each selected I-80 and US-101 recording using seven fields needed for aggregation and probe replay. It verifies row counts by minute and writes a resumable checksum manifest. The measured report gives tracked crossings and vehicle-storage accounting separately from detector-only PeMS results. Its queue references are space-mean-speed proxies with a declared threshold and adjacency requirement. Neither source independently identifies network path flows. [FHWA NGSIM](https://catalog.data.gov/dataset/next-generation-simulation-ngsim-vehicle-trajectories-and-supporting-data)

The shadow process writes `artifacts/crest/shadow.json` and per-window JSON records under `incidents/`. It only consumes observations available by the report time. Stale feeds produce an unavailable response. The measured command also writes `measured-development.json` with two outage replays and, when both recording archives are complete, cross-site development scores and a sampled-trajectory advisory check. The advisory projects the probe's path from its position and speed at issuance. Its later sampled positions are used for scoring. [Packaged development receipt](../research_manuscript/generated/roadmap_development.json)

## Incremental replay and benchmark exchange

Take the first site's `baseline_manifest` and an outage `time_ms` from the measured report. This command lets a reviewer remove a detector value and see the affected windows, source partition hashes, reconstructed values and recomputation count.

```powershell
python -m src.research.roadmap replay --site-index 0 --manifest MANIFEST --correct TIME_MS 3 missing
```

Replay caches depend on both the graph estimator's lookback and optional lookahead. Repeating the same correction produces the same manifest. The measured report compares the corrected output to a full run. Its latency is local to the machine and window count. Use representative larger logs before claiming a speedup.

The benchmark export writes `inputs/`, `scorer-only/`, and a checksummed manifest under `artifacts/crest/benchmark`. Share the input files with participants and keep scoring targets on the evaluator side for a blind test. Input arrays use channel, time and cell order. `observed` contains NaNs for masked inputs and `visible` carries the measurement mask. The scorer accepts a shape-matched NPZ with `point`, optional `lower` and `upper` arrays, and an optional boolean `fallback` mask. NaN represents a withheld estimate or interval. Error is reported alongside eligibility, abstention, fallback counts and interval coverage to keep denominators visible.

```powershell
python -m src.research.roadmap score-benchmark --scenario evaluation-241-0 --prediction predictions.npz
```

## Freezing the ITSC experiment

The final runner requires disjoint train, calibration and evaluation dates, with every evaluated station held out of development and at least ten independent evaluation days. Each cohort entry supplies `district`, `freeway`, `direction`, `first_station` and a `days` array of Pacific calendar dates. The config also requires development-derived `flow_noninferiority_veh_per_hour` and `random_mask_noninferiority_kmh` margins. Set the margins and dates before inspecting final outcomes.

```powershell
python -m src.research.roadmap freeze --config PROTOCOL.json --receipt artifacts/crest/pre-evaluation.json
python -m src.research.roadmap final --config PROTOCOL.json --receipt artifacts/crest/pre-evaluation.json
```

The freeze records code, configuration, topology and source hashes. Final execution checks them and reserves each evaluation station-day in the local registry, so an overlapping corridor or a subset cannot be used to repeat a consumed cohort. It compares 30- and 60-minute masks with an isolated-dropout control and reports paired cluster intervals against the comparator selected on calibration data. The trial does not start with insufficient measured coverage.

## Submission evidence

| Venue | Available evidence | Remaining gate |
| --- | --- | --- |
| ITE | Scored injected outage, separate congestion window, and a draft abstract. | Author approval and portal submission. |
| ACM SIGMOD demo | Interactive CLI correction, versioned partitions, and exact parity with full replay. | Repeat the demo on representative longer traces and verify the intellectual and archival overlap rules. |
| ITS World Congress | Live-input shadow command and a two-corridor retrospective replay. | Complete the independent multi-period shadow pilot before a Technical Paper. |
| IEEE IV workshop | Causal advisory messages with expiry, availability and location limits, plus trajectory and simulated-probe reports. | A fitting workshop call and evidence of useful warning performance under its criteria. |
| IEEE ITSC | Sparse point estimator, conditional LP, baselines and freeze checks. | Collect the held-out cohort and clear the frozen measured gates. |
| ACM KDD | Synthetic benchmark export, target-isolated scorer and common-mask comparisons. | Confirm Cycle 2, find a redistributable measured release and establish an independently reproducible benchmark finding. |

[Architecture and submission plan](crest-roadmap-2027.md) · [ITE abstract draft](crest-ite-abstract.md)
