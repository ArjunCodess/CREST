# Running the CREST roadmap

The evaluation report gives the comparison results, denominators, and limits. Use this guide to rerun the experiments and trace individual outputs. [Evaluation](crest-evaluation.md)

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
python -m src.research.roadmap shadow-report
```

Run the polling commands in separate processes. PeMS files are stored under `data/crest/pems` as content-addressed snapshots with arrival times, raw hashes, and a daily metadata version. The current compact station rows differ from the generic PeMS export. The parser checks the observed compact layout and reads station count, occupancy, speed, and reported percentage observed. The development benchmark scores only fully observed detector readings. Keep the raw data local while redistribution and compact feed semantics are being confirmed. [Caltrans](https://pems.dot.ca.gov/)

The FHWA downloader retrieves all rows of each selected I-80 and US-101 recording using seven fields needed for aggregation and probe replay. It verifies row counts by minute and writes a resumable checksum manifest. The measured report gives tracked crossings and vehicle-storage accounting separately from detector-only PeMS results. Its queue references are space-mean-speed proxies with a declared threshold and adjacency requirement. Neither source independently identifies network path flows. [FHWA NGSIM](https://catalog.data.gov/dataset/next-generation-simulation-ngsim-vehicle-trajectories-and-supporting-data)

The shadow process writes `artifacts/crest/shadow.json`, per-window decisions under `incidents/`, and one record per poll under `shadow-history/`. Snapshots and their topology enter a decision only after the measurement window has closed and the collector has received the file. The process records missing feeds, stale feeds, and arrived files with no usable speed readings separately, so downtime stays visible in the pilot. `shadow-report` groups those outcomes by corridor and Pacific day, counts distinct current windows, and reports the 95th-percentile lag from window close to file capture for those windows.

The measured command writes `measured-development.json` with two outage replays and, when both recording archives are complete, cross-site development scores and a sampled-trajectory advisory check. The advisory projects the probe's path from its position and speed at issuance. Its later sampled positions are used for scoring. [Packaged development receipt](../research_manuscript/generated/roadmap_development.json)

Run the extended development comparisons with the same local archive.

```powershell
python -m src.research.roadmap --output artifacts/crest/roadmap-audit operational-validation --available-by-ms 1795168500000
python -m src.research.roadmap --output artifacts/crest/roadmap-audit ablate
python -m src.research.roadmap --output artifacts/crest/roadmap-audit demand-transfer
python -m src.research.roadmap --output artifacts/crest/roadmap-audit advisory-sweep
```

`operational-validation.json` records a fixed arrival cutoff, source digests, 30- and 60-minute detector blackouts, 15-minute added delays, decision and observation recovery, and a time-blocked detector-proxy queue comparison. The replay scores graph reconstruction and latest-reading persistence on the same observed targets. Recoveries outside the observation window are censored. The replay also measures the effect of late edits on longer measured and synthetic traces. Use `replay --log-dir` to reopen a run from its manifest in `long-event-log`.

The November 20 rerun scored 58 injected fault cases. Incremental replay matched full recomputation for both lookahead settings on the 240- and 241-window measured traces and the 576-window synthetic trace. [Expanded operational validation](../results/crest-roadmap/operational-validation-expanded.json)

`ablations.json` compares matched storage priors with and without balance, narrower inlet and outlet bounds, and hard instead of soft observation handling. It also tests lagged neighbor features at the same tree budget and pairs each synthetic blackout with a random mask of exactly the same size. `demand-transfer.json` evaluates light and surge junction traffic using priors and calibration fitted on nominal-demand chains. `advisory-sweep.json` tests delivery delay and probe penetration on disjoint five-minute blocks from the existing US-101 development recording, with I-80 used for training. These blocks share one evaluation recording.

The two offline transfer commands use the archived data and require no shadow process.

```powershell
python -m src.research.roadmap detector-transfer --available-by-ms 1795168500000
python -m src.research.roadmap trajectory-transfer
```

The detector transfer tests 30- and 60-minute masks in both corridor directions. It trains and calibrates on separate blocks of the first Pacific day, with a 30-minute buffer, and evaluates the other corridor's later archived windows. The trajectory transfer tests 4- and 8-minute masks on complete destination recordings. Because the US-101 archive has one contiguous recording, that training direction uses a six-minute gap between its training and calibration blocks. Both commands are development runs. [Detector transfer](../results/crest-roadmap/detector-transfer.json) · [Trajectory transfer](../results/crest-roadmap/trajectory-transfer.json)

The trajectory report counts one alarm per warning episode and reports detected and missed onsets, false alarms and lead time against the speed-derived queue proxy. US-101 had 11 eligible cell onsets. The queue head detected zero, and the causal graph trend detected two. The two I-80 recordings had zero eligible onsets. [Queue event results](../results/crest-roadmap/trajectory-transfer.json)

## Incremental replay and benchmark exchange

Take the first site's `baseline_manifest` and an outage `time_ms` from the measured report. This command lets a reviewer remove a detector value and see the affected windows, source partition hashes, reconstructed values and recomputation count.

```powershell
python -m src.research.roadmap replay --site-index 0 --manifest MANIFEST --correct TIME_MS 3 missing
python -m src.research.roadmap replay --site-index 0 --manifest EDITED_MANIFEST --compare BASE_MANIFEST
```

Replay caches depend on both the graph estimator's lookback and optional lookahead. Repeating the same correction produces the same manifest. The measured report compares the corrected output to a full run. Its latency is local to the machine and window count. Use representative larger logs before claiming a speedup.

The benchmark export writes `inputs/`, `scorer-only/`, and a checksummed manifest under `artifacts/crest/benchmark`. Share the input files with participants and keep scoring targets on the evaluator side for a blind test. Input arrays use channel, time and cell order. `observed` contains NaNs for masked inputs and `visible` carries the measurement mask. The scorer accepts a shape-matched NPZ with `point`, optional `lower` and `upper` arrays, and an optional boolean `fallback` mask. NaN represents a withheld estimate or interval. Error is reported alongside eligibility, abstention, fallback counts and interval coverage. An optional `confidence` array gives matched-coverage MAE at 50%, 80% and 100% of eligible targets, highest confidence first. The scorer uses array order for ties and leaves MAE unset when a method supplies too few estimates.

```powershell
python -m src.research.roadmap score-benchmark --scenario evaluation-241-0 --prediction predictions.npz
python -m src.research.roadmap verify-benchmark
python -m src.research.roadmap export-benchmark --source ngsim
python -m src.research.roadmap verify-benchmark --source ngsim
```

The NGSIM export uses complete recordings as split units and publishes derived, one-minute observations without vehicle identifiers. It carries FHWA attribution, the source catalog link, the transformations, and the catalog's CC BY-SA 3.0 license. The measured export omits capacity values because the source does not supply them. Add `--source ngsim` when scoring one of its cases. The verifier checks each participant-input checksum and visibility array and evaluates historical median, persistence and interpolation, with an observed-only submission as an abstention check. The source recordings and synthetic seeds are public, so a blind evaluation requires a fresh evaluator-controlled cohort. [FHWA source and license](https://data.transportation.gov/api/views/8ect-6jqj.json)

## Freezing the ITSC experiment

The final runner requires disjoint train, calibration and evaluation dates, with every evaluated station held out of development and at least ten independent evaluation days. Each cohort entry supplies `district`, `freeway`, `direction`, `first_station` and a `days` array of Pacific calendar dates. The config also requires development-derived `flow_noninferiority_veh_per_hour` and `random_mask_noninferiority_kmh` margins. Set the margins and dates before inspecting final outcomes.

```powershell
python -m src.research.roadmap readiness --config PROTOCOL.json
python -m src.research.roadmap freeze --config PROTOCOL.json --receipt artifacts/crest/pre-evaluation.json
python -m src.research.roadmap final --config PROTOCOL.json --receipt artifacts/crest/pre-evaluation.json
```

Readiness checks completed days using observation availability and the same outage masks as the scorer. For each scheduled duration and channel, an evaluation corridor-day needs at least one fully observed block to hide and some observed targets for the random mask. The report identifies days still in progress, days absent from the archive, and days with insufficient data. It leaves the run registry untouched.

The freeze records code, configuration, topology and source hashes. Final execution checks them and reserves each evaluation station-day in the local registry, so an overlapping corridor or a subset cannot be used to repeat a consumed cohort. It compares 30- and 60-minute masks with a random-dropout control and reports paired day-level intervals against the comparator selected on calibration data. Each comparison needs ten paired evaluation days to pass its gate.

The November 20 readiness check found zero of ten eligible evaluation days for the candidate dates. The proposed training date also had no archived snapshots. The check used the two development corridors to diagnose archive coverage. A final protocol still needs an untouched evaluation corridor, usable training and calibration dates, and margins derived from development data. The synthetic receipt verifies the freeze and scoring workflow. A learned graph-recurrent imputer is included in the development comparison. The frozen measured run and an independently held trajectory reserve remain open. [Readiness check](../results/crest-roadmap/itsc-readiness.json) · [Synthetic workflow check](../results/crest-roadmap/synthetic-freeze-check.json)

The trajectory fallback uses a separate archive of independent recordings in the NGSIM adapter format. Each role lists objects with a `site` slug and a zero-based `recording` index. Set `minimum_evaluation_recordings` before scoring, using independent recording groups from the provider. Evaluation sites must be outside the development sites. The freeze records the protocol, code and source digests before scoring and the final runner reserves the source partitions so they can be used only once. The primary gate is paired recording-level speed MAE under both 4- and 8-minute outages, with a 95% interval against the calibration-selected baseline. Flow, random masks and coverage are secondary. The synthetic check scored two recording groups and verified that the registry rejects a repeat. [Trajectory freeze check](../results/crest-roadmap/trajectory-freeze-check.json)

```powershell
python -m src.research.roadmap --ngsim ARCHIVE freeze --source trajectory --config PROTOCOL.json --receipt trajectory-freeze.json
python -m src.research.roadmap --ngsim ARCHIVE final --source trajectory --config PROTOCOL.json --receipt trajectory-freeze.json
```

The [roadmap results](../results/crest-roadmap/) include the development and stress-test receipts, the shadow summary, and both participant benchmark archives. The participant archives contain masked inputs, manifests and, for NGSIM, attribution. The local archive retains raw PeMS snapshots and evaluator-only targets.

## Submission evidence

| Venue | Available evidence | Remaining gate |
| --- | --- | --- |
| ITE | Scored injected outage and separate congestion case, with the larger retrospective replay available for context. | Author approval and portal submission. |
| ACM SIGMOD demo | Incremental replay matched full recomputation on 240- and 241-window PeMS traces and a 576-window synthetic trace. | Verify the archival overlap rules. General speedup claims need larger traces and repeated timings. |
| ITS World Congress | Archived replay of 58 injected faults. Graph reconstruction scored 4.65 km/h MAE on 468 target instances versus 5.86 for persistence. Both methods missed all ten queue-proxy positives. | The poster describes retrospective replay. Claims about operator decisions need independent incident labels. |
| IEEE IV workshop | One US-101 recording supplied nine blocks. The gate detected 9 of 235 proxy encounters at a 7.5-second delay. | A fitting workshop call and useful recall on independent recordings. |
| IEEE ITSC | Persistence had lower speed and flow error in both archived PeMS directions. NGSIM speed results changed with the direction of transfer. | The candidate schedule supplied zero of ten eligible PeMS days. Final evaluation needs untouched stations or an independent trajectory archive under its own protocol. |
| ACM KDD | Matched-coverage scores on synthetic and NGSIM exports, plus a persistence/tree ranking reversal under equal-size random and contiguous masks. | Confirm Cycle 2 and evaluate generality on evaluator-controlled cohorts. |

[Architecture and submission plan](crest-roadmap-2027.md) · [ITE abstract draft](crest-ite-abstract.md)
