# Detector outage evaluation

Persistence beat graph Huber reconstruction on every archived PeMS transfer comparison. The injected-fault replay found lower speed error when the smoother used a prior built from readings already received on the target corridor. The two experiments change both the prior and the evaluation windows, so their difference does not isolate the effect of the prior.

## Speed and flow transfer

The PeMS runs split the first Pacific day into training and calibration blocks separated by 30 minutes. They score later windows on the other corridor. Each method sees the same mask and scores the same available targets within a comparison. The NGSIM runs use complete destination recordings and shorter masks at one-minute cadence. The US-101 training direction uses two blocks of one recording with a six-minute gap.

| Source to destination | Outage | Scored targets per method | Graph Huber speed MAE, km/h | Calibration-selected baseline speed MAE, km/h |
| --- | ---: | ---: | ---: | ---: |
| PeMS I-880 to I-10 | 30 min | 24 | 12.13 | 3.56, persistence |
| PeMS I-880 to I-10 | 60 min | 48 | 14.14 | 4.21, persistence |
| PeMS I-10 to I-880 | 30 min | 30 | 22.41 | 2.20, persistence |
| PeMS I-10 to I-880 | 60 min | 60 | 24.35 | 2.07, persistence |
| NGSIM I-80 to US-101 | 4 min | 12 | 6.11 | 10.28, persistence |
| NGSIM I-80 to US-101 | 8 min | 24 | 7.53 | 10.43, persistence |
| NGSIM US-101 to I-80 | 4 min | 24 | 7.41 | 6.95, graph recurrent |
| NGSIM US-101 to I-80 | 8 min | 48 | 8.86 | 6.38, graph recurrent |

PeMS persistence also had lower flow MAE in all four contiguous comparisons. The graph model's nominal 90% speed intervals covered 95–100% of those targets with widths of 65–80 km/h. Its flow intervals covered zero of 162 scored target instances. Broad speed intervals and failed flow coverage make the calibration unsuitable for this transfer. Each PeMS direction has one paired evaluation group, so the runner supplies no confidence interval. Targets can recur across masks, which rules out treating pooled counts as independent observations. [PeMS transfer receipt](../results/crest-roadmap/detector-transfer.json)

The NGSIM speed finding reverses with direction. Flow also depends on direction and comparator. Graph recurrent gives lower contiguous-mask flow error on the US-101 test, while graph Huber gives lower flow error than the selected tree on the two I-80 recordings. The reverse comparison has two recording groups. Its bootstrap interval describes those two observations and gives little evidence about other sites. [NGSIM transfer receipt](../results/crest-roadmap/trajectory-transfer.json)

## What balance and the synthetic tests explain

The graph speed smoother and the sparse storage fit answer separate questions. In the US-101 storage experiment, a chain that treated tracked crossings as complete was infeasible. Allowing discrepancies up to 17 vehicles per cell and interval restored feasibility. Storage MAE moved from 6.36 vehicles for persistence to 6.25 for the bounded fit. The experiment shows how incomplete crossing accounting affects feasibility, with little evidence of a point-accuracy gain. [Measured storage receipt](../results/crest-roadmap/measured-development.json)

Synthetic bias tests made hard observation intervals infeasible in all four cases. The fallback stayed in the error denominator. Narrower boundary limits reduced mean inlet interval width from 2,454 to 326 vehicles and retained the truth in all eight feasible cases. This result depends on the simulated boundaries and the validity of the tighter limits. Likewise, one total count left the first route anywhere from 0 to 100 vehicles. Adding a branch counter identified 70 in the fixture. Independent measured route truth remains unavailable. [Ablations](../results/crest-roadmap/ablations.json) · [Route fixture](../results/crest-roadmap/fixture.json)

Changing demand did not rescue contiguous speed recovery. Graph Huber trailed the selected tree under both light and surge demand, with MAE of 10.35 versus 9.28 km/h and 9.44 versus 8.08 km/h, respectively. It had lower point error under random masking in both regimes, although the four-seed interval for surge included zero. Under contiguous loss, its speed intervals covered 66% of light-demand targets and 59% of surge targets against a nominal 90%. Mask structure and demand shift both matter for the evaluation. [Demand transfer receipt](../results/crest-roadmap/demand-transfer.json)

## Operations, queues and replay

The expanded fault replay injected 58 blackouts or delays. On 468 scored speed target instances, graph reconstruction had 4.65 km/h MAE and latest-reading persistence had 5.86 km/h. Both methods missed all ten low-speed proxy positives. A fresh reading returned inside the observation window in 28 cases. The remaining 30 recovery times are censored. [Expanded operational replay](../results/crest-roadmap/operational-validation-expanded.json)

At one-minute cadence, the queue head detected none of 11 eligible US-101 cell onsets. The causal graph trend detected two, with a 90-second median lead among those detections. Both I-80 recordings had zero eligible onsets. The separate vehicle replay used a 30-second forecast and sampled positions for scoring. At 7.5 seconds of delivery delay and 35% probe penetration, the gated advisory detected 9 of 235 proxy encounters and gave 136 false warnings on 9,639 vehicle-origin pairs. The ungated rule warned on every pair. At 30 seconds the gated message expired on every pair. These counts describe one US-101 recording. Neither experiment establishes field warning reliability. [Onset evaluation](../results/crest-roadmap/trajectory-transfer.json) · [Vehicle advisory sweep](../results/crest-roadmap/advisory-sweep.json)

After two out-of-order edits, incremental replay matched all six full recomputations across the 240- and 241-window PeMS traces and the 576-window synthetic trace. It recomputed eight outputs with three windows of history, and twelve when the estimator also looked ahead two windows. The cache saves solves while the replay still traverses the manifest. Timings come from one host and one run per configuration. [Replay receipt](../results/crest-roadmap/operational-validation-expanded.json) · [Example revision](../results/crest-roadmap/replay-comparison.json)

## Benchmark scoring and remaining evidence

On all 360 synthetic evaluation speed targets, linear interpolation had 3.96 km/h MAE and historical median had 21.84. On the 33 NGSIM evaluation targets, historical median scored 8.13 and interpolation scored 9.00. The same ordering held at the 50% and 80% per-case score thresholds. The aggregate weights each case MAE by the number of targets admitted at that threshold, rounded up within each case. An observed-only submission scored zero withheld targets, so it has no MAE at any threshold. A separate experiment reversed the persistence/tree ranking when it swapped contiguous masks for random masks of equal size. The public simulator seeds and FHWA recordings can be inspected by participants, so a blind benchmark needs evaluator-controlled cases. [Synthetic scores](../results/crest-roadmap/benchmark-baselines.json) · [NGSIM scores](../results/crest-roadmap/benchmark-ngsim-baselines.json) · [Matched-mask ablation](../results/crest-roadmap/ablations.json)

The November 20 PeMS readiness check found zero of ten eligible evaluation days for the candidate schedule. Both corridors and the three NGSIM recordings have already been examined during development. The trajectory fallback ran end to end on synthetic recordings and rejected a second use of the reserve. The frozen measured claim still depends on ten eligible untouched PeMS days or a separately frozen independent trajectory archive. The live pilot is outside this evaluation. [PeMS readiness](../results/crest-roadmap/itsc-readiness.json) · [Trajectory workflow check](../results/crest-roadmap/trajectory-freeze-check.json)
