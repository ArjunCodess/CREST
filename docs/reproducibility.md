# Reproducibility

Run commands from the CREST project directory. This source release supports original-model experiments and replay of restored final checkpoints. Replaying weights is not retraining, and building a PDF from saved tables is not rerunning its experiments.

## Environment

The recorded replay environment used Python 3.14, four CPU inference threads, and an RTX 4060 with 8 GB VRAM. Install `requirements-v8.lock.txt` for recorded versions or `requirements.txt` for a less constrained development installation. Choose a CUDA-compatible PyTorch build for GPU execution. Package resolution on other Python/platform combinations has not been established by the historical replay.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-v8.lock.txt
python -m pytest
```

## Input acquisition and restoration

Obtain the official participant release through Kaggle under the benchmark's rules. Supply its root explicitly with `--release-root`; it contains `corridors/`, `submission_key.csv`, and `sample_submission.csv`. Raw observations are not redistributed here.

The source checkout does not contain fitted checkpoints. The final organizer delivery contains `models/` subdirectories `state`, `queue`, `events`, `onset`, `enhanced`, `ongoing`, `fd_decisions`, and `wave_v7`, as well as the external assets described in [../THIRD_PARTY.md](../THIRD_PARTY.md). Restore that package separately, or restore its matching inputs into a copy of this source tree. Do not substitute arbitrary retrained weights and claim an exact replay.

The historical organizer archive SHA-256 is `37ad5f2b02fc62791a83dbc0cb8b75cd265454cbfcdd6cd4d2317ac7259c0d72`. It was distributed privately; this repository does not provide a public download for the original fitted checkpoints. Public external assets alone are insufficient to reproduce the original component. A fresh clone without restored inputs can run tests and fixture experiments but cannot reproduce the final leaderboard entry.

## Final checkpoint replay

```powershell
python scripts/reproduce_v8_submission.py --release-root PATH_TO_PUBLIC_RELEASE --package-root PATH_TO_RESTORED_PACKAGE --output artifacts/replayed --device GPU
```

The package root supplies replay scripts, source modules, `models/`, `starter/`, and `external/`. When using an unchanged organizer package, the command runs the scripts shipped in that package. To exercise current source changes, restore matching models and external assets into this checkout and pass `--package-root .`.

Use a fresh output directory. Add `--variant cat` for the fallback. Add `--reference PATH_TO_REFERENCE.csv` for a comparison after prediction; the reference is not an input to model inference. The full hybrid uses three U-Net2 seeds, 42, 7, and 123. Model manifests preserve inference threading choices because floating-point reduction order can alter tree-threshold decisions. CPU and GPU results can differ numerically.

The historical independent replay compared 6,980,503 rows. Maximum state difference was 5.46e-12, and queues and path flows matched exactly. The comparison used numerical tolerances; differing CSV byte hashes are expected. No fresh training or leaderboard rescore follows from that replay.

The preserved [checkpoint replay receipt](../results/checkpoint_replay.json) contains the historical hashes, component checks, and numerical comparisons with local drive paths removed. Run `scripts/validate_v8_prereqs.py --release-root PATH_TO_PUBLIC_RELEASE --package-root PATH_TO_RESTORED_PACKAGE` before a replay. The PowerShell wrapper `replay.ps1 -CheckOnly` performs the same prerequisite gate for restored assets in this checkout.

## Original-model training

```powershell
python -m src.pipeline inspect --release-root PATH_TO_PUBLIC_RELEASE --output artifacts/original
python -m src.pipeline train --release-root PATH_TO_PUBLIC_RELEASE --output artifacts/original --epochs 1 --rounds 80 --full-train
python -m src.pipeline predict --release-root PATH_TO_PUBLIC_RELEASE --model-root artifacts/original --output artifacts/original_prediction --split both
```

These commands train and predict with the base original pipeline. They do not train every later residual, wave, onset, and ongoing refinement or external model. The reusable refinement/campaign scripts retain their own CLI settings; consult `--help` before running them. Reconstructing a full development campaign requires its recorded model-selection inputs and artifacts. The selected ensemble's historical score is not a guarantee for this base training recipe.

Reusable experiment drivers include `pilot_state_v4.py` and `refit_state_v4.py` for cross-sensor residuals, `pilot_onset.py` and `refit_onset_v4.py` for conditional onset, `pilot_queue_v5.py` and `refit_queue_v5.py` for ongoing localization, and `pilot_v7.py`, `audit_state_v7.py`, and `export_state_v7.py` for wave-aware state experiments. They keep explicit release/output arguments and promotion evidence. Historical staging, email, archive-delivery, and browser-download scripts are not part of the maintained source.

`run.ps1` coordinates base training and completion; `resume.ps1` can acquire the essential release when authenticated. Completion writes experiment reports inside the chosen output directory and compiles the maintained manuscript without replacing its text or frozen results.

## Synthetic integration run

```powershell
python scripts/make_fixture.py --output artifacts/fixture/release
python -m src.pipeline inspect --release-root artifacts/fixture/release --output artifacts/fixture/run
python -m src.pipeline train --release-root artifacts/fixture/release --output artifacts/fixture/run --epochs 1 --rounds 5
python -m src.pipeline predict --release-root artifacts/fixture/release --model-root artifacts/fixture/run --output artifacts/fixture/predictions --split both
```

Fixtures are marked explicitly. Their outputs are software checks, not scientific evidence, and must never be uploaded as benchmark predictions.

## Manuscript and figures

```powershell
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error working_notes.tex
```

IEEEtran, BibTeX, and the referenced figures/tables are required. Saved figure and table inputs are included for document builds. To regenerate plots from measurements, set `TRAFFICFLOW_ARTIFACTS_OFFICIAL` to an original experiment directory containing `*_validation_states.npz` and `temporal_cv.csv`, and `TRAFFICFLOW_ARTIFACTS_CONTENDER` to the corresponding refinement artifacts, then run `python paper/make_figures.py` and `python paper/make_tables.py` from the project root. Missing artifacts must fail rather than invent results.

## Scientific boundaries

Released inputs do not include official queue labels, all conservation boundary flows, or hidden path-flow truth. Local metrics therefore cover reconstruction and explicitly labeled proxies. Previously inspected development blocks and the synthetic benchmark do not establish independent generalization. Public scores are historical receipts; private results remain unknown.
