# Research evaluation protocol

## Development vs final

| Aspect | Development (`dev_pilot.json`) | Final (`final_protocol.json`) |
|--------|--------------------------------|--------------------------------|
| Default CLI | Yes (`--mode development`) | Requires `--mode final` |
| Seeds | 11, 22, 33, 44, 55 | **v3 active** (10 seeds; manifest on disk) |
| Network size | 5 links, 8 steps | 7 links, 8 steps (chain) |
| Registry + receipt | Registry optional | Ledger + `final_receipt.json` in `--output` |

## Holdout v1 exposure (retired)

The original final seeds **101, 202, 303, 404, 505** were executed during an integration review while probing runner mode/config bugs (development mode, 40 runs). They are **not pristine**. Evidence is preserved in:

- `configs/research/final_protocol_v1_retired.json`
- `final_receipt.json` field `exposure_record` when a v2 final run completes
- Protocol registry entry `final_holdout_v1` (`status: retired_exposed`)

Do **not** run v1 seeds again. Unit tests and development configs must remain disjoint from retired v1/v2 final seeds.

## Holdout v2 exposure (retired, not replaced)

Seeds **701, 802, 903, 1004, 1105** were reserved as holdout v2 after v1 retirement. On **2026-11-11** an independent review executed `run_one_scenario` on at least **701** and **802** with `final_protocol.json` scenario parameters despite instructions not to run v2. **All v2 seeds are retired**; exposure subset and configuration are recorded in `configs/research/final_protocol.json` (`exposure_record`).

**v3 archive (2026-11-11):** Seeds in `configs/research/final_holdout_v3_seed_manifest.json` and `configs/research/final_protocol.json` are **retired** (`holdout_status: retired_failed`). The single frozen attempt failed during legacy generative capacity checks (receipt preserved). **No v3 rerun.**

**v4 amended reserve (2026-11-11):** New seeds in `configs/research/final_holdout_v4_seed_manifest.json` (written before execution). Active bundled/`final_protocol_v4.json` config; constants `FINAL_HOLDOUT_V4_ACTIVE_SEEDS` in `config_validation.py`. Stress variants and `scenario_params` match the archived v3 declaration. Single final execution → `results/research_final_v4/`; pre-freeze `results/research/pre_evaluation_freeze_v4.json` (v3 freeze untouched).

## Mode and config coupling

- CLI `--mode` must match `protocol_role` inside the JSON file.
- Default configs are **bundled** in the installed package (`src.research.bundled_configs`); path string differences do not change behavior.
- Running `--mode final` with a development protocol file is rejected.
- Running `--mode development` with the final protocol file is rejected by validation.

## Fingerprints and provenance

- Protocol fingerprint hashes **canonical semantic content** only (`protocol_id`, seeds, params, variants)—not filesystem paths.
- Result provenance includes `research_code_hash`: content hash of `src/research/**`, `configs/research/**`, `src/research/bundled_configs/**`, `pyproject.toml`, and `requirements.txt` (and `requirements-lock.txt` when present).
- JSON artifacts use strict finite JSON (`allow_nan=False`).

## Final run guard (registry + receipt)

Registry default: `./.crest_research_protocol_registry.json` in the current working directory, or `CREST_RESEARCH_REGISTRY`. This is **accidental-reuse protection**, not cryptographic security against deliberate file tampering.

```powershell
python -m src.research.runner --mode final --config configs/research/final_protocol_v4.json --output results/research_final_v4 --registry .crest_research_protocol_registry.json
```

A second completed run with the same protocol fingerprint is blocked even if `--output` differs, unless:

```powershell
python -m src.research.runner --mode final --override-final --override-reason "documented rerun after bugfix"
```

Interrupted runs are recorded (`in_progress` / `failed`) so a failed final attempt does not silently authorize another peek without override.

## Development output overwrite

Development runs refuse to overwrite `pilot_results.json` unless `--force` is passed.

## Embargo for future real data

The competition February evaluation week must not be labeled “fresh holdout” without organizer evidence. Synthetic pilot results do not transfer to leaderboard scores.

## Pilot selection honesty

No automatic winner is promoted from the pilot. Scenario/seed summaries are descriptive means and standard deviations only; no cell-level significance claims.
