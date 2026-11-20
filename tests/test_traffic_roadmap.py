import gzip
import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pytest
from scipy import sparse

from src.research.baselines import od_lp_bounds_baseline
from src.research.conservation_constraints import assemble_conservation_polytope
from src.research.lp_core import linprog_interval
from src.research.pems import compact_rows, local_interval
from src.research import roadmap
from src.research.observations import load_pems_corridor
from src.research.roadmap import reserve_evaluation
from src.research.sparse_traffic import conditional_interval, conservation_system, robust_fit
from src.research.traffic_estimation import QueueHead, QueuePrediction
from src.research.traffic_exchange import export_suite, score_submission
from src.research.traffic_replay import Replay, outage_triage, queue_advisory
from src.research.traffic_simulator import simulate


def test_missing_od_count_differs_from_zero():
    incidence = np.array([[1., 1.], [1., 0.]])
    total, empty = np.array([100., 0.]), np.array([100., np.nan])
    args = (np.array([1., 0.]), np.full(2, 50.))
    missing = od_lp_bounds_baseline(incidence, empty, empty, *args, observed_mask=np.array([True, True]))
    zero = od_lp_bounds_baseline(incidence, total, total, *args)
    assert missing["flows_point"][0] == pytest.approx(50.)
    assert zero["flows_point"][0] < 1.
    assert missing["aggregate_bounds"]["upper"] == pytest.approx(100.)
    assert zero["aggregate_bounds"]["upper"] == pytest.approx(0.)


def test_sparse_interval_parity_and_robust_fit():
    pytest.importorskip("osqp")
    B = np.array([[-1.], [1.]])
    R = np.array([[1., 0.], [0., -1.]])
    initial = np.full(2, 20.)
    cap = np.full(2, 50.)
    limits = np.full((2, 2), 10.)
    observation = np.array([20., 21., 22.])
    mask = np.array([[True, False], [True, False], [True, False]])
    values = np.c_[observation, np.full(3, np.nan)]
    dense = assemble_conservation_polytope(B, R, .2, initial, np.zeros_like(limits), limits,
                                           obs_mask=mask, obs_values=values, obs_tolerance=.2,
                                           z_upper=10., capacity=cap)
    legacy = np.zeros(dense.n_vars)
    legacy[2::2] = .2
    dense_bounds = linprog_interval(legacy, lower=dense.lower, upper=dense.upper, A_ub=dense.A_ub, b_ub=dense.b_ub)
    graph = np.c_[B, R]
    sparse_model = conservation_system(graph, initial, cap, np.zeros((2, 3)), np.full((2, 3), 10.), .2)
    objective = np.zeros(12)
    objective[[7, 10]] = .2
    indices = np.flatnonzero(mask.ravel())
    interval = conditional_interval(sparse_model, objective, indices, observation, .2)
    assert interval.status == "optimal"
    assert [interval.lower, interval.upper] == pytest.approx([x.value for x in dense_bounds])
    inconsistent = np.array([20., 21., 200.])
    assert conditional_interval(sparse_model, objective, indices, inconsistent, .2).status == "infeasible"
    fit = robust_fit(sparse_model, np.r_[np.full(6, 20.), np.full(6, 5.)], indices, inconsistent)
    assert fit.status == "optimal"
    assert fit.balance_error < 1e-5
    assert fit.values[4] < 25.


def test_causal_queue_ignores_future_and_late_values():
    model = QueueHead([simulate(1).window, simulate(2).window])
    window = simulate(3).window
    index = 25
    origin = window.time_ms[index] + window.cadence_ms
    arrivals = window.arrival_ms.copy()
    arrivals[index] = origin + 1
    changed = window.values["speed"].copy()
    changed[index:] = 1.
    original = replace(window, arrival_ms=arrivals)
    later = replace(original, values={**window.values, "speed": changed})
    assert np.array_equal(model.predict(original, index).probability, model.predict(later, index).probability)
    assert np.array_equal(model.predict(original, index).onset_probability, model.predict(later, index).onset_probability)


def test_incremental_correction_matches_full_and_is_idempotent(tmp_path):
    pytest.importorskip("osqp")
    graph = sparse.diags([1., 1.], -1, shape=(3, 3))
    replay = Replay(tmp_path, graph, "fixture", 60., 25., history=2, lookahead=1)
    values = np.full((8, 3), 60.)
    stamps = np.arange(8) * 300_000
    original = replay.ingest(values, stamps)
    old = replay.run(original)
    replay = Replay.from_manifest(tmp_path, original)
    # Correct an early snapshot after all later events were already ingested.
    corrected = replay.correct(original, int(stamps[2]), 1, None)
    assert corrected == replay.correct(corrected, int(stamps[2]), 1, None)
    new = replay.run(corrected)
    full = replay.run(corrected, full=True)
    assert new["outputs"] == full["outputs"]
    assert new["recomputed"] == 4
    assert new["outputs"][stamps[0]] == old["outputs"][stamps[0]]
    assert new["outputs"][stamps[-1]] == old["outputs"][stamps[-1]]
    assert replay.run(corrected)["recomputed"] == 0
    assert replay.run(corrected, previous=original)["recomputed"] == 4
    assert outage_triage(np.array([[90, 90, 90], [90, np.nan, 90]]), 90, 1)["action"] == "inspect_detector"
    assert outage_triage(np.array([[90, 90, 90], [20, 30, 20]]), 90, 1)["action"] == "corroborate_queue"


def test_compact_pems_station_rows_and_ambiguous_clock():
    content = b"11/19/2026 09:40:00\n400000,349,.0734,65.9,144.835,2.2,65.9,,0,40,100\n"
    stamp, rows = compact_rows(content)
    assert stamp == datetime(2026, 11, 19, 17, 40, tzinfo=UTC)
    assert rows[0][1:4] == ["349", ".0734", "65.9"]
    with pytest.raises(ValueError, match="repeats"):
        compact_rows(content + content.splitlines(keepends=True)[1])
    with pytest.raises(ValueError, match="Ambiguous"):
        local_interval("11/01/2026 01:15:00")


def test_shadow_only_uses_arrived_snapshots_and_counts_distinct_windows(tmp_path, monkeypatch):
    root = tmp_path / "pems"
    district = root / "D4"
    district.mkdir(parents=True)
    metadata_root = root / "station-metadata"
    metadata_root.mkdir()
    stations = "".join(f'<vds id="{i}" abs_pm="{i}" type="ML" freeway_id="880" freeway_dir="N"/>'
                       for i in range(1, 9))
    metadata = f'<pems><district id="4">{stations}</district></pems>'.encode()
    topology = hashlib.sha256(metadata).hexdigest()
    (metadata_root / f"{topology}.xml.gz").write_bytes(gzip.compress(metadata))
    arrivals = (("09:00", "17:06", True), ("09:05", "17:15", False),
                ("09:10", "17:11", False))
    for local, utc, arrived in arrivals:
        header = f"11/19/2026 {local}:00"
        content = (header + "\n" + "".join(f"{i},349,.07,65,144,2,65,,0,40,100\n"
                                            for i in range(1, 9))).encode()
        digest = hashlib.sha256(content).hexdigest()
        receipt = {"feed_local_time": header, "arrival": f"2026-11-19T{utc}:00+00:00",
                   "raw_sha256": digest, "raw_file": f"{digest}.txt.gz",
                   "topology_sha256": topology if arrived else "late-topology"}
        stamp = local.replace(":", "")
        (district / f"20261119{stamp}00.json").write_text(json.dumps(receipt))
        if arrived:
            (district / receipt["raw_file"]).write_bytes(gzip.compress(content))
    (district / "20261101011500.json").write_text("obsolete")
    now = int(datetime(2026, 11, 19, 17, 12, tzinfo=UTC).timestamp() * 1000)
    window = load_pems_corridor(root, 4, 880, "N", after_ms=now - 12 * 60_000, available_by_ms=now)
    assert window.topology_sha256 == (topology,)
    assert len(window.raw_sha256) == len(window.time_ms) == 1

    monkeypatch.setattr(roadmap, "PEMS", ((4, 880, "N", "1"), (7, 10, "W", "2")))
    output = tmp_path / "reports"
    for at in (now, now + 60_000):
        cycle = roadmap.shadow_cycle(root, output, at)
        assert [site["status"] for site in cycle["sites"]] == ["current", "no_recent_feed"]
    report = roadmap.summarize_shadow(output)
    available = report["sites"]["D4-880N-1"]["2026-11-19"]
    assert available == {"statuses": {"current": 2}, "distinct_current_windows": 1, "capture_lag_p95_ms": 60_000}
    assert len(list((output / "incidents" / "D4-880N-1").glob("*.json"))) == 1


def test_cohort_readiness_checks_outage_targets_before_freezing(monkeypatch, tmp_path):
    historical = simulate(7).window
    failed = np.ones_like(historical.quality)
    # Leave plenty of observations outside each scheduled outage target.
    for name, mask in roadmap.benchmark_masks(failed.shape, 12):
        if name.startswith("block"):
            failed[mask] = 0.
    transfer = replace(historical, site="held-out", quality=failed,
                       cell_ids=tuple(f"held-out-{i}" for i in range(len(historical.cell_ids))))
    monkeypatch.setattr(roadmap, "load_pems_corridor",
                        lambda root, district, *args, **kwargs: historical if district == 4 else transfer)
    train = {"district": 4, "freeway": 880, "direction": "S", "first_station": "1", "days": ["2026-01-01"]}
    calibration = {**train, "days": ["2026-01-02"]}
    evaluation = {**train, "district": 7, "days": [f"2026-01-{day:02}" for day in range(3, 13)]}
    config = {"cohorts": {"train": [train], "calibration": [calibration], "evaluation": [evaluation]},
              "flow_noninferiority_veh_per_hour": 1., "random_mask_noninferiority_kmh": 1.}
    _, report = roadmap.inspect_cohorts(config, tmp_path)
    assert not report["ready"]
    assert report["ready_evaluation_days"] == 0
    assert report["cohort_days"][-1]["status"] == "insufficient_mask_truth"
    assert "speed:12:contiguous" in report["cohort_days"][-1]["unscorable_masks"]

    transfer = replace(transfer, quality=historical.quality)
    cohorts, report = roadmap.inspect_cohorts(config, tmp_path)
    assert report["ready"]
    assert len(cohorts["evaluation"]) == report["ready_evaluation_days"] == 10
def test_junction_truth_balances_storage():
    sim = simulate(7, "junction")
    assert len(sim.boundary_channels) == 4
    assert np.max(np.abs(np.diff(sim.storage, axis=0) - (sim.flux @ sim.incidence.T) / 120)) < 1e-10
    assert np.all((sim.storage >= 0) & (sim.storage <= sim.capacity))


def test_benchmark_export_masks_targets_and_counts_abstentions(tmp_path):
    suite = tmp_path / "suite"
    manifest = export_suite(suite, {"evaluation": (3,)})
    scenario = manifest["cases"][0]["id"]
    with np.load(suite / "inputs" / f"{scenario}.npz") as data:
        assert "target" not in data
        assert np.isnan(data["observed"][~data["visible"]]).all()
        point = data["observed"].copy()
    prediction = tmp_path / "prediction.npz"
    np.savez(prediction, point=point)
    speed = score_submission(suite, scenario, prediction)["channels"]["speed_kmh"]
    assert speed["abstained"] == speed["eligible"] > 0
    assert speed["point_scored"] == 0 and speed["point_mae"] is None


def test_final_reservation_cannot_reuse_a_partial_cohort(tmp_path):
    reserved = reserve_evaluation(tmp_path, {("station-1", "2026-11-20"), ("station-2", "2026-11-20")}, {})
    assert (reserved / "started.json").exists()
    with pytest.raises(ValueError, match="already reserved"):
        reserve_evaluation(tmp_path, {("station-2", "2026-11-20"), ("station-2", "2026-11-21")}, {})
    assert reserve_evaluation(tmp_path, {("station-2", "2026-11-21")}, {}) != reserved


def test_queue_advisory_keeps_disjoint_road_segments_and_expires():
    prediction = QueuePrediction(0, 2, np.array([.9, .1, .8, .7]), np.zeros(4), np.ones(4, dtype=bool),
                                 np.array([True, False, False, True]))
    args = (prediction, np.arange(4) * 100.)
    fresh = queue_advisory(*args, now_ms=1000, cadence_ms=15_000, direction="E")
    assert fresh.queue_intervals_m == ((0., 100.), (200., 400.))
    assert fresh.tail_interval_m == (0., 100.)
    expired = queue_advisory(*args, now_ms=30_000, cadence_ms=15_000, direction="E")
    assert not expired.available and not expired.queue_intervals_m
