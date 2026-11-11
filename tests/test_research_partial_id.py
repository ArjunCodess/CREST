import numpy as np

from src.research.partial_identification import (
    aggregate_bounds,
    flow_interval_feasibility,
    observability_diagnostics,
    od_synthetic_cases,
)
from src.research.lp_core import linprog_extremum


def test_lp_analytic_box():
    lo = linprog_extremum(np.array([1.0]), lower=2.0, upper=5.0, sense='min')
    hi = linprog_extremum(np.array([1.0]), lower=2.0, upper=5.0, sense='max')
    assert lo.status == 'optimal' and lo.value == 2.0
    assert hi.status == 'optimal' and hi.value == 5.0


def test_infeasible_lp_reports_no_value():
    res = linprog_extremum(
        np.array([1.0]),
        A_ub=np.array([[1.0]]),
        b_ub=np.array([-1.0]),
        lower=0.0,
    )
    assert res.status == 'infeasible' and res.value is None


def test_shared_link_total_identifiable():
    for case in od_synthetic_cases():
        diag = observability_diagnostics(case['incidence'], np.array([True] * case['incidence'].shape[0]))
        assert diag.nullity == case['expected_nullity']
        bounds = aggregate_bounds(
            case['incidence'],
            case['count_lower'],
            case['count_upper'],
            case['weights'],
        )
        if case['expected_identifiable']:
            assert bounds['status'] == 'optimal'
            assert bounds['width'] is not None and bounds['width'] < 1e-5
        if case['name'] == 'wide_interval':
            assert bounds['status'] == 'optimal'
            assert bounds['upper'] - bounds['lower'] > case.get('expected_width_min', 10.0)


def test_rank_tolerance_stable_under_scaling():
    A = np.array([[1.0, 1.0], [2.0, 2.0]])
    d1 = observability_diagnostics(A, np.array([True, True]))
    d2 = observability_diagnostics(1e6 * A, np.array([True, True]))
    assert d1.rank_observed == d2.rank_observed == 1


def test_constraint_induced_feasibility():
    A = np.array([[1.0, 1.0]])
    feas = flow_interval_feasibility(A, np.zeros(2), np.array([5.0, 5.0]), np.array([4.0]), np.array([6.0]))
    assert feas['feasible']
