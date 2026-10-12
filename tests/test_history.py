from evalkit.history import compare, latest_baseline, save, summarise


def report(adapter="qa:a", error=0.0, consistency=1.0, cases=None, cost=None):
    return {
        "adapter": adapter,
        "runs_per_case": 3,
        "metrics": {
            "error_rate": error,
            "unflagged_error_rate": 0.0,
            "consistency": consistency,
            "fairness_agreement": None,
            "mean_latency_s": 1.0,
            "total_cost_usd": cost,
        },
        "cases": [{"id": i, "status": s} for i, s in (cases or {"c1": "ok", "c2": "ok"}).items()],
    }


def test_the_latest_baseline_for_the_same_adapter_is_used(tmp_path):
    path = tmp_path / "history.jsonl"
    save(path, summarise(report(error=0.1), "first"))
    save(path, summarise(report(adapter="qa:other"), "other model"))
    save(path, summarise(report(error=0.2), "second"))

    assert latest_baseline(path, "qa:a")["label"] == "second"
    assert latest_baseline(path, "qa:missing") is None
    assert latest_baseline(tmp_path / "none.jsonl", "qa:a") is None


def test_a_rate_that_moves_the_wrong_way_beyond_the_tolerance_is_drift():
    baseline = summarise(report(error=0.10, consistency=1.0))

    assert not compare(baseline, summarise(report(error=0.12)), tolerance=0.05).worse
    drift = compare(baseline, summarise(report(error=0.20, consistency=0.9)), tolerance=0.05)
    assert [name for name, *_, worse in drift.metrics if worse] == ["error_rate", "consistency"]


def test_latency_and_cost_are_shown_but_never_judged():
    drift = compare(summarise(report(cost=0.01)), summarise(report(cost=9.0)), tolerance=0)

    assert not drift.worse


def test_a_case_that_passed_before_and_fails_now_is_a_regression_even_if_rates_hold():
    baseline = summarise(report(cases={"c1": "ok", "c2": "flagged error"}))
    current = summarise(report(cases={"c1": "flagged error", "c2": "ok"}))

    drift = compare(baseline, current, tolerance=0)
    assert (drift.regressed, drift.fixed) == (["c1"], ["c2"])
    assert drift.worse


def test_runs_over_different_cases_compare_cases_but_not_rates():
    baseline = summarise(report(error=0.0, cases={"c1": "ok", "c2": "ok"}))
    current = summarise(report(error=0.5, cases={"c1": "ok", "c3": "flagged error"}))

    drift = compare(baseline, current, tolerance=0)
    assert not drift.comparable
    assert drift.not_compared == ["c2", "c3"]
    assert not drift.worse
