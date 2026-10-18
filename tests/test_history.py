from evalkit.history import compare, git_commit, latest_baseline, save, summarise
from evalkit.report import drift_text


def report(adapter="qa:a", error=0.0, consistency=1.0, cases=None, cost=None, models=None):
    return {
        "adapter": adapter,
        "models": models or ["model-a"],
        "git_commit": "abc123",
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


def test_a_baseline_records_which_models_and_commit_produced_it():
    entry = summarise(report(models=["model-a"]), "first")

    assert (entry["models"], entry["git_commit"]) == (["model-a"], "abc123")


def test_a_model_change_is_shown_but_not_judged_on_its_own():
    baseline = summarise(report(models=["model-a"]))
    drift = compare(baseline, summarise(report(models=["model-b"])), tolerance=0)

    assert drift.models_changed == (["model-a"], ["model-b"])
    assert not drift.worse
    assert "The model changed: model-a -> model-b" in drift_text(drift, baseline)


def test_a_baseline_saved_before_models_were_recorded_is_still_compared():
    old = summarise(report())
    del old["models"]

    assert compare(old, summarise(report()), tolerance=0).models_changed is None


def test_the_git_commit_is_none_outside_a_repository(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert git_commit() is None
