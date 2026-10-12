import pytest

from evalkit.faults import Fault, drop_flags, force_decision, scale_field
from evalkit.oversight import answer_key, build_plan, queue_rows, read_decisions, score_decisions
from tests.helpers import FixedAdapter, case, proposal

CASES = [
    case("a", decision="approve"),
    case("b", decision="decline", covers=("R2",)),
    case("c", decision="refer"),
]
CLEAN = {"a": proposal("approve"), "b": proposal("decline"), "c": proposal("refer", flags=("x",))}
FAULTS = [
    Fault("approves", ("R1", "R2"), "", force_decision("approve")),
    Fault("hides-flags", ("R1",), "", drop_flags()),
    Fault("scales", ("R1",), "", scale_field("income", 2)),
]


def make_plan(seed=0):
    return build_plan(lambda: FixedAdapter(CLEAN), CASES, FAULTS, seed)


def test_only_faults_that_change_the_decision_become_seeds():
    plan = make_plan()

    seeded = sorted((i.case_id, i.fault_id) for i in plan.items if i.kind == "seeded")
    assert seeded == [("b", "approves"), ("c", "approves")]
    assert plan.unused_faults == ["hides-flags", "scales"]


def test_every_correct_proposal_becomes_a_control():
    controls = sorted(i.case_id for i in make_plan().items if i.kind == "control")

    assert controls == ["a", "b", "c"]


def test_the_queue_order_is_shuffled_but_repeatable():
    order = [i.case_id for i in make_plan(seed=1).items]

    assert order == [i.case_id for i in make_plan(seed=1).items]
    assert [i.item_id for i in make_plan().items] == [f"item-{n:03d}" for n in range(1, 6)]


def test_the_queue_never_reveals_which_items_are_seeded():
    rows = queue_rows(make_plan(), CASES)

    assert {"kind", "fault_id", "correct_decision", "case_id"}.isdisjoint(rows[0])
    assert rows[0]["documents"] == [{"name": "doc.txt", "text": "text"}]


KEY = [
    {"item_id": "s1", "kind": "seeded", "fault_id": "approves", "proposed_decision": "approve"},
    {"item_id": "s2", "kind": "seeded", "fault_id": "approves", "proposed_decision": "approve"},
    {"item_id": "s3", "kind": "seeded", "fault_id": "declines", "proposed_decision": "decline"},
    {"item_id": "c1", "kind": "control", "fault_id": None, "proposed_decision": "approve"},
    {"item_id": "c2", "kind": "control", "fault_id": None, "proposed_decision": "refer"},
]


def test_the_team_score_counts_catches_wrong_approvals_and_false_alarms():
    decisions = [("s1", "refer"), ("s2", "approve"), ("s3", "approve"), ("c1", "approve"), ("c1", "decline")]

    score = score_decisions(KEY, decisions)

    assert score.catch_rate == pytest.approx(2 / 3)
    assert score.wrong_approval_rate == 0.5
    assert score.false_alarm_rate == 0.5
    assert score.by_fault == {"approves": (1, 2), "declines": (1, 1)}
    assert score.unreviewed == ["c2"]


def test_decisions_are_read_and_checked(tmp_path):
    path = tmp_path / "decisions.csv"
    path.write_text("item_id,decision\ns1, Refer \nc1,approve\n", encoding="utf-8")

    assert read_decisions(path, KEY) == [("s1", "refer"), ("c1", "approve")]


def test_bad_decision_rows_are_all_reported(tmp_path):
    path = tmp_path / "decisions.csv"
    path.write_text("item_id,decision\nzz,approve\ns1,maybe\n", encoding="utf-8")

    with pytest.raises(ValueError) as err:
        read_decisions(path, KEY)
    assert "line 2: unknown item 'zz'" in str(err.value)
    assert "line 3: decision must be one of approve, decline, refer, got 'maybe'" in str(err.value)


def test_a_reviewer_column_is_refused(tmp_path):
    path = tmp_path / "decisions.csv"
    path.write_text("item_id,decision,reviewer\ns1,refer,anna\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected the header 'item_id,decision'"):
        read_decisions(path, KEY)


def test_the_answer_key_keeps_what_the_queue_hides():
    key = answer_key(make_plan())

    assert {k["kind"] for k in key} == {"seeded", "control"}
    assert all(k["correct_decision"] for k in key)
