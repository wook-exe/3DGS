import json

from src.ab_testing import assign_variant, assignments_for_user, track_event


def test_ab_assignment_is_stable_for_same_user():
    first = assign_variant("dashboard_chart_density", "student-001")
    second = assign_variant("dashboard_chart_density", "student-001")

    assert first == second


def test_ab_assignments_include_two_experiments():
    assignments = assignments_for_user("student-001")

    assert set(assignments) == {"dashboard_chart_density", "model_status_copy"}


def test_track_event_appends_jsonl(tmp_path):
    log_path = tmp_path / "events.jsonl"

    event = track_event(
        event_name="dashboard_loaded",
        user_id="student-001",
        experiment_key="dashboard_chart_density",
        variant="compact",
        properties={"source": "test"},
        log_path=log_path,
    )

    lines = log_path.read_text(encoding="utf-8").splitlines()
    saved = json.loads(lines[0])
    assert saved["event_name"] == "dashboard_loaded"
    assert saved["variant"] == "compact"
    assert event["properties"] == {"source": "test"}
