"""간단한 규칙 엔진 회귀 테스트.

실행:
    python regression_tests.py

Streamlit이 설치되지 않은 환경에서도 핵심 규칙 함수만 확인할 수 있도록
아주 작은 streamlit 더미 객체를 넣고 app.py를 import한다.
"""
from __future__ import annotations

import sys
import types

if "streamlit" not in sys.modules:
    sys.modules["streamlit"] = types.SimpleNamespace(set_page_config=lambda **kwargs: None, secrets={})

import app  # noqa: E402


def primary_for(problem: str, goal: str) -> tuple[str, dict[str, bool]]:
    cues = app.auto_detect_cues(problem, "")
    rec = app.recommend_method(goal, cues)
    return rec.primary, cues


def test_rule_engine_examples() -> None:
    cases = [
        (
            "마찰 없는 트랙에서 높이 h부터 출발해 반지름 R인 원형 고리를 지난다. 꼭대기에서 떨어지지 않기 위한 최소 높이를 구하라.",
            "접촉 유지 조건",
            "복합 풀이",
        ),
        (
            "질량 5 kg인 물체가 마찰계수 0.2인 30도 경사면 위에 있다. 경사면 아래 방향 가속도를 구하라.",
            "가속도",
            "뉴턴 제2법칙 F=ma",
        ),
        (
            "반지름 R인 원판이 미끄러지지 않고 경사면을 굴러 내려온다. 높이 h만큼 내려왔을 때 중심 속도를 구하라.",
            "속도",
            "복합 풀이",
        ),
        (
            "공이 지면에서 속도 20 m/s, 각도 30도로 발사된다. 최고 높이와 비행 시간을 구하라.",
            "시간",
            "운동학",
        ),
        (
            "마찰 없는 경사면에서 물체가 높이 h만큼 내려와 바닥에서 속도를 구하라.",
            "속도",
            "일-에너지 원리",
        ),
        (
            "마찰 있는 경사면에서 물체의 가속도를 구하라.",
            "가속도",
            "뉴턴 제2법칙 F=ma",
        ),
        (
            "마찰 있는 경사면에서 3 m 이동한 뒤 속도를 구하라.",
            "속도",
            "복합 풀이",
        ),
        (
            "수평면에서 스프링이 압축된다. 최대 압축 거리를 구하라.",
            "위치/변위",
            "일-에너지 원리",
        ),
        (
            "질량 없는 도르래에 연결된 두 물체의 가속도를 구하라.",
            "가속도",
            "뉴턴 제2법칙 F=ma",
        ),
        (
            "질량 있는 도르래의 관성모멘트 I가 주어지고 줄이 미끄러지지 않는다. 가속도를 구하라.",
            "가속도",
            "복합 풀이",
        ),
        (
            "반지름 R인 원판이 고정축 주위로 회전한다. 각가속도를 구하라.",
            "각속도/각가속도",
            "강체 평면운동",
        ),
    ]
    for problem, goal, expected in cases:
        actual, cues = primary_for(problem, goal)
        assert actual == expected, f"expected={expected}, actual={actual}, cues={cues}"


def test_negation_and_radius_context() -> None:
    cues = app.auto_detect_cues("마찰 없는 트랙에서 물체가 내려온다.", "")
    assert cues["friction"] is False

    cues = app.auto_detect_cues("frictionless circular loop에서 움직인다.", "")
    assert cues["friction"] is False
    assert cues["circular"] is True

    cues = app.auto_detect_cues("반지름 R인 원판이 굴러 내려온다.", "")
    assert cues["rotation"] is True
    assert cues["circular"] is False

    cues = app.auto_detect_cues("반지름 R인 원형 고리를 지난다.", "")
    assert cues["circular"] is True


def test_distance_detection() -> None:
    cues = app.auto_detect_cues("마찰 있는 경사면에서 3 m 이동한 뒤 속도를 구하라.", "")
    assert cues["distance"] is True
    assert cues["friction"] is True


def test_reason_log_and_checklists() -> None:
    diagnosis = app.build_diagnosis(
        problem="높이 h에서 출발해 반지름 R인 원형 고리를 지난다. 꼭대기에서 떨어지지 않기 위한 최소 높이를 구하라.",
        solution="에너지 보존만 쓰면 될 것 같다.",
        goal="접촉 유지 조건",
        user_method="일-에너지 원리",
        manual_cues={key: False for key in app.CUE_LABELS},
    )
    logs = app.build_reason_log(diagnosis)
    checklists = app.get_problem_type_checklists(diagnosis)
    assert logs, "판단 이유 로그가 비어 있으면 안 된다."
    assert "원운동 체크리스트" in checklists
    assert "일-에너지 체크리스트" in checklists


def test_v4_highlight_and_fbd_helpers() -> None:
    problem = "마찰 없는 원형 트랙에서 높이 h부터 출발한다."
    cues = app.auto_detect_cues(problem, "")
    spans = app.get_highlight_spans(problem, cues)
    html = app.highlighted_problem_html(problem, cues)
    assert spans, "단서 하이라이트 span이 있어야 한다."
    assert "마찰 없는" in html, "부정 단서도 하이라이트되어야 한다."
    assert cues["friction"] is False
    assert cues["circular"] is True
    expected = app.expected_fbd_items(cues)
    assert "중력 mg" in expected
    assert "수직항력 N" in expected


def test_v4_learning_record_helpers() -> None:
    diagnosis = app.build_diagnosis(
        problem="마찰 있는 경사면에서 물체의 가속도를 구하라.",
        solution="F=ma를 쓴다.",
        goal="가속도",
        user_method="뉴턴 제2법칙 F=ma",
        manual_cues={key: False for key in app.CUE_LABELS},
    )
    record = app.make_record("문제", "풀이", "가속도", "뉴턴 제2법칙 F=ma", diagnosis, "메모")
    assert record["recommended"] == "뉴턴 제2법칙 F=ma"
    assert isinstance(record["missing_categories"], list)
    assert app.categorize_issue("좌표축과 양의 방향을 정하라") == "좌표축"


def test_constant_acceleration_consistency() -> None:
    valid, _, warnings = app.solve_constant_acceleration({"s": None, "u": 0, "v": None, "a": 2, "t": 3}, "v")
    assert valid == [6.0]
    assert warnings == []

    valid, _, warnings = app.solve_constant_acceleration({"s": 100, "u": 0, "v": None, "a": 2, "t": 3}, "v")
    assert valid == []
    assert warnings, "모순 입력이면 경고가 있어야 한다."


if __name__ == "__main__":
    test_rule_engine_examples()
    test_negation_and_radius_context()
    test_distance_detection()
    test_reason_log_and_checklists()
    test_v4_highlight_and_fbd_helpers()
    test_v4_learning_record_helpers()
    test_constant_acceleration_consistency()
    print("OK: regression tests passed")
