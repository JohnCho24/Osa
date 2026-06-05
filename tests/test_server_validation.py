"""HTTP-level request validation. These run without a model — they only exercise
the Pydantic schemas + the path-traversal guard, which is fast and deterministic.
"""
import pytest
from pydantic import ValidationError

from src.server import main as server_main
from src.server.main import (
    GenerateRequest,
    PlayerArrow,
    BallPassArrow,
    _safe_match_path,
    MAX_K,
    MAX_HORIZON_FRAMES,
    MAX_ARROWS,
    _content_length_exceeds_limit,
)


# ── Arrow discriminated union ──────────────────────────────────────────────
def test_player_arrow_valid():
    a = PlayerArrow(team="team0", player="Player2", to=[10.0, 5.0])
    assert a.kind == "player"


def test_ball_pass_arrow_valid():
    a = BallPassArrow(kind="ball_pass", to=[10.0, 5.0],
                      recipient={"team": "team0", "player": "Player2"})
    assert a.kind == "ball_pass"
    assert a.recipient.team == "team0"


def test_ball_pass_arrow_missing_recipient_rejected():
    with pytest.raises(ValidationError):
        BallPassArrow(kind="ball_pass", to=[10.0, 5.0])


def test_arrow_wrong_team_format_rejected():
    with pytest.raises(ValidationError):
        PlayerArrow(team="team_red", player="Player2", to=[10.0, 5.0])


def test_arrow_to_must_be_2d():
    with pytest.raises(ValidationError):
        PlayerArrow(team="team0", player="Player2", to=[10.0, 5.0, 1.0])


# ── GenerateRequest bounds ─────────────────────────────────────────────────
def _ok_body():
    return {
        "decision_frame": 15000,
        "match": "data/processed/Sample_Game_1.json",
        "horizon_frames": 25,
        "k": 1,
        "arrows": [{"kind": "player", "team": "team0", "player": "Player2", "to": [10.0, 5.0]}],
    }


def test_generate_request_happy_path():
    GenerateRequest(**_ok_body())


def test_k_over_max_rejected():
    body = _ok_body()
    body["k"] = MAX_K + 1
    with pytest.raises(ValidationError):
        GenerateRequest(**body)


def test_horizon_over_max_rejected():
    body = _ok_body()
    body["horizon_frames"] = MAX_HORIZON_FRAMES + 1
    with pytest.raises(ValidationError):
        GenerateRequest(**body)


def test_too_many_arrows_rejected():
    body = _ok_body()
    body["arrows"] = [
        {"kind": "player", "team": "team0", "player": f"P{i}", "to": [0, 0]}
        for i in range(MAX_ARROWS + 1)
    ]
    with pytest.raises(ValidationError):
        GenerateRequest(**body)


def test_guidance_scale_out_of_range_rejected():
    body = _ok_body()
    body["guidance_scale"] = 99.0
    with pytest.raises(ValidationError):
        GenerateRequest(**body)


def test_negative_guidance_scale_rejected():
    body = _ok_body()
    body["guidance_scale"] = -1.0
    with pytest.raises(ValidationError):
        GenerateRequest(**body)


def test_missing_kind_on_arrow_rejected():
    """Discriminator field is required — drift from v0.1 must be loud."""
    body = _ok_body()
    body["arrows"] = [{"team": "team0", "player": "Player2", "to": [10.0, 5.0]}]
    with pytest.raises(ValidationError):
        GenerateRequest(**body)


# ── _safe_match_path ───────────────────────────────────────────────────────
def test_path_traversal_rejected():
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        _safe_match_path("../../../etc/passwd")


def test_absolute_path_outside_processed_rejected():
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        _safe_match_path("/etc/hosts")


def test_non_json_extension_rejected():
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        _safe_match_path("data/processed/Sample_Game_1.csv")


def test_valid_match_path_resolves(tmp_path, monkeypatch):
    match_dir = tmp_path / "data" / "processed"
    match_dir.mkdir(parents=True)
    match_file = match_dir / "Sample_Game_1.json"
    match_file.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(server_main, "ROOT", tmp_path)
    monkeypatch.setattr(server_main, "MATCH_DIR", match_dir)

    p = _safe_match_path("data/processed/Sample_Game_1.json")
    assert p.suffix == ".json"
    assert p.is_file()


# ── body size guard ────────────────────────────────────────────────────────
def test_content_length_guard_accepts_missing_and_bounded_values():
    assert not _content_length_exceeds_limit(None, limit=1024)
    assert not _content_length_exceeds_limit("1024", limit=1024)


def test_content_length_guard_rejects_oversized_or_invalid_values():
    assert _content_length_exceeds_limit("1025", limit=1024)
    assert _content_length_exceeds_limit("-1", limit=1024)
    assert _content_length_exceeds_limit("not-an-int", limit=1024)
