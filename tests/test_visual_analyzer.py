"""Tests for analyzers/visual_analyzer.py — pure logic + availability gating.

Frame extraction and the actual Ollama vision call are never exercised
here (no ffmpeg/network in unit tests) — only the parts that don't need
either: the "is this caption interesting" heuristic, and describe_blocks()'s
short-circuit when the vision model isn't available.
"""
import analyzers.visual_analyzer as va


# ---------------------------------------------------------------------------
# is_visually_interesting
# ---------------------------------------------------------------------------

def test_empty_caption_not_interesting():
    assert va.is_visually_interesting("") is False
    assert va.is_visually_interesting(None) is False


def test_action_caption_is_interesting():
    assert va.is_visually_interesting("A man punches another man in a bar fight.") is True


def test_static_caption_not_interesting():
    assert va.is_visually_interesting("A black screen, nothing happening.") is False
    assert va.is_visually_interesting("The end credits are rolling.") is False
    assert va.is_visually_interesting("A title card on a blank screen.") is False


# ---------------------------------------------------------------------------
# describe_blocks — short-circuits cleanly when the vision model is off
# ---------------------------------------------------------------------------

def test_describe_blocks_skips_when_unavailable(monkeypatch):
    monkeypatch.setattr(va, "visual_analysis_available", lambda *a, **kw: False)
    calls = []
    monkeypatch.setattr(va, "describe_frame", lambda *a, **kw: calls.append(1) or ("should not run", None))

    blocks = [
        {"start": 0, "end": 10},
        {"start": 10, "end": 20},
    ]
    result = va.describe_blocks("fake.mp4", blocks)

    assert calls == []  # never attempted a real frame grab
    assert all(b["visual"] == "" for b in result)


def test_describe_blocks_captions_each_block_when_available(monkeypatch):
    monkeypatch.setattr(va, "visual_analysis_available", lambda *a, **kw: True)

    seen_timestamps = []

    def fake_describe_frame(video_path, timestamp_sec, model=None):
        seen_timestamps.append(timestamp_sec)
        return f"caption at {timestamp_sec}", None

    monkeypatch.setattr(va, "describe_frame", fake_describe_frame)

    blocks = [
        {"start": 0, "end": 10},
        {"start": 100, "end": 140},
    ]
    result = va.describe_blocks("fake.mp4", blocks)

    # sampled at each block's mid-point
    assert seen_timestamps == [5.0, 120.0]
    assert result[0]["visual"] == "caption at 5.0"
    assert result[1]["visual"] == "caption at 120.0"


def test_visual_analysis_available_respects_config_flag(monkeypatch):
    import config
    monkeypatch.setattr(config, "VISUAL_ANALYSIS_ENABLED", False)
    va._availability["checked"] = False  # reset cache so the flag is actually consulted
    assert va.visual_analysis_available() is False
