"""Tests for batch processing functions."""
import pytest
from core.batch import _snap_scene_boundary, _diversity_filter, _deduplicate_clips
from core.batch import _resolve_movie_title
from core.batch import _is_credit_or_silent


def test_snap_to_sentence_end():
    """Segment with '.' at 32s, max_dur=30 → snap to 32s (latest in [27, 35])."""
    segments = [
        {"start": 0, "end": 20, "text": "First part here"},
        {"start": 20, "end": 32, "text": "Ends with a period."},
    ]
    new_end, extended = _snap_scene_boundary(segments, 0, 60, 30)
    # 32 is the latest sentence end in [27, 35], and 32 > 30
    assert new_end == 32
    assert extended


def test_snap_no_dialogue():
    """Empty segments → end = start + max_dur."""
    new_end, extended = _snap_scene_boundary([], 10, 60, 30)
    assert new_end == 40  # 10 + 30
    assert not extended


def test_snap_sentence_not_found():
    """Segments without punctuation → fallback to max_dur."""
    segments = [
        {"start": 0, "end": 10, "text": "no punctuation here"},
        {"start": 10, "end": 20, "text": "still no punctuation"},
    ]
    new_end, extended = _snap_scene_boundary(segments, 0, 60, 15)
    # No punctuation in [12, 20], should fallback to max_dur
    assert new_end == 15
    assert not extended


def test_snap_extended_flag():
    """Snap > max_dur → extended=True."""
    segments = [
        {"start": 0, "end": 34, "text": "End of sentence."},
    ]
    new_end, extended = _snap_scene_boundary(segments, 0, 60, 30)
    # Sentence end at 34 is in [27, 35] and > 30
    assert new_end == 34
    assert extended


def test_diversity_filter_spread():
    """10 scenes across 3 segments → results distributed."""
    scenes = [
        {"start": 10, "end": 20, "score": 9},
        {"start": 30, "end": 40, "score": 8},
        {"start": 50, "end": 60, "score": 7},
        {"start": 70, "end": 80, "score": 6},
        {"start": 90, "end": 100, "score": 5},
    ]
    result = _diversity_filter(scenes, 3, 100)
    assert len(result) <= 3
    assert result == sorted(result, key=lambda x: x["start"])


def test_diversity_filter_fewer():
    """Fewer scenes than num_clips → all returned."""
    scenes = [
        {"start": 10, "end": 20, "score": 9},
        {"start": 30, "end": 40, "score": 8},
    ]
    result = _diversity_filter(scenes, 5, 100)
    assert len(result) == 2


def test_diversity_filter_threshold():
    """Test diversity filter removes low-scoring scenes."""
    from core.batch import _diversity_filter
    scenes = [
        {"start": 10, "end": 20, "duration": 10, "text": "test", "score": 9, "title": "a"},
        {"start": 30, "end": 40, "duration": 10, "text": "test", "score": 3, "title": "b"},
        {"start": 50, "end": 60, "duration": 10, "text": "test", "score": 8, "title": "c"},
    ]
    # With total_duration large enough, all scenes should be spread into separate segments
    result = _diversity_filter(scenes, num_clips=2, total_duration=120)
    assert len(result) <= 2
    # Best scenes (highest score) should be selected first
    for s in result:
        assert s["score"] >= 7


def test_dedup_removes_close_lower_score():
    """A(8,100) B(5,150) gap=50 < 120 → B removed."""
    clips = [
        {"start": 100, "score": 8, "title": "A"},
        {"start": 150, "score": 5, "title": "B"},
    ]
    result = _deduplicate_clips(clips)
    assert len(result) == 1
    assert result[0]["title"] == "A"


def test_dedup_keeps_distant_clips():
    """A(8,100) B(5,250) gap=150 >= 120 → both kept."""
    clips = [
        {"start": 100, "score": 8, "title": "A"},
        {"start": 250, "score": 5, "title": "B"},
    ]
    result = _deduplicate_clips(clips)
    assert len(result) == 2


def test_dedup_chain_removal():
    """A(8,100) B(5,150) C(7,220) → A kept (kills B), C kept (gap 120 from A)."""
    clips = [
        {"start": 100, "score": 8, "title": "A"},
        {"start": 150, "score": 5, "title": "B"},
        {"start": 220, "score": 7, "title": "C"},
    ]
    result = _deduplicate_clips(clips)
    assert len(result) == 2
    assert [c["title"] for c in result] == ["A", "C"]


def test_diversity_filter_min_score_segment_filtered():
    """Segment with max score below min_score yields 0 clips; total met from high-scorers."""
    scenes = [
        {"start": 10, "end": 20, "score": 5},
        {"start": 30, "end": 40, "score": 4},
        {"start": 60, "end": 70, "score": 8},
        {"start": 80, "end": 90, "score": 9},
    ]
    # total_duration=100, num_clips=2, segment_dur=50
    # Segment 0 (0-50): scores 5,4 — both < 6.0 → 0 clips
    # Segment 1 (50-100): scores 8,9 — both >= 6.0 → picks score 9
    # fill-remaining: fills 1 more → score 8
    result = _diversity_filter(scenes, num_clips=2, total_duration=100, min_score=6.0)
    assert len(result) == 2
    assert all(s["score"] >= 8 for s in result)


def test_diversity_filter_min_score_zero():
    """min_score=0 → all segments qualify, low-score scenes can be selected."""
    scenes = [
        {"start": 10, "end": 20, "score": 2},
        {"start": 30, "end": 40, "score": 1},
        {"start": 50, "end": 60, "score": 3},
    ]
    result = _diversity_filter(scenes, num_clips=2, total_duration=100, min_score=0)
    assert len(result) == 2
    assert any(s["score"] <= 2 for s in result)


def test_diversity_filter_min_score_all_below():
    """All scenes below min_score, fewer than num_clips → returns all scenes, no crash."""
    scenes = [
        {"start": 10, "end": 20, "score": 3},
        {"start": 50, "end": 60, "score": 5},
    ]
    result = _diversity_filter(scenes, num_clips=5, total_duration=100, min_score=7.0)
    assert len(result) == 2
    assert len(result) < 5


def test_dedup_empty():
    """Empty list → empty."""
    assert _deduplicate_clips([]) == []


def test_dedup_single():
    """Single clip → kept."""
    clips = [{"start": 100, "score": 5, "title": "A"}]
    result = _deduplicate_clips(clips)
    assert len(result) == 1
    assert result[0]["title"] == "A"


# --- Todo 25: movie_title fallback visibility warning ---

def test_resolve_movie_title_empty_warns_with_stem(capsys):
    """Empty movie_title → stem returned AND warning printed containing marker + stem."""
    title = _resolve_movie_title({"movie_title": ""}, "D:/Movies/Inception.2010.1080p.mkv")
    assert title == "Inception.2010.1080p"
    out = capsys.readouterr().out
    assert "No movie title was entered" in out
    assert "Inception.2010.1080p" in out


def test_resolve_movie_title_missing_key_warns(capsys):
    """Key absent entirely → same fallback path, warning printed."""
    title = _resolve_movie_title({}, "D:/Movies/SomeFilm.mp4")
    assert title == "SomeFilm"
    out = capsys.readouterr().out
    assert "No movie title was entered" in out
    assert "SomeFilm" in out


def test_resolve_movie_title_given_no_warning(capsys):
    """Non-empty movie_title → returned as-is, no warning."""
    title = _resolve_movie_title({"movie_title": "Inception"}, "D:/Movies/Inception.mkv")
    assert title == "Inception"
    assert "No movie title was entered" not in capsys.readouterr().out


# --- T5: batch sizing + prompt budget guard ---
#
# Since the cloud providers (Yandex/OpenRouter/OpenCode Zen) were removed,
# there's a single active local model and _batch_size_for_model/
# _max_prompt_chars no longer vary by model name — they read
# config.DEFAULT_LLM_BATCH_SIZE / config.OLLAMA_NUM_CTX directly. Tests pin
# those via monkeypatch rather than depend on whatever model is configured.

import config as _config
from core.batch import _batch_size_for_model, _max_prompt_chars, _prompt_content_chars, _split_batches

TEMPLATE_LEN = 2000


def test_batch_size_is_the_configured_default(monkeypatch):
    """Model name no longer matters — always DEFAULT_LLM_BATCH_SIZE."""
    monkeypatch.setattr(_config, "DEFAULT_LLM_BATCH_SIZE", 4)
    assert _batch_size_for_model("mistral-nemo:12b") == 4
    assert _batch_size_for_model("anything-else") == 4


def test_max_prompt_chars_scales_with_num_ctx(monkeypatch):
    """ctx * 2.5 chars/token * 0.5 input budget."""
    monkeypatch.setattr(_config, "OLLAMA_NUM_CTX", 1_000_000)
    assert _max_prompt_chars("mistral-nemo:12b") == 1_250_000


def test_normal_blocks_keep_batch_8(monkeypatch):
    """24 small blocks + a generous context → six batches of exactly 4, no truncation."""
    monkeypatch.setattr(_config, "OLLAMA_NUM_CTX", 1_000_000)
    monkeypatch.setattr(_config, "DEFAULT_LLM_BATCH_SIZE", 4)
    blocks = [
        {"start": i * 120, "end": (i + 1) * 120, "text": "Short dialogue." * 10}
        for i in range(24)
    ]
    limit = _max_prompt_chars("mistral-nemo:12b")
    batches = _split_batches(blocks, _batch_size_for_model("mistral-nemo:12b"), limit - TEMPLATE_LEN)
    assert len(batches) == 6
    assert all(len(b) == 4 for b in batches)
    # nothing truncated
    assert all(len(blk["text"]) == len(blocks[0]["text"]) for b in batches for blk in b)


def test_oversized_dialogues_fit_limit(monkeypatch):
    """12 huge-dialogue blocks + a small context → every batch fits, dialogues truncated, 200-char floor."""
    monkeypatch.setattr(_config, "OLLAMA_NUM_CTX", 16_000)  # int(16000*2.5*0.5) = 20_000
    big = "words " * 10000  # 60_000 chars per dialogue
    blocks = [
        {"start": i * 120, "end": (i + 1) * 120, "text": big}
        for i in range(12)
    ]
    limit = _max_prompt_chars("mistral-nemo:12b")
    batches = _split_batches(blocks, 2, limit - TEMPLATE_LEN)
    assert len(batches) == 12  # halved down to single-block batches
    for b in batches:
        est = TEMPLATE_LEN + _prompt_content_chars(b)
        assert est <= limit, f"batch estimate {est} > limit {limit}"
        for blk in b:
            assert len(blk["text"]) >= 200          # floor respected
            assert len(blk["text"]) < len(big)      # actually truncated


def test_console_batch_before_block():
    from pathlib import Path
    src = Path("core/batch.py").read_text(encoding="utf-8")
    # console order: Batch header before Block line
    assert src.index("Batch {batch_num}") < src.index("Block {global_idx+1}")


# ---------------------------------------------------------------------------
# _is_credit_or_silent — dialogue-silent blocks rescued by a visual caption
# ---------------------------------------------------------------------------

def test_silent_block_dropped_without_visual():
    """No dialogue, no visual description → filtered (unchanged old behavior)."""
    block = {"text": "", "audio_peaks": {"silence_ratio": 0.1}}
    assert _is_credit_or_silent(block) is True


def test_silent_block_rescued_by_interesting_visual():
    """No dialogue but the vision model describes real action → kept."""
    block = {
        "text": "",
        "visual": "Two men fighting violently in a dark alley.",
        "audio_peaks": {"silence_ratio": 0.1},
    }
    assert _is_credit_or_silent(block) is False


def test_silent_block_not_rescued_by_static_visual():
    """No dialogue and the vision model describes an empty/static shot → still filtered."""
    block = {
        "text": "",
        "visual": "A black screen, nothing happening.",
        "audio_peaks": {"silence_ratio": 0.1},
    }
    assert _is_credit_or_silent(block) is True


def test_credits_text_dropped_even_with_interesting_visual():
    """Explicit credits text is always filtered, regardless of visuals."""
    block = {
        "text": "Subtitles by example.com — synced by someone",
        "visual": "An intense car chase through a city street.",
        "audio_peaks": {"silence_ratio": 0.0},
    }
    assert _is_credit_or_silent(block) is True


def test_block_with_real_dialogue_kept_regardless_of_visual():
    """Enough dialogue on its own is still sufficient, exactly like before."""
    block = {
        "text": "This is a long enough line of real dialogue to pass the filter easily.",
        "visual": "",
        "audio_peaks": {"silence_ratio": 0.0},
    }
    assert _is_credit_or_silent(block) is False


def test_mostly_silent_audio_rescued_by_visual():
    """High silence_ratio alone used to drop a block — a visual description saves it."""
    block = {
        "text": "",
        "visual": "A tense standoff, two characters staring each other down.",
        "audio_peaks": {"silence_ratio": 0.95},
    }
    assert _is_credit_or_silent(block) is False
