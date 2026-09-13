"""T4 — null/0-parsed split-retry and adaptive max_tokens."""
import json
import pytest
import core.batch as cb

def _make_blocks(n=4, start_step=60):
    blocks = []
    for i in range(n):
        blocks.append({
            "start": i * start_step,
            "end": (i+1) * start_step,
            "text": f"Dialogue block {i} with enough length to pass filter. " * 3,
            "cut_count": 2,
            "pause_points": [],
            "audio_peaks": {"silence_ratio": 0.1},
        })
    return blocks

def _valid_json_for(offset, count):
    """Return JSON array with one clip per block, absolute timestamps inside block."""
    items = []
    for k in range(count):
        idx = k  # block-local index for sub-batch
        # Use block start + 5 to 35
        items.append({"start": offset*60 + 5 + k*60, "end": offset*60 + 35 + k*60, "title": f"Clip {k}", "score": 8.5, "reason": "test", "block": idx})
    return json.dumps(items)

def test_split_retry_on_null_batch(monkeypatch):
    """Batch 4 returns None, halves return valid JSON → merged with index correction."""
    blocks4 = _make_blocks(4)
    # _max_tokens_for() scales off config.OLLAMA_NUM_CTX — pin it so this
    # test doesn't depend on whatever model happens to be configured live.
    monkeypatch.setattr(cb.config, "OLLAMA_NUM_CTX", 16000)
    # prevent merging/filtering/batching side effects
    monkeypatch.setattr(cb, "_merge_blocks_for_llm", lambda x, *a, **kw: x)
    monkeypatch.setattr(cb, "_is_credit_or_silent", lambda b: False)
    monkeypatch.setattr(cb, "_split_batches", lambda blocks, bs, budget: [blocks])
    # no real ffmpeg/Ollama vision calls in tests — pass blocks through untouched
    monkeypatch.setattr("analyzers.visual_analyzer.describe_blocks", lambda video_path, blocks, *a, **kw: blocks)

    # mock detect_and_transcribe to return our blocks, no file IO
    monkeypatch.setattr("analyzers.scene_analyzer.detect_and_transcribe", lambda *a, **kw: blocks4)

    # validation: accept any clips (bypass duration/score checks by patching)
    # Actually _validate_sub_clips is used; let it pass through — our clips are 30s duration within bounds
    monkeypatch.setattr(cb, "_validate_sub_clips", lambda clips, s, e, d, *a, **kw: clips)
    monkeypatch.setattr(cb, "_deduplicate_clips", lambda clips: sorted(clips, key=lambda c: c["start"]))

    calls = []
    # first call (whole batch 4) -> None, then two halves each 2 -> valid JSON
    half1_json = json.dumps([
        {"start": 5, "end": 35, "title": "Clip0", "score": 8.5, "reason": "x", "block": 0},
        {"start": 65, "end": 95, "title": "Clip1", "score": 7.5, "reason": "x", "block": 1},
    ])
    half2_json = json.dumps([
        {"start": 125, "end": 155, "title": "Clip2", "score": 8.0, "reason": "x", "block": 0},
        {"start": 185, "end": 215, "title": "Clip3", "score": 7.0, "reason": "x", "block": 1},
    ])
    seq = [None, half1_json, half2_json]
    def fake_call(prompt, max_tokens=4096):
        calls.append(max_tokens)
        return seq[min(len(calls)-1, len(seq)-1)]
    monkeypatch.setattr(cb, "call_llm", fake_call)
    # also patch analyzers.text_analyzer.call_llm if imported elsewhere
    import analyzers.text_analyzer as ta
    monkeypatch.setattr(ta, "call_llm", fake_call)

    # need to avoid diversity/dedup messing — patch to identity where possible but keep behavior
    # Use small num_clips large enough
    result = cb.find_best_clips_context("fake.mp4", "TestFilm", max_duration=60, min_duration=15, num_clips=10, score_threshold=7.0)
    # Should have recovered 4 clips via split-retry, before threshold filtering
    assert result is not None
    assert len(result) == 4
    # max_tokens adaptive, capped at half of OLLAMA_NUM_CTX=16000 -> 8000:
    # first call for 4 blocks wants 10240, capped to 8000; halves (2 each)
    # want 6144, under the cap so untouched.
    assert calls[0] == 8000
    assert calls[1] == 6144
    assert calls[2] == 6144
    # check index correction: starts should be distinct and increasing
    starts = [c["start"] for c in result]
    assert starts == sorted(starts)
    assert len(set(starts)) == 4

def test_single_block_null_no_loop(monkeypatch):
    """Single block null → no split, fallback to smart centering."""
    blocks1 = _make_blocks(1)
    monkeypatch.setattr(cb.config, "OLLAMA_NUM_CTX", 16000)
    monkeypatch.setattr(cb, "_merge_blocks_for_llm", lambda x, *a, **kw: x)
    monkeypatch.setattr(cb, "_is_credit_or_silent", lambda b: False)
    monkeypatch.setattr(cb, "_split_batches", lambda blocks, bs, budget: [blocks])
    monkeypatch.setattr("analyzers.visual_analyzer.describe_blocks", lambda video_path, blocks, *a, **kw: blocks)
    monkeypatch.setattr("analyzers.scene_analyzer.detect_and_transcribe", lambda *a, **kw: blocks1)
    # keep validate, fallback uses _find_best_window
    monkeypatch.setattr(cb, "_find_best_window", lambda segs, s, e, md: (s+5, s+35))

    calls = []
    def fake_call_null(prompt, max_tokens=4096):
        calls.append(max_tokens)
        return None
    monkeypatch.setattr(cb, "call_llm", fake_call_null)
    import analyzers.text_analyzer as ta
    monkeypatch.setattr(ta, "call_llm", fake_call_null)

    result = cb.find_best_clips_context("fake.mp4", "SingleFilm", max_duration=60, min_duration=15, num_clips=10, score_threshold=7.0)
    # single block should not loop; fallback creates 1 clip with score 5.0
    assert len(calls) == 1
    assert calls[0] == 4096  # 4096*1//2+2048=4096, well under the 8000 cap
    assert result is not None
    assert len(result) == 1
    assert result[0]["score"] == 5.0
