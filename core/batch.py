"""
MovieShort AI — Batch processor.
Processes a full movie: auto-detect best scenes → process each into a Short.
"""
import os
import re
import shutil
from pathlib import Path

import config
from analyzers.text_analyzer import call_llm
from analyzers.visual_analyzer import is_visually_interesting
from core.pipeline import process_multiple
from core.subtitle import (
    find_external_subtitle,
    load_external_subtitles,
    load_segments_json,
    save_segments_json,
)
from utils import get_video_basename


# credits/music filter — R7b-6
_CREDITS_RE = re.compile(
    r'subtitles? by|subs? by|synced? by|sync and correct|translat(ed|ion) by|'
    r'proofread|encoded by|www\.|\.com',
    re.I,
)

# SDH bracketed/parenthetical sound cue, e.g. "[gunshot]", "(door slams)" —
# real signal about what's happening on screen even with zero spoken dialogue,
# so it should never be treated the same as true silence.
_SDH_SOUND_CUE_RE = re.compile(r'[\[(][^\])]{2,60}[\])]')


def _is_credit_or_silent(block):
    """True if block should be filtered out entirely.

    Credits/junk subtitle text is always filtered, regardless of visuals.
    A block with little/no dialogue (<30 chars) or mostly silent audio is
    ONLY filtered if it ALSO has no other signal that something's actually
    happening:
      - its visual description (from the vision model, see
        analyzers/visual_analyzer.py) shows nothing interesting, AND
      - it has no SDH bracketed sound/action cue ("[gunshot]", "(door
        slams)") — those are real signal on their own, even with zero
        spoken dialogue and no visual analysis available for that block.
    A real fight, chase, or silent emotional beat can carry a clip even
    with no dialogue at all, which text-only analysis used to miss
    entirely.
    """
    text = (block.get("text") or "").strip()
    if _CREDITS_RE.search(text):
        return True

    visually_interesting = is_visually_interesting(block.get("visual"))
    has_sound_cue = bool(_SDH_SOUND_CUE_RE.search(text))

    if (not text or len(text) < 30) and not visually_interesting and not has_sound_cue:
        return True

    audio_peaks = block.get("audio_peaks") or {}
    if (audio_peaks.get("silence_ratio", 0) > 0.85
            and not visually_interesting and not has_sound_cue):
        return True

    return False


def _resolve_movie_title(settings, video_path):
    """User title if given; else filename stem (with a visibility warning —
    renamed files leak garbage into the LLM prompt)."""
    raw = (settings.get("movie_title") or "").strip()
    title = raw or Path(video_path).stem
    if raw:
        print(f"Movie title: «{title}» (used for AI context and output filenames)")
    else:
        print(f"⚠ No movie title was entered — using the filename «{title}» "
              "instead, which may confuse the AI if it's full of release-group "
              "junk (resolution, codec, etc). Enter a title for better results.")
    return title


def process_movie(video_path, settings=None):
    """
    Full auto pipeline: analyze movie → find best clips → process each → output.

    Args:
        video_path: path to movie file
        settings: dict with keys:
            - max_duration / min_duration: not accepted from the GUI —
              every clip is a fixed 120-180s, see config.DEFAULT_MIN/MAX_CLIP_DURATION.
            - subtitles (bool): enable subtitles (default True)
            - face_tracking (bool): enable face tracking (default True)
            - srt_subtitle_path (str): REQUIRED — plain dialogue-only
              subtitle file, used ONLY for the burned-in captions on the
              final rendered clip.
            - sdh_subtitle_path (str): REQUIRED — SDH subtitle file, used
              ONLY for scene analysis / clip selection. Never burned in
              (an SDH tag like "[grunting]" must never appear as a caption).
              Both subtitle files are required with no auto-discovery
              fallback — see core/subtitle.py's find_external_subtitle.
            - series_context / season_context / episode_context (str):
              optional background info (plot, characters, spoilers allowed)
              fed to the LLM alongside the auto-generated story primer, to
              give it real narrative grounding instead of judging each
              block of dialogue in isolation.
            - anti_copyright (bool): enable anti-copyright measures
            - blur_background (bool): enable blurred background
            - banner_top (int): top banner padding
            - banner_bottom (int): bottom banner padding
            - num_clips (int): max number of clips to produce
            - score_threshold (float): minimum score for clip selection
            - auto_cleanup (bool): delete temp files after processing

    Returns:
        List of output file paths (None entries for failed clips).
    """
    if settings is None:
        settings = {}

    max_duration = settings.get("max_duration", config.DEFAULT_MAX_CLIP_DURATION)
    min_duration = settings.get("min_duration", config.DEFAULT_MIN_CLIP_DURATION)
    subtitles = settings.get("subtitles", True)
    face_tracking = settings.get("face_tracking", True)

    video_path = str(video_path)
    movie_title = _resolve_movie_title(settings, video_path)
    # Sanitize for folder name (remove chars invalid on Windows)
    safe_name = re.sub(r'[\\/:*?"<>|]', '', movie_title).strip() or "untitled"

    # Create output subdirectory for this movie
    movie_output = config.OUTPUT_DIR / safe_name
    os.makedirs(movie_output, exist_ok=True)

    print(f"Processing movie: {os.path.basename(video_path)}")
    print(f"Settings: subtitles burned in = {subtitles}, "
          f"smart face/person centering = {face_tracking}, "
          f"clip length = {min_duration}-{max_duration}s")
    print()

    # Both subtitle files are required — no auto-discovery fallback (see
    # core/subtitle.py's find_external_subtitle). SDH feeds scene analysis
    # only; the plain file feeds only the burned-in captions.
    srt_subtitle_path = find_external_subtitle(
        video_path, explicit_path=settings.get("srt_subtitle_path"),
    )
    sdh_subtitle_path = find_external_subtitle(
        video_path, explicit_path=settings.get("sdh_subtitle_path"),
    )
    if not srt_subtitle_path or not sdh_subtitle_path:
        missing = []
        if not srt_subtitle_path:
            missing.append("plain subtitle file (for burned-in captions)")
        if not sdh_subtitle_path:
            missing.append("SDH subtitle file (for scene analysis)")
        print(f"❌ Missing subtitles: no {' and no '.join(missing)} was uploaded.")
        print("   Both are required — supported formats: .srt, .ass, .ssa, .vtt")
        print("Stopping.")
        return []
    print(f"Using subtitles: {os.path.basename(srt_subtitle_path)} (captions), "
          f"{os.path.basename(sdh_subtitle_path)} (SDH, analysis)")

    # Step 1: Find best clips
    print("=" * 50)
    print("STEP 1: Picking the best moments from the movie")
    print("=" * 50)

    best_scenes = find_best_clips_context(
        video_path, movie_title,
        max_duration, min_duration,
        num_clips=settings.get("num_clips", config.DEFAULT_NUM_CLIPS),
        score_threshold=settings.get("score_threshold", 7.0),
        sdh_subtitle_path=sdh_subtitle_path,
        series_context=settings.get("series_context", ""),
        season_context=settings.get("season_context", ""),
        episode_context=settings.get("episode_context", ""),
    )
    if best_scenes is None:
        print("❌ The AI scene analysis didn't produce a result — see the "
              "errors above for why. Stopping.")
        return []

    if not best_scenes:
        print("No suitable moments were found in this movie.")
        return []

    print()
    print("=" * 50)
    print(f"STEP 2: Rendering {len(best_scenes)} clip(s) to vertical video")
    print("=" * 50)

    # Parse the PLAIN subtitle file into its own cache — deliberately
    # separate from the SDH cache Step 1 built internally — so burned-in
    # captions can never show an SDH-only tag like "[grunting]". Cached by
    # content hash exactly like the SDH one, so re-running the same episode
    # doesn't re-parse.
    import hashlib
    video_basename = get_video_basename(video_path)
    hash_input = (
        f"{video_path}_"
        f"{srt_subtitle_path}_"
        f"{os.path.getmtime(srt_subtitle_path)}"
    )
    file_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
    transcript_json = str(config.CACHE_DIR / f"full_transcript_srt_{video_basename}_{file_hash}.json")
    if not os.path.exists(transcript_json):
        try:
            srt_segments = load_external_subtitles(srt_subtitle_path)
            os.makedirs(config.CACHE_DIR, exist_ok=True)
            save_segments_json(srt_segments, transcript_json)
        except Exception as e:
            print(f"❌ Missing subtitles: couldn't parse the plain subtitle "
                  f"file for burned-in captions: {e}")
            print("Stopping.")
            return []

    # Pre-load transcript segments for smart clip centering
    clip_segments = None
    if transcript_json and os.path.exists(transcript_json):
        try:
            clip_segments = load_segments_json(transcript_json)
        except Exception:
            pass

    # Step 2: Build timestamp list
    timestamps = []
    titles = []
    for scene in best_scenes:
        scene_start = scene["start"]
        scene_end = scene["end"]
        scene_dur = scene_end - scene_start
        title = scene.get("title", "")

        # Smart centering: if scene is longer than max_duration, find best window
        if scene_dur > max_duration and clip_segments:
            new_start, new_end = _find_best_window(
                clip_segments, scene_start, scene_end, max_duration
            )
            if new_start is not None:
                orig_start_fmt = _format_time(scene_start)
                orig_end_fmt = _format_time(scene_end)
                scene_start, scene_end = new_start, new_end
                scene_dur = scene_end - scene_start
                print(f"  🎯 The AI's pick ({orig_start_fmt}-{orig_end_fmt}, "
                      f"{scene_dur:.0f}s) was longer than the {max_duration}s "
                      "max, so trimmed it to the densest dialogue window: "
                      f"{_format_time(scene_start)}-{_format_time(scene_end)}")

        timestamps.append((_format_time(scene_start), _format_time(scene_end)))
        titles.append(title)

    # Step 3: Process clips in parallel
    base_options = {
        "subtitles": subtitles,
        "face_tracking": face_tracking,
        "max_duration": max_duration,
        "movie_title": movie_title,
        "anti_copyright": settings.get("anti_copyright", config.DEFAULT_ANTI_COPYRIGHT),
        "blur_background": settings.get("blur_background", config.DEFAULT_BLUR_BACKGROUND),
        "banner_top": settings.get("banner_top", config.DEFAULT_BANNER_TOP),
        "banner_bottom": settings.get("banner_bottom", config.DEFAULT_BANNER_BOTTOM),
    }
    # R7b-7: Editor subtitle style flows to final clips (same font_style path as render_full_preview)
    for _k in ("subtitle_font", "subtitle_font_name", "subtitle_size", "subtitle_outline", "subtitle_color", "subtitle_bold", "subtitle_italic", "subtitle_shadow", "subtitle_position_y", "font_style"):
        if _k in settings and settings[_k] is not None:
            base_options[_k] = settings[_k]
    base_options["transcript_path"] = transcript_json

    results = process_multiple(video_path, timestamps, base_options, titles=titles, max_workers=2)

    # Move outputs to movie subfolder
    final_results = []
    for i, result in enumerate(results):
        if result and os.path.exists(result):
            src_name = Path(result).stem
            new_name = movie_output / f"{src_name}.mp4"
            shutil.move(result, new_name)
            final_results.append(str(new_name))
        else:
            final_results.append(None)

    results = final_results

    # Summary
    done = sum(1 for r in results if r is not None)
    print(f"\n{'=' * 50}")
    print(f"Complete: {done}/{len(best_scenes)} clip(s) ready "
          "(processed 100% locally — no API cost)")
    print(f"Output folder: {movie_output}")
    print(f"{'=' * 50}")

    # Print output list
    for i, r in enumerate(results):
        if r:
            fname = os.path.basename(r)
            print(f"  ✅ {fname}")
        else:
            clip_title = titles[i] if i < len(titles) and titles[i] else f"clip {i+1}"
            print(f"  ❌ «{clip_title}» — failed, see its errors above")

    # Auto-cleanup: delete temp files if enabled.
    # CACHE_DIR survives auto_cleanup by design (reusable transcripts/person/RMS caches).
    if settings.get("auto_cleanup", True):
        cleanup_temp_dir()

    return results


def cleanup_temp_dir():
    """Wipe ONLY output/temp contents; never touches config.CACHE_DIR."""
    if os.path.exists(config.TEMP_DIR):
        for f in os.listdir(str(config.TEMP_DIR)):
            fp = os.path.join(str(config.TEMP_DIR), f)
            try:
                if os.path.isfile(fp) or os.path.islink(fp):
                    os.unlink(fp)
                elif os.path.isdir(fp):
                    shutil.rmtree(fp)
            except Exception:
                pass
        print("  🧹 Temp files removed")


def _format_time(seconds):
    """Format seconds to HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _snap_scene_boundary(clip_segments, scene_start, scene_end, max_dur):
    """Snap clip end to nearest sentence boundary within [max_dur-3, max_dur+5].

    Priority:
    1. Sentence end (. ! ?) in [max_dur-3, max_dur+5]
    2. Pause in dialogue >0.8s between subtitle segments
    3. Word boundary (fallback)

    Returns (new_end, extended) where new_end <= scene_end.
    """
    if not clip_segments:
        return scene_start + max_dur, False

    # Filter segments in the [scene_start, scene_end] range
    overlapping = [
        s for s in clip_segments
        if s["start"] < scene_end and s["end"] > scene_start
    ]
    if not overlapping:
        return scene_start + max_dur, False

    # Search range: [max_dur - 3, max_dur + 5]
    search_start = scene_start + max(0, max_dur - 3)
    search_end = min(scene_end, scene_start + max_dur + 5)

    # Priority 1: Find sentence-ending punctuation in word timestamps
    best_end = None

    for seg in overlapping:
        seg_end = seg["end"]
        if search_start <= seg_end <= search_end:
            text = seg.get("text", "").strip()
            if text and text[-1] in ".!?…":
                if best_end is None or seg_end > best_end:
                    best_end = seg_end

    if best_end is not None:
        extended = best_end > scene_start + max_dur
        return best_end, extended

    # Priority 2: Find dialogue pause > 0.8s between consecutive segments
    sorted_segs = sorted(overlapping, key=lambda s: s["start"])
    for i in range(len(sorted_segs) - 1):
        gap = sorted_segs[i + 1]["start"] - sorted_segs[i]["end"]
        if gap > 0.8 and search_start <= sorted_segs[i]["end"] <= search_end:
            if best_end is None or sorted_segs[i]["end"] > best_end:
                best_end = sorted_segs[i]["end"]

    if best_end is not None:
        extended = best_end > scene_start + max_dur
        return best_end, extended

    # Priority 3: Word boundary — just use max_dur
    new_end = min(scene_end, scene_start + max_dur)
    return new_end, False


# ---------------------------------------------------------------------------
# Context Mode — LLM sees real scene transcripts, picks best scenes directly
# ---------------------------------------------------------------------------

def _validate_sub_clips(sub_clips, block_start, block_end, block_duration,
                        min_duration=20, max_duration=75):
    """Validate sub-clips from LLM response.

    Applies per-sub-clip, no exceptions:
    1. Within block bounds
    2. Duration >= min_duration
    3. Duration <= max_duration
    4. Score 1-10
    5. No negative start or overflow

    Reports every applicable reason for a rejection in one line — this is
    the only place a rejection gets printed; don't re-log the same
    candidates elsewhere or they'll show up twice with different wording.

    Returns filtered list with logged drops.
    """
    valid = []
    for sc in sub_clips:
        sc_start = block_start + sc.get("start", 0)
        sc_end = block_start + sc.get("end", block_duration)
        sc_dur = sc_end - sc_start
        sc_score = sc.get("score", 5)
        sc_title = sc.get("title", "") or "untitled"

        reasons = []
        if sc_start < 0 or sc_end > block_end:
            reasons.append(
                "its timestamps fall outside the section the AI was "
                "given for this block (it likely mixed up which part "
                "of the movie it was looking at)"
            )
        if sc_dur < min_duration:
            reasons.append(f"{sc_dur:.0f}s is shorter than the {min_duration}s minimum")
        if sc_dur > max_duration:
            reasons.append(f"{sc_dur:.0f}s is longer than the {max_duration}s maximum")
        if sc_score < 1 or sc_score > 10:
            reasons.append(f"its score ({sc_score}) isn't a valid 1-10 rating")

        if reasons:
            print(f"  ⛔ «{sc_title}» ({sc_start:.0f}-{sc_end:.0f}s) skipped: "
                  + "; ".join(reasons))
            continue

        valid.append({
            "start": sc_start,
            "end": sc_end,
            "duration": sc_dur,
            "text": sc.get("text", ""),
            "score": sc_score,
            "title": sc_title[:40],
        })
    return valid


def _expand_short_clips(clips, min_duration):
    """Expand clips shorter than min_duration symmetrically."""
    result = []
    for clip in clips:
        dur = clip["end"] - clip["start"]
        if dur < min_duration:
            mid = (clip["start"] + clip["end"]) / 2
            new_start = max(0, mid - min_duration / 2)
            new_end = new_start + min_duration
            clip["start"] = new_start
            clip["end"] = new_end
            clip["duration"] = new_end - new_start
        result.append(clip)
    return result


def _merge_blocks_for_llm(blocks, target_duration=120, max_duration=150):
    """Merge adjacent blocks into super-blocks of ~target_duration seconds.

    Preserves original scene boundaries within each super-block metadata.
    Combines text, pause_points, cut_count. Keeps audio_peaks from the
    longest constituent block.

    Args:
        blocks: list of block dicts from detect_and_transcribe()
        target_duration: target duration in seconds (default 120)
        max_duration: maximum duration before force-finalize (default 150)

    Returns:
        list of merged block dicts with same schema as input blocks
    """
    if not blocks:
        return []

    merged = []
    buffer = dict(blocks[0])  # shallow copy

    for b in blocks[1:]:
        b = dict(b)
        # Force-finalize if buffer already at max
        if buffer["duration"] >= max_duration:
            merged.append(buffer)
            buffer = b
            continue

        # If buffer is below target, merge this block in
        if buffer["duration"] < target_duration:
            buffer["end"] = b["end"]
            buffer["duration"] = buffer["end"] - buffer["start"]
            # Combine texts
            txt_a = buffer.get("text", "") or ""
            txt_b = b.get("text", "") or ""
            buffer["text"] = (txt_a + " " + txt_b).strip()
            # Merge pause_points
            buffer["pause_points"] = (
                buffer.get("pause_points", []) + b.get("pause_points", [])
            )
            # Sum cut_count
            buffer["cut_count"] = buffer.get("cut_count", 0) + b.get("cut_count", 0)
            # Keep audio_peaks from the longer sub-block
            if b.get("duration", 0) > buffer.get("_dominant_dur", 0):
                buffer["audio_peaks"] = b.get("audio_peaks", {})
                buffer["_dominant_dur"] = b.get("duration", 0)
        else:
            # Buffer is >= target — finalize, start new buffer
            merged.append(buffer)
            buffer = b

    if buffer:
        merged.append(buffer)

    # Strip internal helper fields
    for m in merged:
        m.pop("_dominant_dur", None)

    return merged


# --- Model-aware batch sizing + prompt budget guard (T5) ---

# Chars per block around the dialogue (timestamps, labels, joiner, and the
# "Visual: ..." line's own framing) — estimate
_BLOCK_FRAMING_CHARS = 260
# Minimum dialogue chars kept when a single-block batch still overflows the budget
_TRUNCATE_FLOOR_CHARS = 200


def _batch_size_for_model(model):
    """Blocks per LLM call.

    `model` is unused now that there's a single active local model — kept so
    call sites don't need to change if per-model tuning is reintroduced later.
    """
    return config.DEFAULT_LLM_BATCH_SIZE


def _max_prompt_chars(model):
    """Prompt char budget: context tokens * chars/token estimate * input share.

    `model` is unused now that there's a single active local model — kept so
    call sites don't need to change. The context size that matters is
    whatever config.OLLAMA_NUM_CTX actually tells Ollama to allocate (see
    analyzers/text_analyzer.py's call_llm()), not a generic default: budgeting
    against a bigger number than the model is actually given room for would
    let a batch grow past what fits, and Ollama would silently drop whatever
    doesn't fit rather than error.
    """
    ctx = getattr(config, "OLLAMA_NUM_CTX", 8192)
    return int(ctx * config.PROMPT_CHARS_PER_TOKEN * config.PROMPT_INPUT_BUDGET)


def _prompt_content_chars(batch_blocks):
    """Estimated prompt chars contributed by a batch's blocks (framing +
    dialogue + visual caption)."""
    return sum(
        _BLOCK_FRAMING_CHARS
        + len(b.get("text", "").strip() or "(no dialogue)")
        + len(b.get("visual", "").strip() or "(not analyzed)")
        for b in batch_blocks
    )


def _truncate_batch(chunk, content_budget):
    """Proportionally truncate each block's dialogue so the chunk fits content_budget.

    Returns shallow copies — input blocks are never mutated. Floor 200 chars wins.
    """
    dialogues = [b.get("text", "").strip() or "(no dialogue)" for b in chunk]
    dialogue_total = sum(len(d) for d in dialogues)
    fixed = _prompt_content_chars(chunk) - dialogue_total
    available = content_budget - fixed
    share = available / dialogue_total if dialogue_total > 0 else 0.0
    out = []
    for b, d in zip(chunk, dialogues):
        keep = max(_TRUNCATE_FLOOR_CHARS, int(len(d) * share))
        trimmed = dict(b)
        trimmed["text"] = d[:keep]
        out.append(trimmed)
    return out


def _split_batches(blocks, batch_size, content_budget):
    """Cursor-based split of blocks into LLM batches fitting content_budget.

    While a batch overflows and size > 1 → halve size and re-split the remainder.
    A size-1 overflow gets proportional dialogue truncation. Pure: never mutates
    input blocks; oversized batches come back as shallow copies with cut text.
    """
    batches = []
    cursor = 0
    size = max(1, batch_size)
    while cursor < len(blocks):
        while (_prompt_content_chars(blocks[cursor:cursor + size]) > content_budget
               and size > 1):
            size //= 2
        chunk = blocks[cursor:cursor + size]
        if _prompt_content_chars(chunk) > content_budget:
            chunk = _truncate_batch(chunk, content_budget)
        batches.append(chunk)
        cursor += len(chunk)
    return batches


# Soft cap on each user-supplied context field — this text gets added to
# EVERY batch call (not just once), so a giant pasted synopsis would eat
# real budget from every single request. ~700 chars is roughly 150-200
# words, matching the length asked for in the prompts suggested for
# generating this content externally.
_CONTEXT_FIELD_CHAR_CAP = 700


def _build_context_block(series_context, season_context, episode_context, story_primer):
    """Combine the auto-generated primer with whatever series/season/episode
    context was supplied into one preamble block for the batch prompt.

    Returns "" (no section at all) if nothing was supplied — the prompt
    template's {context_block} placeholder then contributes nothing, and
    scoring behaves exactly as it did before this feature existed.
    """
    parts = []
    if story_primer:
        parts.append(f"Story summary (auto-generated from this episode's "
                      f"subtitles): {story_primer.strip()[:_CONTEXT_FIELD_CHAR_CAP]}")
    if series_context and series_context.strip():
        parts.append(f"Series background: {series_context.strip()[:_CONTEXT_FIELD_CHAR_CAP]}")
    if season_context and season_context.strip():
        parts.append(f"Season background: {season_context.strip()[:_CONTEXT_FIELD_CHAR_CAP]}")
    if episode_context and episode_context.strip():
        parts.append(f"Episode background: {episode_context.strip()[:_CONTEXT_FIELD_CHAR_CAP]}")

    if not parts:
        return ""

    return "Background info (use this to judge what actually matters):\n" + "\n\n".join(parts) + "\n\n"


def find_best_clips_context(video_path, movie_title,
                            max_duration=180, min_duration=60,
                            num_clips=10, score_threshold=7.0,
                            sdh_subtitle_path=None,
                            series_context="", season_context="",
                            episode_context=""):
    """Context mode: detect blocks → local LLM splits each block into sub-clips.

    Each block from detect_and_transcribe() carries metadata:
    pause_points, cut_count, audio_peaks. The LLM decides boundaries and
    titles per block in one call. Falls back to smart centering when the
    LLM returns empty.

    Args:
        video_path: path to video file
        movie_title: movie name for LLM context
        max_duration: max clip length in seconds
        min_duration: min clip length in seconds
        num_clips: max number of clips to return (default 10)
        score_threshold: minimum score (default 7.0)
        sdh_subtitle_path: required SDH subtitle file used for analysis
        series_context / season_context / episode_context: optional
            user-supplied background info (spoilers allowed), combined with
            an auto-generated story primer (see _build_context_block) and
            fed to every batch call so the LLM judges blocks with real
            narrative grounding instead of in total isolation.

    Returns list of {start, end, duration, text, score, title} or None.
    """
    import time
    from pathlib import Path

    from analyzers.scene_analyzer import detect_and_transcribe
    from analyzers.text_analyzer import (
        PROMPT_BATCH_TO_CLIPS,
        _parse_batch_response,
        generate_story_primer,
    )

    video_basename = Path(video_path).stem
    total_start = time.time()
    print(f"\n🎬 Analyzing \"{video_basename}\" to find the best {min_duration}-"
          f"{max_duration}s clips...")

    # Step 1: Detect scenes and map the external SDH subtitle file to them.
    print("  Detecting visual scene changes and matching them to the "
          "SDH subtitle file's dialogue and sound cues...")
    blocks = detect_and_transcribe(
        video_path,
        sdh_subtitle_path=sdh_subtitle_path,
    )

    if not blocks:
        # Should not normally happen — detect_and_transcribe() already
        # raises if subtitles are missing/unusable, or falls back to a
        # single whole-movie block if scene detection finds nothing.
        print("  No usable scenes/subtitles came back — can't continue.")
        return None

    total_duration = blocks[-1]["end"] if blocks else 0
    print(f"  Movie is {_format_time(total_duration)} long ({total_duration:.0f}s), "
          f"split into {len(blocks)} scene(s) with matching dialogue")

    # Step 1.5: Merge small blocks into super-blocks for LLM to work with.
    # Blocks must be comfortably larger than the requested max clip length —
    # a clip can never be selected past its own block's end (see
    # _validate_sub_clips's bounds check) — so a fixed 120-150s super-block
    # silently capped every clip at ~75s regardless of the max_duration the
    # user asked for.
    before_merge = len(blocks)
    super_target = max(120, max_duration + 40)
    super_cap = super_target + 30
    blocks = _merge_blocks_for_llm(blocks, target_duration=super_target, max_duration=super_cap)
    print(f"  Grouped those {before_merge} scenes into {len(blocks)} bigger "
          f"sections (~{super_target}s each, so there's enough dialogue in "
          f"each one for the AI to judge) to send to the AI for scoring")

    # Step 1.55: Visual analysis — caption one sample frame per section with
    # a local vision model. Runs BEFORE the silence filter below so a
    # dialogue-free section (a fight, a chase, a silent reveal) can survive
    # on its visual description alone instead of being dropped as "silent".
    from analyzers.visual_analyzer import describe_blocks
    blocks = describe_blocks(video_path, blocks)

    # Step 1.6: Filter out silent / credits / music blocks
    before_filter = len(blocks)
    blocks = [b for b in blocks if not _is_credit_or_silent(b)]
    filtered = before_filter - len(blocks)
    if filtered:
        print(f"  Skipped {filtered} section(s) with little/no dialogue "
              "(likely credits, silence, or background music)")

    if not blocks:
        print("  Nothing left with actual dialogue in it — nothing to process.")
        return None

    # Step 1.65: Auto-generated story primer + background context.
    #
    # One extra LLM call, made ONCE for the whole movie/episode (not per
    # block): summarizes the surviving blocks' actual dialogue into a short
    # plot/character primer, grounded in real transcript text rather than
    # the model's own training-data guesses. Combined with whatever
    # series/season/episode context was supplied, this becomes a fixed
    # preamble prepended to every batch call below, so each block is judged
    # with real narrative context instead of in total isolation.
    print("  Summarizing the full transcript into a short story primer, so "
          "the AI has real plot/character context when judging each "
          "section (not just that section's own dialogue)...")
    full_transcript_text = "\n".join(
        b.get("text", "").strip() for b in blocks if b.get("text", "").strip()
    )
    story_primer = generate_story_primer(full_transcript_text, movie_title)
    if story_primer:
        print(f"  ✓ Story primer: {story_primer[:150]}"
              f"{'...' if len(story_primer) > 150 else ''}")
    context_block = _build_context_block(
        series_context, season_context, episode_context, story_primer
    )

    model = config.OLLAMA_MODEL
    batch_size = _batch_size_for_model(model)
    batch_template = PROMPT_BATCH_TO_CLIPS
    all_sub_clips = []
    batches = _split_batches(blocks, batch_size,
                             _max_prompt_chars(model) - len(batch_template) - len(context_block))
    total_batches = len(batches)

    # Step 2: Process blocks in batches through LLM
    print(f"  Asking the local AI model ({model}) to pick clips from "
          f"{len(blocks)} section(s), {batch_size} at a time "
          f"(~{total_batches} request(s) total)...")
    print("  Per section below: \"Cuts\" = how many camera/scene changes "
          "happen inside it (higher = more action); \"Pauses\" = natural "
          "gaps in the dialogue the AI can use as a clip's start/end point.")
    batch_start = 0
    for batch_num, batch_blocks in enumerate(batches, 1):
        print(f"\n  ── Batch {batch_num}/{total_batches} "
              f"(blocks {batch_start+1}-{batch_start+len(batch_blocks)}) ──")

        # Build blocks_text for the combined prompt
        block_texts = []
        for i, block in enumerate(batch_blocks):
            global_idx = batch_start + i
            block_start = block["start"]
            block_end = block["end"]
            block_dur = block_end - block_start
            dialogue = block.get("text", "").strip() or "(no dialogue)"
            visual = block.get("visual", "").strip() or "(not analyzed)"
            cut_count = block.get("cut_count", 0)
            pause_points = block.get("pause_points", [])

            dialogue_preview = (dialogue[:80] + "...") if len(dialogue) > 80 else dialogue
            visual_preview = (visual[:80] + "...") if len(visual) > 80 else visual
            print(f"\n  Block {global_idx+1}/{len(blocks)}: {_format_time(block_start)}-{_format_time(block_end)} ({block_dur:.0f}s)")
            print(f"    Dialogue: {dialogue_preview}")
            print(f"    Visual: {visual_preview}")
            print(f"    Cuts: {cut_count}, Pauses: {len(pause_points)}")

            block_texts.append(
                f"--- BLOCK {i} ({_format_time(block_start)}-{_format_time(block_end)}, {block_dur:.0f}s) ---\n"
                f"Dialogue: {dialogue}\n"
                f"Visual: {visual}\n"
                f"Cut count: {cut_count} (high = action)\n"
                f"Dialogue pauses: {pause_points}"
            )

        blocks_text = "\n\n".join(block_texts)
        prompt = batch_template.format(
            movie_name=movie_title,
            context_block=context_block,
            blocks_text=blocks_text,
            min_duration=min_duration,
            max_duration=max_duration,
            pref_duration=(min_duration + max_duration) // 2,
        )

        # T4: adaptive max_tokens, capped to whatever's actually left of the
        # context window after _max_prompt_chars's input share — the old
        # fixed 4096-8192 range was sized for a 16-32K context and would ask
        # for more output than a smaller num_ctx (e.g. 8192) has room for
        # once the input side is accounted for.
        def _max_tokens_for(n: int) -> int:
            ctx = getattr(config, "OLLAMA_NUM_CTX", 8192)
            output_budget = max(1024, int(ctx * (1 - config.PROMPT_INPUT_BUDGET)))
            requested = 4096 * n // 2 + 2048
            return max(min(4096, output_budget), min(requested, output_budget))

        # Call LLM once for the entire batch
        raw_response = None
        try:
            raw_response = call_llm(prompt, max_tokens=_max_tokens_for(len(batch_blocks)))
        except Exception as e:
            print(f"  ⚠️ Batch {batch_num}: the local AI didn't respond ({e}) "
                  "— will retry this batch's sections one at a time below")

        # Parse batch response into per-block clip lists
        if raw_response:
            block_start_times = [b["start"] for b in batch_blocks]
            batch_clips = _parse_batch_response(raw_response, block_start_times)
        else:
            batch_clips = {}

        # Log raw response if nothing was parsed
        if raw_response and not batch_clips:
            raw_short = raw_response.strip()
            if len(raw_short) > 500:
                raw_short = raw_short[:500] + "..."
            print(f"  ⚠️ Batch {batch_num}: got a response but couldn't make "
                  "sense of it as clip data. What the AI actually said "
                  "(truncated):")
            print(f"     {raw_short}")

        # T4: split-retry when null/0-parsed and batch size > 1
        if (raw_response is None or not batch_clips) and len(batch_blocks) > 1:
            print(f"  ↻ Batch {batch_num}: no usable answer for all "
                  f"{len(batch_blocks)} sections together — asking about "
                  "each one separately instead")
            from collections import deque as _dq
            merged: dict[int, list[dict]] = {}
            mid0 = len(batch_blocks) // 2
            queue = _dq([(batch_blocks[:mid0], 0), (batch_blocks[mid0:], mid0)])
            while queue:
                sub_blocks, offset = queue.popleft()
                # build prompt for sub_blocks
                sub_texts = []
                for j, sb in enumerate(sub_blocks):
                    sb_start = sb["start"]
                    sb_end = sb["end"]
                    sb_dur = sb_end - sb_start
                    dlg = sb.get("text", "").strip() or "(no dialogue)"
                    vis = sb.get("visual", "").strip() or "(not analyzed)"
                    cc = sb.get("cut_count", 0)
                    pp = sb.get("pause_points", [])
                    sub_texts.append(
                        f"--- BLOCK {j} ({_format_time(sb_start)}-{_format_time(sb_end)}, {sb_dur:.0f}s) ---\n"
                        f"Dialogue: {dlg}\n"
                        f"Visual: {vis}\n"
                        f"Cut count: {cc} (high = action)\n"
                        f"Dialogue pauses: {pp}"
                    )
                sub_prompt = batch_template.format(
                    movie_name=movie_title,
                    context_block=context_block,
                    blocks_text="\n\n".join(sub_texts),
                    min_duration=min_duration,
                    max_duration=max_duration,
                    pref_duration=(min_duration + max_duration) // 2,
                )
                sub_raw = None
                try:
                    sub_raw = call_llm(sub_prompt, max_tokens=_max_tokens_for(len(sub_blocks)))
                except Exception as e:
                    print(f"  ⚠️ Retry for section(s) {offset+1}-{offset+len(sub_blocks)}: "
                          f"still no response ({e})")
                    sub_raw = None
                if sub_raw:
                    sub_starts = [b["start"] for b in sub_blocks]
                    sub_clips = _parse_batch_response(sub_raw, sub_starts)
                else:
                    sub_clips = {}
                if not sub_clips and len(sub_blocks) > 1:
                    sm = len(sub_blocks) // 2
                    queue.append((sub_blocks[:sm], offset))
                    queue.append((sub_blocks[sm:], offset + sm))
                    print(f"  ↻ Still no usable answer for section(s) "
                          f"{offset+1}-{offset+len(sub_blocks)} — splitting "
                          "further and retrying")
                elif sub_clips:
                    for k, v in sub_clips.items():
                        merged.setdefault(k + offset, []).extend(v)
            if merged:
                batch_clips = merged
                print(f"  ✅ Recovered {sum(len(v) for v in batch_clips.values())} "
                      f"clip(s) from batch {batch_num} by asking one section "
                      "at a time")
            else:
                print(f"  ⚠️ Still nothing usable for batch {batch_num} even "
                      "one section at a time — those blocks will fall back "
                      "to automatic centering below")

        # Process each block in the batch
        for i, block in enumerate(batch_blocks):
            global_idx = batch_start + i
            block_start = block["start"]
            block_end = block["end"]
            block_dur = block_end - block_start
            dialogue = block.get("text", "").strip()

            # Get clips for this block (absolute timestamps from batch parser)
            block_clips = batch_clips.get(i, [])

            # Validate — batch clips have absolute timestamps, so pass
            # block_start=0. _validate_sub_clips() prints why anything gets
            # rejected — don't duplicate that here.
            valid_clips = _validate_sub_clips(block_clips, 0, block_end, block_dur,
                                              min_duration, max_duration) if block_clips else []
            for vc in valid_clips:
                vc["text"] = dialogue

            if len(valid_clips) == 0:
                if not block_clips:
                    print(f"  ℹ️ Section {global_idx+1}: the AI didn't "
                          "suggest any clips for it")
            else:
                print(f"  ✅ Section {global_idx+1}: {len(valid_clips)} "
                      "clip(s) accepted")

            # Fallback: if LLM returned nothing but block has dialogue, use smart centering
            if not valid_clips and dialogue:
                segments = [{"start": block_start, "end": block_end, "text": dialogue}]
                fb_start, fb_end = _find_best_window(segments, block_start, block_end, max_duration)
                if fb_start is not None:
                    print(f"    → No AI-picked clip survived for this "
                          f"section, so picking the {fb_start:.0f}-{fb_end:.0f}s "
                          "window with the most dialogue in it instead")
                    valid_clips.append({
                        "start": fb_start,
                        "end": fb_end,
                        "duration": fb_end - fb_start,
                        "text": dialogue,
                        "score": 5.0,
                        "title": movie_title[:40],
                    })

            all_sub_clips.extend(valid_clips)

        batch_start += len(batch_blocks)
    if not all_sub_clips:
        print("  Nothing survived — neither the AI nor the automatic "
              "fallback produced any usable clip. Stopping.")
        return None

    # Step 3: Sort by start time
    all_sub_clips.sort(key=lambda x: x["start"])

    # Step 4: Score threshold filter
    before = len(all_sub_clips)
    filtered = [c for c in all_sub_clips if c["score"] >= score_threshold]
    if not filtered:
        # If nothing passes threshold, keep top N clips anyway
        all_sub_clips.sort(key=lambda x: x["score"], reverse=True)
        filtered = all_sub_clips[:max(1, num_clips // 4)]
        print(f"  None of the {before} candidate clip(s) scored "
              f"{score_threshold}/10 or higher — keeping the "
              f"{len(filtered)} best-scored one(s) anyway rather than "
              "producing nothing")
    else:
        print(f"  Keeping clips scored {score_threshold}/10 or higher: "
              f"{before} candidate(s) → {len(filtered)} kept")

    # Step 5: Diversity filter — spreads the kept clips across the whole
    # movie instead of letting them clump in one part of it.
    before = len(filtered)
    filtered.sort(key=lambda x: x["score"], reverse=True)
    if total_duration > 0:
        filtered = _diversity_filter(filtered, num_clips, total_duration)
    else:
        filtered = filtered[:num_clips]
    print(f"  Spreading clips across the whole movie (top clip per "
          f"timeline segment, capped at {num_clips}): {before} → "
          f"{len(filtered)} clip(s)")

    # Step 6: Deduplication — two clips starting within min_gap of each
    # other are treated as the same moment; only the higher-scored one survives.
    before = len(filtered)
    filtered = _deduplicate_clips(filtered)
    print(f"  Removing clips that are basically the same moment (starting "
          f"too close together): {before} → {len(filtered)} clip(s)")

    # Step 7: Min duration expansion — safety net for a fallback clip that
    # ended up short (see _find_best_window); the AI-picked ones already
    # can't be shorter than min_duration by this point (_validate_sub_clips).
    short_before = [c for c in filtered if (c["end"] - c["start"]) < min_duration]
    filtered = _expand_short_clips(filtered, min_duration)
    if short_before:
        print(f"  Stretched {len(short_before)} clip(s) that came out "
              f"under {min_duration}s up to exactly {min_duration}s")

    # Step 8: Sort by start and resolve overlaps
    filtered.sort(key=lambda x: x["start"])
    for i in range(1, len(filtered)):
        prev = filtered[i-1]
        curr = filtered[i]
        if curr["start"] < prev["end"]:
            mid = (prev["end"] + curr["start"]) / 2
            if prev["end"] > mid:
                prev["end"] = mid
                prev["duration"] = prev["end"] - prev["start"]
            if curr["start"] < mid:
                curr["start"] = mid
                curr["duration"] = curr["end"] - curr["start"]
            print(f"  Two selected clips overlapped in time — split them "
                  f"at {_format_time(mid)} so they don't share any footage")

    elapsed = time.time() - total_start
    print(f"\n✓ Picked {len(filtered)} final clip(s) in {elapsed:.0f}s "
          f"({elapsed/60:.1f} min)")
    return filtered


# ---------------------------------------------------------------------------
# Diversity filter — spread selected clips across the movie timeline
# ---------------------------------------------------------------------------

def _diversity_filter(scenes, num_clips, total_duration, min_score=7.0):
    """Spread selected clips across the movie timeline.

    Divides the movie into num_clips segments and picks the best scene
    from each segment. If a segment has no qualifying scenes, fills
    from remaining (highest-score) scenes.

    Args:
        scenes: list of {start, end, score, ...} sorted by score desc
        num_clips: max number of clips to return
        total_duration: total movie duration in seconds
        min_score: minimum score for a scene to be a segment candidate (default 7.0)

    Returns:
        list of scenes sorted by start time, max num_clips entries
    """
    if len(scenes) <= num_clips:
        return scenes

    segment_dur = total_duration / num_clips
    selected = []
    used_indices = set()

    for seg_idx in range(num_clips):
        seg_start = seg_idx * segment_dur
        seg_end = (seg_idx + 1) * segment_dur
        # Find best (highest score) scene in this segment
        candidates = [
            (i, s) for i, s in enumerate(scenes)
            if seg_start <= s["start"] < seg_end and i not in used_indices
            and s["score"] >= min_score
        ]
        candidates.sort(key=lambda x: x[1]["score"], reverse=True)
        if candidates:
            best_idx, best_scene = candidates[0]
            selected.append(best_scene)
            used_indices.add(best_idx)

    # If we have fewer than num_clips, fill from remaining top-scored
    if len(selected) < num_clips:
        remaining = [
            s for i, s in enumerate(scenes) if i not in used_indices
        ]
        remaining.sort(key=lambda x: x["score"], reverse=True)
        while len(selected) < num_clips and remaining:
            selected.append(remaining.pop(0))

    selected.sort(key=lambda x: x["start"])
    return selected


def _deduplicate_clips(clips, min_gap=120.0):
    sorted_clips = sorted(clips, key=lambda x: x["start"])
    kept = []
    for clip in sorted_clips:
        if not kept:
            kept.append(clip)
            continue
        gap = clip["start"] - kept[-1]["start"]
        if gap < min_gap:
            if clip["score"] > kept[-1]["score"]:
                removed = kept.pop()
                print(f"🗑️ «{removed.get('title','')}» dropped — only "
                      f"{gap:.0f}s from «{clip.get('title','')}» (kept the "
                      "higher-scored one)")
                kept.append(clip)
            else:
                print(f"🗑️ «{clip.get('title','')}» dropped — only "
                      f"{gap:.0f}s from «{kept[-1].get('title','')}» (kept "
                      "the higher-scored one)")
        else:
            kept.append(clip)
    return kept


# ---------------------------------------------------------------------------
# Smart clip centering — find the best window by dialogue density
# ---------------------------------------------------------------------------

def _find_best_window(segments, scene_start, scene_end, window_dur):
    """Find the best `window_dur` window within [scene_start, scene_end].

    "Best" = the window with the highest word count (dialogue density).
    Falls back to the first window if no segments overlap the scene.

    Returns (new_start, new_end) or (None, None) if no adjustment needed.
    """
    # Filter segments overlapping this scene
    overlapping = [
        s for s in segments
        if s["start"] < scene_end and s["end"] > scene_start
    ]
    if not overlapping:
        return None, None  # no dialogue — keep original start

    scene_len = scene_end - scene_start
    window_dur = min(window_dur, scene_len)

    # Slide window across the scene, compute word count for each position
    best_start = scene_start
    best_words = -1

    # Step size = 1 second for precision
    step = 1.0
    max_start = scene_end - window_dur

    pos = scene_start
    while pos <= max_start:
        win_end = pos + window_dur
        # Count words in this window
        word_count = 0
        for s in overlapping:
            # Check if segment overlaps the window
            if s["start"] < win_end and s["end"] > pos:
                word_count += len(s["text"].split())
        if word_count > best_words:
            best_words = word_count
            best_start = pos
        pos += step

    new_start = best_start
    new_end = new_start + window_dur

    # Extend to natural boundaries:
    # - Start: nearest segment start (go back to where dialogue begins)
    for s in overlapping:
        if s["start"] < new_start and s["end"] > new_start:
            new_start = min(new_start, s["start"])
        elif s["start"] <= new_start < s["end"]:
            new_start = min(new_start, s["start"])

    # - End: nearest segment end (go forward to where dialogue ends)
    for s in overlapping:
        if s["start"] < new_end < s["end"]:
            new_end = max(new_end, s["end"])
        elif s["start"] >= new_end:
            # This segment starts after the window — might be part of the same scene
            # Don't extend past scene boundaries though
            pass

    # Clamp to scene boundaries
    new_start = max(scene_start, new_start)
    new_end = min(scene_end, new_end)

    # The natural-boundary extension above can push new_end past a subtitle
    # segment that runs well beyond the window — clamped only to scene_end,
    # a single long segment could blow well past window_dur (max_duration).
    # Trim back to the cap so a fallback clip can never violate the same
    # duration rule LLM-selected clips are held to.
    if new_end - new_start > window_dur:
        new_end = new_start + window_dur

    # Ensure at least some minimum duration
    if new_end - new_start < 5:
        new_start = scene_start
        new_end = min(scene_end, scene_start + window_dur)

    return new_start, new_end
