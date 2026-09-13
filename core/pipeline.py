"""
MovieShort AI — Pipeline orchestrator.
Connects FFmpeg clipping, subtitle generation, face tracking, and vertical crop.

The automatic movie pipeline is subtitle-only. External subtitle files are
parsed before clip selection and the resulting transcript JSON is used here
for final subtitle generation. Whisper is intentionally not used as a
fallback.
"""
import os
import re
import subprocess
from pathlib import Path

import config
from utils.ffmpeg_utils import (
    clip_video, embed_subtitles,
    convert_to_vertical, FFmpegError,
    pad_with_banners, blur_background,
    _detect_gpu_accel,
)
from core.subtitle import (
    generate_word_group_srt,
    load_segments_json,
    filter_segments_in_range,
)
from core.processor import apply_vertical_crop
from utils.font_manager import FONTS_DIR, ensure_font
from utils.clip_log import log, set_clip_label, clear_clip_label


def _resolve_font_style(options):
    """Build font_style dict from pipeline options (R7b-7).

    Priority:
      1. options["font_style"] if present (gui wired Editor controls already).
      2. subtitle_* flat keys (subtitle_font / subtitle_font_name etc.) — build dict + ensure_font.
      3. fallback to user_config.load() (same _get_font_style path as preview via gui/app.py helper).
    Shared path with render_full_preview: both ultimately call ffmpeg_utils._build_style_string.
    """
    if options.get("font_style"):
        return options["font_style"]
    has_subtitle_keys = any(k.startswith("subtitle_") for k in options)
    if has_subtitle_keys:
        font = options.get("subtitle_font")
        if not font:
            name = options.get("subtitle_font_name") or "Arial"
            try:
                font = ensure_font(name, FONTS_DIR)
            except Exception:
                font = name
        return {
            "font": font,
            "size": options.get("subtitle_size", 13),
            "color": options.get("subtitle_color", "&H00FFFFFF"),
            "outline": options.get("subtitle_outline", 1),
            "bold": options.get("subtitle_bold", True),
            "italic": options.get("subtitle_italic", False),
            "shadow": options.get("subtitle_shadow", False),
            "position_y": options.get("subtitle_position_y", 400),
        }
    try:
        from utils import user_config
        cfg = user_config.load()
        return {
            "font": cfg.get("subtitle_font", "Arial"),
            "size": cfg.get("subtitle_size", 13),
            "color": cfg.get("subtitle_color", "&H00FFFFFF"),
            "outline": cfg.get("subtitle_outline", 1),
            "bold": cfg.get("subtitle_bold", True),
            "italic": cfg.get("subtitle_italic", False),
            "shadow": cfg.get("subtitle_shadow", False),
            "position_y": cfg.get("subtitle_position_y", 400),
        }
    except Exception:
        return None


def _write_subtitles(segments, base_path):
    """Write word-group subtitles next to base_path (no extension).

    Always SRT (subtitle animations removed in v2.0, todo 47).
    """
    srt_path = base_path + ".srt"
    generate_word_group_srt(segments, srt_path)
    return srt_path


_FILENAME_ILLEGAL = r'[/\\:*?<>|]'
_YEAR_RE = r"(19\d\d|20\d\d)"


def _sanitize_name_part(value):
    """Make a string safe for a Windows filename part.

    '"' is DELETED (illegal in Windows names, no replacement),
    /\\:*?<>| become spaces, whitespace collapsed, stripped.
    """
    value = str(value or "").replace('"', "")
    value = re.sub(_FILENAME_ILLEGAL, " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _build_clip_name(movie_title, clip_title, start_time, video_path):
    """YouTube-Shorts style output filename (no extension).

    Format: 'Moment from {movie_clean} ({year}), {clip}'.
    Year is regex-extracted from movie_title and cut out of the clean title;
    no year part when movie_title has none. Falls back to the timestamp-based
    '{stem}_{start}' name when clip_title sanitizes to empty.
    R7b-9: no suffix.
    """
    safe_start = start_time.replace(":", "-")
    clip_clean = _sanitize_name_part(clip_title)
    if not clip_clean:
        return f"{Path(video_path).stem}_{safe_start}"

    movie_title = str(movie_title or "").strip()
    # strip empty parens like "Inception ( )" -> "Inception"
    movie_title = re.sub(r"\(\s*\)", "", movie_title).strip()
    movie_title = re.sub(r"\s+", " ", movie_title).strip()
    match = re.search(r"(19\d\d|20\d\d)", movie_title)
    if match:
        year = match.group(1)
        clean_title = re.sub(r"\s*\(?\s*(19|20)\d\d\s*\)?\s*", " ", movie_title).strip()
        clean_title = re.sub(r"\(\s*\)", "", clean_title).strip()
        clean_title = _sanitize_name_part(clean_title)
        movie_part = f"{clean_title} ({year})" if clean_title else f"({year})"
    else:
        clean_title = re.sub(r"\(\s*\)", "", movie_title).strip()
        movie_part = _sanitize_name_part(clean_title)

    if movie_part:
        name = f"Moment from {movie_part}, {clip_clean}"
    else:
        name = clip_clean
    return name[:200].strip()


def _time_to_seconds(hh_mm_ss: str) -> float:
    """Convert 'HH:MM:SS' or 'MM:SS' to seconds."""
    parts = hh_mm_ss.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


def process_clip(video_path, start_time, end_time, options=None, title="", log_label=None):
    """
    Process a single clip: cut → subtitle → scale → embed subtitles → pad → export.

    Args:
        video_path: path to source video
        start_time: "HH:MM:SS"
        end_time: "HH:MM:SS"
        options: dict with keys:
            - subtitles (bool): generate and embed subtitles (default True)
            - face_tracking (bool): apply face-tracking vertical crop (default True)
            - max_duration (int): max clip length in sec (default 60)
            - anti_copyright (bool): enable anti-copyright measures (default True)
            - blur_background (bool): enable blurred background (default True)
            - banner_top (int): top banner padding (default 300)
            - banner_bottom (int): bottom banner padding (default 300)
            - font_style (dict): subtitle font settings
            - transcript_path (str): path to parsed external-subtitle JSON
            - movie_title (str): movie name used in the output filename
        title: optional short clip title to include in output filename
        log_label: prefix (e.g. "Clip 2/4 · Hacking the system") stamped on
            every log line this call produces, including from functions it
            calls (apply_vertical_crop, ffmpeg steps). process_multiple()
            renders clips in parallel worker threads with no other way to
            tell whose output is whose in the interleaved console — see
            utils/clip_log.py.

    Returns:
        Path to the final output file, or None on failure.
    """
    if options is None:
        options = {}

    set_clip_label(log_label)

    subtitles_enabled = options.get("subtitles", True)
    face_tracking_enabled = options.get("face_tracking", True)
    anti_copyright = options.get("anti_copyright", config.DEFAULT_ANTI_COPYRIGHT)
    blur_enabled = options.get("blur_background", config.DEFAULT_BLUR_BACKGROUND)
    banner_top = options.get("banner_top", config.DEFAULT_BANNER_TOP)
    banner_bottom = options.get("banner_bottom", config.DEFAULT_BANNER_BOTTOM)
    font_style = _resolve_font_style(options)
    gpu_opts = _detect_gpu_accel()

    video_path = str(video_path)
    os.makedirs(config.TEMP_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    movie_title = options.get("movie_title", "")
    clip_name = _build_clip_name(movie_title, title, start_time, video_path)

    raw_clip = str(config.TEMP_DIR / f"{clip_name}_raw.mp4")
    clip_with_audio = str(config.TEMP_DIR / f"{clip_name}_audio.mp4")
    sub_path = str(config.TEMP_DIR / f"{clip_name}.srt")
    vertical_clip = str(config.TEMP_DIR / f"{clip_name}_vert.mp4")
    final_output = str(config.OUTPUT_DIR / f"{clip_name}.mp4")

    # Initialize subtitled_clip before try so finally can reference it safely
    subtitled_clip = vertical_clip

    try:
        # Step 1: Cut the segment
        log(f"[1/5] Cutting the {start_time}–{end_time} segment out of the source video...")
        clip_video(video_path, start_time, end_time, raw_clip, gpu_opts=gpu_opts)

        # Step 2: Generate subtitles from the external-subtitle transcript.
        #
        # IMPORTANT:
        # The automatic pipeline is subtitle-only.
        # Never transcribe the clip with Whisper here.
        if subtitles_enabled:
            transcript_path = options.get("transcript_path")

            if not transcript_path or not os.path.exists(transcript_path):
                raise RuntimeError(
                    "Missing subtitles: no parsed subtitle transcript "
                    "was provided for this clip."
                )

            log("[2/5] Slicing this clip's lines out of the movie's subtitle file...")

            start_sec = _time_to_seconds(start_time)
            end_sec = _time_to_seconds(end_time)

            all_segments = load_segments_json(transcript_path)

            clip_segments = filter_segments_in_range(
                all_segments,
                start_sec,
                end_sec,
            )

            if not clip_segments:
                raise RuntimeError(
                    "Missing subtitles: selected clip contains no "
                    "subtitle cues."
                )

            log(f"      Found {len(clip_segments)} subtitle line(s) in "
                "this time range")

            sub_path = _write_subtitles(
                clip_segments,
                str(config.TEMP_DIR / clip_name),
            )

            # External subtitles are already attached to the source timeline,
            # so the original cut remains the clip with its audio.
            clip_with_audio = raw_clip
        else:
            # Automatic movie processing is subtitle-only. Do not allow a
            # subtitle-disabled path to silently bypass the required transcript.
            raise RuntimeError(
                "Missing subtitles: subtitles are required for automatic "
                "movie processing."
            )

        # Step 3: Scale to fit content area (1080 × content_h, preserves aspect ratio)
        if face_tracking_enabled:
            log("[3/5] Converting to vertical 9:16 — looking for faces/people to "
                "keep them centered in frame...")
            apply_vertical_crop(
                clip_with_audio,
                vertical_clip,
                anti_copyright=anti_copyright,
                banner_top=banner_top,
                banner_bottom=banner_bottom,
            )
        else:
            log("[3/5] Converting to vertical 9:16 (plain center crop — "
                "smart centering is off)...")
            convert_to_vertical(
                clip_with_audio,
                vertical_clip,
                anti_copyright=anti_copyright,
                banner_top=banner_top,
                banner_bottom=banner_bottom,
                gpu_opts=gpu_opts,
            )

        # Step 4: Embed subtitles (on content-area video, before banner padding)
        if os.path.exists(sub_path) and os.path.getsize(sub_path) > 0:
            log("[4/5] Burning the subtitles into the video...")
            subtitled_clip = str(config.TEMP_DIR / f"{clip_name}_subs.mp4")
            embed_subtitles(
                vertical_clip,
                sub_path,
                subtitled_clip,
                font_style=font_style,
                banner_top=banner_top,
                banner_bottom=banner_bottom,
                gpu_opts=gpu_opts,
                fontsdir=os.path.abspath(FONTS_DIR),
            )
        else:
            raise RuntimeError(
                "Missing subtitles: subtitle SRT could not be generated."
            )

        # Step 5: Blurred background → full 9:16 output.
        # background_source=vertical_clip (pre-subtitle) so the blurred
        # backdrop never contains a blurred "ghost" of the subtitles — only
        # the sharp foreground (subtitled_clip) shows them.
        if blur_enabled:
            log("[5/5] Filling the top/bottom bars with a blurred copy of the "
                "video and exporting the final file...")
            blur_background(
                subtitled_clip,
                final_output,
                enabled=True,
                banner_top=banner_top,
                banner_bottom=banner_bottom,
                gpu_opts=gpu_opts,
                background_source=vertical_clip,
            )
        else:
            log("[5/5] Padding the top/bottom bars with plain black and "
                "exporting the final file (blur is off)...")
            # If blur disabled, pad the content-area video to full 9:16
            pad_with_banners(
                subtitled_clip,
                final_output,
                banner_top=banner_top,
                banner_bottom=banner_bottom,
                gpu_opts=gpu_opts,
            )

        log(f"✅ Done → {os.path.basename(final_output)}")
        return final_output

    except FFmpegError as e:
        log(f"❌ FFmpeg failed while rendering this clip: {e}")
        return None
    except RuntimeError as e:
        log(f"❌ {e}")
        return None
    except subprocess.TimeoutExpired:
        log("❌ Rendering this clip took too long and was aborted (timeout).")
        return None
    except Exception as e:
        log(f"❌ Unexpected error while rendering this clip: {e}")
        return None
    finally:
        # Cleanup temp files
        cleanup_files = [raw_clip, clip_with_audio, sub_path, vertical_clip]
        if subtitled_clip != vertical_clip:
            cleanup_files.append(subtitled_clip)
        for f in cleanup_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass
        clear_clip_label()


def process_multiple(video_path, timestamps_list, options=None, titles=None, max_workers=2):
    """
    Process multiple clips from one video using parallel workers.

    Args:
        video_path: path to source video
        timestamps_list: list of (start_time, end_time) tuples
        options: dict of options (passed to process_clip)
        titles: optional list of scene titles (same length as timestamps_list)
        max_workers: max parallel workers (default 2, 1 = sequential)

    Returns:
        List of paths to output files.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    total = len(timestamps_list)
    if total == 0:
        return []

    if titles is None:
        titles = [""] * total

    def _label_for(i):
        """'Clip 2/4 · Hacking the system' — stamped on every line that
        clip's processing produces (see process_clip's log_label)."""
        base = f"Clip {i + 1}/{total}"
        return f"{base} · {titles[i]}" if titles[i] else base

    if max_workers <= 1:
        # Sequential mode
        results = []
        for i, (start, end) in enumerate(timestamps_list):
            print(f"\n--- {_label_for(i)}: {start} - {end} ---")
            result = process_clip(video_path, start, end, options,
                                   title=titles[i], log_label=_label_for(i))
            results.append(result)
        done = sum(1 for r in results if r is not None)
        print(f"\nDone: {done}/{total} clip(s) processed successfully")
        return results

    # Parallel mode. Rendering is CPU/GPU-bound (ffmpeg, face detection) —
    # max_workers clips are in flight at once, so their console output
    # interleaves. Every line from a clip's processing is prefixed with its
    # "Clip N/total · title" label (see utils/clip_log.py) precisely so this
    # interleaving stays readable.
    print(f"\nRendering {total} clip(s), {max_workers} at a time...")
    results = [None] * total

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for i, (start, end) in enumerate(timestamps_list):
            future = executor.submit(
                process_clip,
                video_path,
                start,
                end,
                options,
                title=titles[i],
                log_label=_label_for(i),
            )
            future_to_idx[future] = i

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results[idx] = result
                status = "finished ✅" if result else "failed ❌"
                print(f"  {_label_for(idx)}: {status}")
            except Exception as e:
                print(f"  {_label_for(idx)}: failed ❌ — {e}")
                results[idx] = None

    done = sum(1 for r in results if r is not None)
    print(f"\nDone: {done}/{total} clip(s) processed successfully")
    return results
