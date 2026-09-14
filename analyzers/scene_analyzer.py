"""
MovieShort AI — Scene detection and external subtitle analysis.

The automatic movie pipeline uses:
    1. PySceneDetect for visual scene boundaries.
    2. External subtitle files for dialogue/action text.

Whisper is intentionally NOT used here.
"""

import hashlib
import os
import subprocess
import time as time_module
from pathlib import Path

from scenedetect import open_video, SceneManager, ContentDetector

import config
from core.subtitle import (
    find_external_subtitle,
    load_external_subtitles,
    save_segments_json,
)
from utils import fmt_duration as _fmt_duration
from utils import get_video_basename


# ---------------------------------------------------------------------------
# Scene detection
# ---------------------------------------------------------------------------

def detect_scenes(video_path, threshold=None):
    """
    Detect scenes in a video using PySceneDetect ContentDetector.

    Uses config.SCENE_FRAME_SKIP to reduce processing time.

    Returns:
        List of scenes:

        [
            {
                "start": float,
                "end": float,
                "duration": float,
            },
            ...
        ]
    """

    if threshold is None:
        threshold = config.SCENE_THRESHOLD

    frame_skip = getattr(
        config,
        "SCENE_FRAME_SKIP",
        2,
    )

    print("=" * 50)
    print("SCENE DETECTION")
    print("=" * 50)
    print("Finding where the camera cuts to a new shot, so later steps "
          "know where one 'scene' ends and the next begins.")

    print(
        f"Detector: ContentDetector "
        f"(threshold={threshold} — lower catches more/smaller cuts)"
    )

    print(
        f"Frame skip: {frame_skip} "
        f"(checks every {frame_skip + 1}th frame, to go faster)"
    )

    print()

    # ------------------------------------------------------------------
    # Open video and collect basic information.
    # ------------------------------------------------------------------
    video = open_video(
        str(video_path)
    )

    duration_sec = video.duration.get_seconds()
    fps = video.frame_rate

    total_frames_est = int(
        duration_sec * fps
    )

    frames_to_process = max(
        1,
        total_frames_est // (frame_skip + 1),
    )

    print(
        f"Video: {total_frames_est} frames "
        f"@ {fps:.2f} fps"
    )

    print(
        f"Duration: {duration_sec:.0f}s "
        f"({duration_sec / 60:.1f}min)"
    )

    print(
        f"Will check ~{frames_to_process} frames for cuts. This step reads "
        "through the whole video, so it typically takes several minutes on "
        "a long movie — a live ETA will appear below once scanning starts."
    )

    print()

    # ------------------------------------------------------------------
    # Configure PySceneDetect.
    # ------------------------------------------------------------------
    scene_manager = SceneManager()

    scene_manager.add_detector(
        ContentDetector(
            threshold=threshold
        )
    )

    scan_start = time_module.time()

    check_interval = max(
        500,
        frames_to_process // 40,
    )

    done_flag = False

    def progress_callback(
        frame,
        frame_number,
    ):
        """
        Display periodic scene-detection progress.

        PySceneDetect versions may provide either an integer or a
        FrameTimecode object as frame_number, so we explicitly convert it.
        """

        nonlocal done_flag

        try:
            frame_num = int(
                frame_number
            )
        except (
            TypeError,
            ValueError,
        ):
            return

        if frame_num <= 0:
            return

        if frame_num % check_interval != 0:
            return

        elapsed = (
            time_module.time()
            - scan_start
        )

        frames_done = min(
            frame_num,
            frames_to_process,
        )

        pct = min(
            100.0,
            (
                frames_done
                / frames_to_process
                * 100
            ),
        )

        # Avoid repeated 100% messages.
        if pct >= 100.0 and done_flag:
            return

        rate = (
            frames_done
            / max(elapsed, 0.1)
        )

        remaining_frames = max(
            0,
            frames_to_process
            - frames_done,
        )

        remaining = (
            remaining_frames
            / max(rate, 0.1)
        )

        print(
            f"  Scan: {pct:.0f}% "
            f"({frames_done}/{frames_to_process}) "
            f"elapsed: {_fmt_duration(elapsed)} "
            f"ETA: {_fmt_duration(remaining)}"
        )

        if pct >= 100.0:
            done_flag = True

    # ------------------------------------------------------------------
    # Run scene detection.
    # ------------------------------------------------------------------
    print("Starting frame scan...")

    scene_manager.detect_scenes(
        video=video,
        frame_skip=frame_skip,
        callback=progress_callback,
    )

    elapsed = (
        time_module.time()
        - scan_start
    )

    print(
        f"  Scan complete in "
        f"{_fmt_duration(elapsed)}"
    )

    # ------------------------------------------------------------------
    # Convert PySceneDetect results to our normalized structure.
    # ------------------------------------------------------------------
    scene_list = (
        scene_manager.get_scene_list()
    )

    scenes = []

    for (
        start_timecode,
        end_timecode,
    ) in scene_list:

        start_sec = (
            start_timecode.get_seconds()
        )

        end_sec = (
            end_timecode.get_seconds()
        )

        scenes.append(
            {
                "start": start_sec,
                "end": end_sec,
                "duration": (
                    end_sec
                    - start_sec
                ),
            }
        )

    print(
        f"  Found {len(scenes)} camera cuts"
    )

    return scenes


# ---------------------------------------------------------------------------
# Scene merging
# ---------------------------------------------------------------------------

def merge_short_scenes(
    scenes,
    min_duration=None,
    max_duration=None,
):
    """
    Merge short scenes with neighbouring scenes.

    Rules:

    1. Start a buffer with the first raw scene.
    2. If the buffer is shorter than min_duration, merge the next scene
       into it.
    3. Once the buffer reaches min_duration:
       - If the next scene is also long enough, finalize the buffer.
       - If the next scene is short, start a new short buffer.
    4. If the buffer reaches max_duration, force-finalize it.

    This avoids the old snowball effect where short scenes could cause
    already-good scenes to become unnecessarily long.
    """

    if min_duration is None:
        min_duration = config.MIN_SCENE_DURATION

    if max_duration is None:
        max_duration = config.MAX_MERGE_DURATION

    if not scenes:
        return []

    merged = []

    buffer = scenes[0].copy()

    for scene in scenes[1:]:
        current = scene.copy()

        # ---------------------------------------------------------------
        # Buffer already reached maximum size.
        # ---------------------------------------------------------------
        if buffer["duration"] >= max_duration:
            merged.append(buffer)
            buffer = current
            continue

        # ---------------------------------------------------------------
        # Buffer is too short.
        # Merge the current scene into it.
        # ---------------------------------------------------------------
        if buffer["duration"] < min_duration:
            buffer["end"] = current["end"]

            buffer["duration"] = (
                buffer["end"]
                - buffer["start"]
            )

        # ---------------------------------------------------------------
        # Current scene is short.
        # Do not extend an already-good buffer.
        # ---------------------------------------------------------------
        elif current["duration"] < min_duration:
            merged.append(buffer)
            buffer = current

        # ---------------------------------------------------------------
        # Both scenes are long enough.
        # Start a new buffer.
        # ---------------------------------------------------------------
        else:
            merged.append(buffer)
            buffer = current

    if buffer is not None:
        merged.append(buffer)

    print(
        f"  Merged cuts shorter than {min_duration:.0f}s into their "
        f"neighbors (too short to be their own scene): {len(merged)} "
        "scene(s) left"
    )

    return merged


# ---------------------------------------------------------------------------
# Scene + external subtitle analysis
# ---------------------------------------------------------------------------

def detect_and_transcribe(
    video_path,
    language=None,
    sdh_subtitle_path=None,
):
    """
    Detect scenes and map an external SDH subtitle file onto those scenes.

    IMPORTANT:
        Whisper is intentionally NOT used.

    An SDH (Subtitles for the Deaf and Hard-of-hearing) subtitle file is
    mandatory here — not the plain dialogue-only subtitle file. SDH cues
    include bracketed sound/action cues ([gunshot], [door slams]) and
    speaker labels that plain subtitles omit, which gives the AI real
    signal about what's happening even in dialogue-free stretches. The
    plain subtitle file is used elsewhere, only for burning captions into
    the final rendered clip — SDH tags like "[grunting]" should never end
    up as a visible caption.

    Supported subtitle formats:

        .srt
        .ass
        .ssa
        .vtt

    Args:
        video_path:
            Source movie/video.

        language:
            Kept for API compatibility with the existing pipeline.
            It is not used for transcription because transcription is
            performed from the supplied external subtitle file.

        sdh_subtitle_path:
            Required explicit path to the SDH subtitle file. There is no
            fallback/auto-discovery — a missing or invalid path always
            raises rather than silently guessing a file next to the video.

    Returns:
        List of scene blocks:

        [
            {
                "start": float,
                "end": float,
                "duration": float,
                "text": str,
                "pause_points": list,
                "cut_count": int,
                "audio_peaks": dict,
            },
            ...
        ]

    Raises:
        RuntimeError:
            If no external subtitle file can be found or parsed.
    """

    # ------------------------------------------------------------------
    # 1. Resolve external subtitles FIRST.
    #
    # This happens before expensive scene detection so that a movie
    # without subtitles fails immediately instead of spending several
    # minutes analyzing the video.
    # ------------------------------------------------------------------
    resolved_sdh_subtitle_path = find_external_subtitle(
        video_path,
        explicit_path=sdh_subtitle_path,
    )

    if not resolved_sdh_subtitle_path:
        raise RuntimeError(
            "Missing subtitles: no SDH subtitle file (.srt, .ass, .ssa "
            "or .vtt) was provided for this movie."
        )

    sdh_subtitle_path = resolved_sdh_subtitle_path

    print()
    print("=" * 50)
    print("SUBTITLE INPUT")
    print("=" * 50)

    print(
        f"  Using subtitles: "
        f"{os.path.basename(sdh_subtitle_path)}"
    )

    # ------------------------------------------------------------------
    # 2. Parse external subtitles.
    # ------------------------------------------------------------------
    try:
        subtitle_segments = (
            load_external_subtitles(
                sdh_subtitle_path
            )
        )

    except Exception as error:
        raise RuntimeError(
            "Missing subtitles / subtitle parsing "
            f"failed: {error}"
        ) from error

    if not subtitle_segments:
        raise RuntimeError(
            "Missing subtitles: "
            f"{os.path.basename(sdh_subtitle_path)} "
            "contains no usable subtitle cues."
        )

    print(
        f"  ✓ Loaded "
        f"{len(subtitle_segments)} subtitle cues"
    )

    # ------------------------------------------------------------------
    # 3. Detect visual scenes.
    # ------------------------------------------------------------------
    scenes = detect_scenes(
        video_path
    )

    merged = merge_short_scenes(
        scenes
    )

    # ------------------------------------------------------------------
    # Count raw visual cuts contained in each merged block.
    # ------------------------------------------------------------------
    for block in merged:
        block["cut_count"] = sum(
            1
            for raw_scene in scenes
            if (
                raw_scene["start"]
                >= block["start"]
            )
            and (
                raw_scene["end"]
                <= block["end"]
            )
        )

    # ------------------------------------------------------------------
    # If PySceneDetect somehow finds nothing, use the whole movie as
    # one block.
    # ------------------------------------------------------------------
    if not merged:
        print(
            "  No camera cuts were detected at all (unusual) — "
            "treating the entire video as one single scene"
        )

        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default="
                    "noprint_wrappers=1:"
                    "nokey=1",
                    str(video_path),
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )

            duration = float(
                result.stdout.strip()
            )

        except Exception:
            # If ffprobe is unavailable, the last subtitle cue gives us
            # a reasonable fallback duration.
            duration = (
                subtitle_segments[-1]["end"]
                if subtitle_segments
                else 0
            )

        merged = [
            {
                "start": 0,
                "end": duration,
                "duration": duration,
                "cut_count": 0,
            }
        ]

    # ------------------------------------------------------------------
    # 4. Map external subtitle cues to each visual scene.
    #
    # There is deliberately NO:
    #
    #   - audio extraction
    #   - Whisper
    #   - speech recognition
    #   - audio RMS calculation
    #
    # The subtitle file is the source of dialogue/action information.
    # ------------------------------------------------------------------
    results = []

    for index, scene in enumerate(
        merged
    ):
        scene_start = scene["start"]
        scene_end = scene["end"]

        # Find subtitle cues that overlap this scene.
        overlapping = [
            segment
            for segment in subtitle_segments
            if (
                segment["start"]
                < scene_end
            )
            and (
                segment["end"]
                > scene_start
            )
        ]

        overlapping.sort(
            key=lambda segment: (
                segment["start"],
                segment["end"],
            )
        )

        # Combine subtitle text for LLM analysis.
        text_parts = [
            segment["text"].strip()
            for segment in overlapping
            if segment.get(
                "text",
                "",
            ).strip()
        ]

        block = {
            "start": scene_start,
            "end": scene_end,
            "duration": (
                scene_end
                - scene_start
            ),
            "text": " ".join(
                text_parts
            ),
            "pause_points": [],
            "cut_count": scene.get(
                "cut_count",
                0,
            ),

            # Subtitle-only mode does not calculate audio RMS.
            #
            # IMPORTANT:
            # silence_ratio is 0 rather than 1 because the absence of
            # audio analysis must NOT cause the LLM/filtering stage to
            # classify every subtitle block as silent.
            "audio_peaks": {
                "peak_rms": 0.0,
                "loud_peak_count": 0,
                "silence_ratio": 0.0,
            },
        }

        # --------------------------------------------------------------
        # Find meaningful gaps between subtitle cues.
        #
        # These gaps become candidate natural dialogue boundaries for
        # later clip selection.
        # --------------------------------------------------------------
        for cue_index in range(
            len(overlapping) - 1
        ):
            current = overlapping[
                cue_index
            ]

            following = overlapping[
                cue_index + 1
            ]

            gap = (
                following["start"]
                - current["end"]
            )

            # A gap over 0.8 sec is a useful candidate boundary.
            if gap > 0.8:
                block[
                    "pause_points"
                ].append(
                    round(
                        current["end"],
                        3,
                    )
                )

        results.append(block)

        if (
            index + 1
        ) % 20 == 0:
            print(
                "  Matched dialogue to "
                f"{index + 1}/"
                f"{len(merged)} scenes so far"
            )

    # ------------------------------------------------------------------
    # 5. Save the parsed subtitle transcript to the existing cache.
    #
    # The cache filename includes the subtitle modification time, so
    # replacing/editing the subtitle file automatically creates a new
    # cache instead of accidentally reusing stale subtitle data.
    # ------------------------------------------------------------------
    try:
        video_basename = get_video_basename(
            video_path
        )

        hash_input = (
            f"{video_path}_"
            f"{sdh_subtitle_path}_"
            f"{os.path.getmtime(sdh_subtitle_path)}"
        )

        file_hash = hashlib.md5(
            hash_input.encode()
        ).hexdigest()[:8]

        cache_path = (
            Path(config.CACHE_DIR)
            / (
                f"full_transcript_sdh_"
                f"{video_basename}_"
                f"{file_hash}.json"
            )
        )

        os.makedirs(
            config.CACHE_DIR,
            exist_ok=True,
        )

        save_segments_json(
            subtitle_segments,
            cache_path,
        )

        print(
            "  Saved the parsed subtitles for reuse when rendering the "
            f"final clips: {cache_path.name}"
        )

    except Exception as error:
        # Cache failure should not destroy an otherwise valid analysis.
        print(
            "  ⚠ Couldn't save the parsed-subtitle cache (analysis will "
            f"still continue, but it'll have to be re-parsed later): {error}"
        )

    print(
        f"  ✓ Matched dialogue to all {len(results)} scenes — ready for "
        "the AI to pick the best ones"
    )

    return results