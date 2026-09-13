"""
MovieShort AI — External subtitle loading and subtitle generation module.

The automatic movie pipeline uses external subtitle files only.

Supported external subtitle formats:
    - SRT
    - ASS
    - SSA
    - VTT

Whisper is intentionally NOT used here.
"""

import html
import json
import re
from pathlib import Path

from utils.clip_log import log


# ---------------------------------------------------------------------------
# SRT / subtitle generation
# ---------------------------------------------------------------------------

def generate_srt(segments, output_path):
    """
    Convert subtitle segments to an SRT file.

    Args:
        segments: list of subtitle dictionaries containing:
            {
                "start": float,
                "end": float,
                "text": str,
                "words": list
            }
        output_path: destination .srt path
    """
    lines = []

    for i, seg in enumerate(segments, 1):
        start = _format_srt_time(seg["start"])
        end = _format_srt_time(seg["end"])
        text = seg.get("text", "").strip()

        if not text:
            continue

        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")

    output_path = str(output_path)

    with open(output_path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines))

    print(f"SRT saved: {output_path}")


def _iter_word_groups(segments, max_chars=150):
    """
    Yield subtitle groups as:

        (start, end, text)

    External subtitles normally do not contain word-level timestamps.
    In that case, the entire subtitle cue is used as the timing source.

    If word timestamps are present, they are used for more precise timing.
    """

    for seg in segments:
        words = seg.get("words", [])

        # ---------------------------------------------------------------
        # External subtitle fallback:
        # no word timestamps, so use the subtitle cue timing directly.
        # ---------------------------------------------------------------
        if not words:
            text = seg.get("text", "").strip()

            if not text:
                continue

            # Keep long subtitle cues readable by splitting them into
            # chunks. Timing remains tied to the original subtitle cue.
            while text:
                chunk = text[:max_chars]
                text = text[max_chars:]

                yield (
                    seg["start"],
                    seg["end"],
                    chunk,
                )

            continue

        # ---------------------------------------------------------------
        # Word-timestamp mode.
        # This is kept for compatibility with cached/manual segment data.
        # ---------------------------------------------------------------
        groups = []
        current_group = []
        current_len = 0

        for word in words:
            word_text = word.get("word", "").strip()

            if not word_text:
                continue

            added_len = len(word_text) + (1 if current_group else 0)

            if (
                current_group
                and current_len + added_len > max_chars
            ):
                groups.append(current_group)
                current_group = [word]
                current_len = len(word_text)
            else:
                current_group.append(word)
                current_len += added_len

        if current_group:
            groups.append(current_group)

        for group in groups:
            group_start = group[0].get(
                "start",
                seg["start"],
            )

            group_end = group[-1].get(
                "end",
                seg["end"],
            )

            group_text = " ".join(
                word.get("word", "").strip()
                for word in group
                if word.get("word", "").strip()
            )

            if not group_text:
                continue

            yield (
                group_start,
                group_end,
                group_text,
            )


def generate_word_group_srt(
    segments,
    output_path,
    max_chars=150,
):
    """
    Generate an SRT using subtitle/word groups.

    External subtitles normally have no word timestamps, so each subtitle
    cue is used as the timing source.

    Returns:
        Number of generated SRT entries.
    """

    lines = []
    index = 0

    for group_start, group_end, group_text in _iter_word_groups(
        segments,
        max_chars,
    ):
        index += 1

        lines.append(str(index))
        lines.append(
            f"{_format_srt_time(group_start)} --> "
            f"{_format_srt_time(group_end)}"
        )
        lines.append(group_text)
        lines.append("")

    output_path = str(output_path)

    with open(output_path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines))

    log(
        f"      Wrote this clip's subtitle file "
        f"({index} caption(s)): {Path(output_path).name}"
    )

    return index


# ---------------------------------------------------------------------------
# Transcript JSON cache
# ---------------------------------------------------------------------------

def save_segments_json(segments, output_path):
    """
    Save parsed external subtitle segments to JSON.

    This cache is reused by the movie analysis and clip-rendering stages.
    """

    output_path = str(output_path)

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            segments,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Segments JSON saved: "
        f"{output_path} ({len(segments)} segments)"
    )


def load_segments_json(input_path):
    """
    Load parsed subtitle segments from JSON.
    """

    input_path = str(input_path)

    with open(
        input_path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def filter_segments_in_range(
    all_segments,
    start_sec,
    end_sec,
):
    """
    Return subtitle segments overlapping [start_sec, end_sec].

    Timestamps are shifted so that start_sec becomes 0.

    This is used when creating subtitles for an individual Short.
    """

    filtered = []

    for segment in all_segments:
        seg_start = segment["start"]
        seg_end = segment["end"]

        # No overlap with requested clip.
        if seg_start >= end_sec or seg_end <= start_sec:
            continue

        clipped_start = max(
            0,
            seg_start - start_sec,
        )

        clipped_end = min(
            end_sec - start_sec,
            seg_end - start_sec,
        )

        new_segment = {
            "start": clipped_start,
            "end": clipped_end,
            "text": segment.get("text", ""),
            "words": [],
        }

        # Preserve word timestamps if the source happens to contain them.
        for word in segment.get("words", []):
            word_start = word.get(
                "start",
                seg_start,
            )

            word_end = word.get(
                "end",
                seg_end,
            )

            if word_start >= end_sec or word_end <= start_sec:
                continue

            new_segment["words"].append(
                {
                    "start": max(
                        0,
                        word_start - start_sec,
                    ),
                    "end": min(
                        end_sec - start_sec,
                        word_end - start_sec,
                    ),
                    "word": word.get("word", ""),
                    "probability": word.get(
                        "probability",
                        0,
                    ),
                }
            )

        filtered.append(new_segment)

    return filtered


# ---------------------------------------------------------------------------
# External subtitle discovery
# ---------------------------------------------------------------------------

def find_external_subtitle(
    video_path,
    explicit_path=None,
):
    """
    Find an external subtitle file for a movie.

    Priority:

    1. Explicit subtitle path supplied by the user.
    2. Exact movie filename:
           movie.srt
           movie.ass
           movie.ssa
           movie.vtt
    3. Common English suffixes:
           movie.en.srt
           movie.eng.srt
           movie.english.srt
           movie.en-US.srt
           movie.en-GB.srt

    Returns:
        str path to subtitle file, or None.
    """

    video_path = Path(video_path)

    extensions = [
        ".srt",
        ".ass",
        ".ssa",
        ".vtt",
    ]

    # ------------------------------------------------------------------
    # 1. Explicit subtitle path
    # ------------------------------------------------------------------
    if explicit_path:
        explicit = Path(str(explicit_path))

        if explicit.exists() and explicit.is_file():
            return str(explicit)

    # ------------------------------------------------------------------
    # 2. Exact movie filename
    # ------------------------------------------------------------------
    for extension in extensions:
        candidate = video_path.with_suffix(extension)

        if candidate.exists() and candidate.is_file():
            return str(candidate)

    # ------------------------------------------------------------------
    # 3. Common language suffixes
    # ------------------------------------------------------------------
    language_suffixes = [
        ".en",
        ".eng",
        ".english",
        ".en-US",
        ".en-GB",
    ]

    for suffix in language_suffixes:
        for extension in extensions:
            candidate = video_path.with_name(
                video_path.stem
                + suffix
                + extension
            )

            if candidate.exists() and candidate.is_file():
                return str(candidate)

    return None


# ---------------------------------------------------------------------------
# External subtitle reading
# ---------------------------------------------------------------------------

def _read_subtitle_file(path):
    """
    Read subtitle text using several common encodings.

    UTF-8 is preferred, followed by common Windows encodings.
    """

    path = Path(path)

    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "cp1251",
        "latin-1",
    ]

    last_error = None

    for encoding in encodings:
        try:
            with open(
                path,
                "r",
                encoding=encoding,
            ) as f:
                return f.read()

        except UnicodeDecodeError as error:
            last_error = error

        except OSError:
            raise

    raise UnicodeDecodeError(
        "subtitle",
        b"",
        0,
        1,
        (
            f"Could not decode subtitle file: "
            f"{path} ({last_error})"
        ),
    )


def _parse_subtitle_timestamp(value):
    """
    Parse a subtitle timestamp into seconds.

    Supported examples:

        00:01:23,456
        00:01:23.456
        0:01:23.45
        01:23.456
        01:23.45
    """

    value = value.strip().replace(",", ".")

    parts = value.split(":")

    try:
        if len(parts) == 3:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])

            return (
                hours * 3600
                + minutes * 60
                + seconds
            )

        if len(parts) == 2:
            minutes = float(parts[0])
            seconds = float(parts[1])

            return minutes * 60 + seconds

        return float(value)

    except ValueError:
        return None


def _clean_subtitle_text(text):
    """
    Clean subtitle formatting while preserving meaningful dialogue
    and sound/action cues.

    Examples that are intentionally preserved:

        [door slams]
        (grunts)
        ♪ dramatic music ♪
        [phone ringing]

    Removed:

        HTML formatting tags
        ASS override tags
        ASS line-break markers
        excessive whitespace
    """

    text = html.unescape(text)

    # ---------------------------------------------------------------
    # ASS formatting overrides.
    #
    # Examples:
    #   {\i1}
    #   {\an8}
    #   {\pos(100,200)}
    # ---------------------------------------------------------------
    text = re.sub(
        r"\{[^}]*\}",
        "",
        text,
    )

    # ---------------------------------------------------------------
    # HTML subtitle formatting.
    #
    # Examples:
    #   <i>Hello</i>
    #   <b>Hello</b>
    # ---------------------------------------------------------------
    text = re.sub(
        r"<[^>]+>",
        "",
        text,
    )

    # ASS explicit line break marker.
    text = text.replace(
        r"\N",
        "\n",
    )

    # ASS secondary line-break marker.
    text = text.replace(
        r"\n",
        "\n",
    )

    # Normalize whitespace while keeping meaningful lines.
    lines = []

    for line in text.splitlines():
        line = re.sub(
            r"[ \t]+",
            " ",
            line,
        ).strip()

        if line:
            lines.append(line)

    return " ".join(lines).strip()


# ---------------------------------------------------------------------------
# SRT parser
# ---------------------------------------------------------------------------

def _parse_srt(text):
    """
    Parse SRT subtitles.

    Returns the normalized MovieShort segment structure:

        {
            "start": float,
            "end": float,
            "text": str,
            "words": []
        }
    """

    segments = []

    # Normalize Windows/Mac line endings.
    text = text.replace(
        "\r\n",
        "\n",
    ).replace(
        "\r",
        "\n",
    )

    # Split subtitle entries.
    entries = re.split(
        r"\n\s*\n",
        text,
    )

    timestamp_re = re.compile(
        r"^\s*"
        r"(\d{1,2}:\d{2}:\d{2}[,.]\d+)"
        r"\s*-->\s*"
        r"(\d{1,2}:\d{2}:\d{2}[,.]\d+)"
    )

    for entry in entries:
        lines = [
            line.strip()
            for line in entry.split("\n")
            if line.strip()
        ]

        if not lines:
            continue

        timestamp_index = None

        for index, line in enumerate(lines):
            if timestamp_re.match(line):
                timestamp_index = index
                break

        if timestamp_index is None:
            continue

        match = timestamp_re.match(
            lines[timestamp_index]
        )

        start = _parse_subtitle_timestamp(
            match.group(1)
        )

        end = _parse_subtitle_timestamp(
            match.group(2)
        )

        if (
            start is None
            or end is None
            or end <= start
        ):
            continue

        subtitle_text = " ".join(
            lines[timestamp_index + 1:]
        )

        subtitle_text = _clean_subtitle_text(
            subtitle_text
        )

        if not subtitle_text:
            continue

        segments.append(
            {
                "start": start,
                "end": end,
                "text": subtitle_text,
                "words": [],
            }
        )

    return segments


# ---------------------------------------------------------------------------
# ASS / SSA parser
# ---------------------------------------------------------------------------

def _parse_ass(text):
    """
    Parse ASS/SSA dialogue events.

    ASS timestamps normally use:

        H:MM:SS.cc

    The [Events] section is located automatically.

    Dialogue fields are parsed according to the ASS Format line when
    available, while also supporting the standard 10-field layout.
    """

    segments = []

    in_events = False
    format_fields = None

    normalized_text = text.replace(
        "\r\n",
        "\n",
    ).replace(
        "\r",
        "\n",
    )

    for raw_line in normalized_text.split("\n"):
        line = raw_line.strip()

        if not line:
            continue

        # ---------------------------------------------------------------
        # Section header
        # ---------------------------------------------------------------
        if line.startswith("["):
            in_events = (
                line.lower() == "[events]"
            )
            format_fields = None
            continue

        if not in_events:
            continue

        # ---------------------------------------------------------------
        # ASS event format declaration
        # ---------------------------------------------------------------
        if line.lower().startswith("format:"):
            format_fields = [
                field.strip().lower()
                for field in line.split(
                    ":",
                    1,
                )[1].split(",")
            ]
            continue

        # ---------------------------------------------------------------
        # Dialogue event
        # ---------------------------------------------------------------
        if not line.lower().startswith(
            "dialogue:"
        ):
            continue

        data = line.split(
            ":",
            1,
        )[1].lstrip()

        # The final Text field can contain commas.
        if format_fields:
            field_count = len(format_fields)
        else:
            field_count = 10

        fields = data.split(
            ",",
            field_count - 1,
        )

        if len(fields) < 3:
            continue

        # ---------------------------------------------------------------
        # Determine field positions.
        # ---------------------------------------------------------------
        if format_fields:
            try:
                start_index = format_fields.index(
                    "start"
                )

                end_index = format_fields.index(
                    "end"
                )

                text_index = format_fields.index(
                    "text"
                )

            except ValueError:
                start_index = 1
                end_index = 2
                text_index = len(fields) - 1

        else:
            # Standard ASS layout:
            #
            # Layer, Start, End, Style, Name,
            # MarginL, MarginR, MarginV, Effect, Text
            start_index = 1
            end_index = 2
            text_index = 9

        if max(
            start_index,
            end_index,
            text_index,
        ) >= len(fields):
            continue

        start = _parse_subtitle_timestamp(
            fields[start_index]
        )

        end = _parse_subtitle_timestamp(
            fields[end_index]
        )

        if (
            start is None
            or end is None
            or end <= start
        ):
            continue

        subtitle_text = _clean_subtitle_text(
            fields[text_index]
        )

        if not subtitle_text:
            continue

        segments.append(
            {
                "start": start,
                "end": end,
                "text": subtitle_text,
                "words": [],
            }
        )

    return segments


# ---------------------------------------------------------------------------
# WebVTT parser
# ---------------------------------------------------------------------------

def _parse_vtt(text):
    """
    Parse basic WebVTT subtitles.
    """

    segments = []

    text = text.replace(
        "\r\n",
        "\n",
    ).replace(
        "\r",
        "\n",
    )

    timestamp_re = re.compile(
        r"^\s*"
        r"(\d{1,2}:\d{2}(?::\d{2})?\.\d+)"
        r"\s*-->\s*"
        r"(\d{1,2}:\d{2}(?::\d{2})?\.\d+)"
    )

    entries = re.split(
        r"\n\s*\n",
        text,
    )

    for entry in entries:
        lines = [
            line.strip()
            for line in entry.split("\n")
        ]

        match = None
        timestamp_index = None

        for index, line in enumerate(lines):
            current_match = timestamp_re.match(line)

            if current_match:
                match = current_match
                timestamp_index = index
                break

        if not match:
            continue

        start = _parse_subtitle_timestamp(
            match.group(1)
        )

        end = _parse_subtitle_timestamp(
            match.group(2)
        )

        if (
            start is None
            or end is None
            or end <= start
        ):
            continue

        subtitle_text = " ".join(
            line
            for line in lines[
                timestamp_index + 1:
            ]
            if line
        )

        subtitle_text = _clean_subtitle_text(
            subtitle_text
        )

        if not subtitle_text:
            continue

        segments.append(
            {
                "start": start,
                "end": end,
                "text": subtitle_text,
                "words": [],
            }
        )

    return segments


# ---------------------------------------------------------------------------
# External subtitle loader
# ---------------------------------------------------------------------------

def load_external_subtitles(subtitle_path):
    """
    Load an external subtitle file.

    Supported formats:

        .srt
        .ass
        .ssa
        .vtt

    Returns:
        List of normalized subtitle segments.
    """

    subtitle_path = Path(subtitle_path)

    if not subtitle_path.exists():
        raise FileNotFoundError(
            f"Missing subtitles: {subtitle_path}"
        )

    if not subtitle_path.is_file():
        raise FileNotFoundError(
            f"Subtitle path is not a file: "
            f"{subtitle_path}"
        )

    text = _read_subtitle_file(
        subtitle_path
    )

    extension = subtitle_path.suffix.lower()

    if extension == ".srt":
        segments = _parse_srt(text)

    elif extension in (".ass", ".ssa"):
        segments = _parse_ass(text)

    elif extension == ".vtt":
        segments = _parse_vtt(text)

    else:
        raise ValueError(
            f"Unsupported subtitle format: "
            f"{extension}"
        )

    # Always keep subtitle cues chronologically ordered.
    segments.sort(
        key=lambda segment: (
            segment["start"],
            segment["end"],
        )
    )

    if not segments:
        raise ValueError(
            "Subtitle file contains no usable "
            f"subtitles: {subtitle_path.name}"
        )

    print(
        "  External subtitles loaded: "
        f"{subtitle_path.name} "
        f"({len(segments)} cues)"
    )

    return segments


# ---------------------------------------------------------------------------
# Time formatting
# ---------------------------------------------------------------------------

def _format_srt_time(seconds):
    """
    Convert seconds to SRT timestamp format.

    Example:

        83.527
        ->
        00:01:23,527
    """

    seconds = max(
        0.0,
        float(seconds),
    )

    hours = int(
        seconds // 3600
    )

    minutes = int(
        (seconds % 3600) // 60
    )

    whole_seconds = int(
        seconds % 60
    )

    milliseconds = int(
        round(
            (seconds - int(seconds))
            * 1000
        )
    )

    # Handle rounding such as 12.9999 → 13.000.
    if milliseconds >= 1000:
        milliseconds = 0
        whole_seconds += 1

        if whole_seconds >= 60:
            whole_seconds = 0
            minutes += 1

            if minutes >= 60:
                minutes = 0
                hours += 1

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{whole_seconds:02d},"
        f"{milliseconds:03d}"
    )