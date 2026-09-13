"""
MovieShort AI — Faster-Whisper subtitle generation module.
"""
import subprocess
import os
import time as time_module
import warnings
from pathlib import Path

import html
import re

import config
from utils import fmt_duration as _fmt_duration

# Module-level model cache (load once, reuse across calls)
_whisper_model = None
_whisper_device = None  # tracks actual device the model runs on


def _get_model():
    """Get or create the Whisper model (cached)."""
    global _whisper_model, _whisper_device
    if _whisper_model is not None:
        return _whisper_model

    import torch
    if config.WHISPER_DEVICE == "auto":
        cuda_avail = torch.cuda.is_available()
        if cuda_avail and not getattr(config, 'FORCE_CPU', False):
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  GPU обнаружена: {gpu_name}")
        else:
            device = "cpu"
            reason = "отключена в config.py (FORCE_CPU=True)" if getattr(config, 'FORCE_CPU', False) else "не найден"
            print(f"  CUDA {reason} — используется CPU")
            if not getattr(config, 'FORCE_CPU', False):
                print(f"  Для ускорения на GPU установи PyTorch с CUDA:")
                print(f"    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
    else:
        device = config.WHISPER_DEVICE

    print(f"  Загрузка модели '{config.WHISPER_MODEL}' на {device}...")

    from faster_whisper import WhisperModel
    if device == "cuda":
        # Try multiple compute types before falling back to CPU
        cuda_types = ["float16", "int8_float16"]
        last_error = None
        for ctype in cuda_types:
            try:
                _whisper_model = WhisperModel(
                    config.WHISPER_MODEL, device="cuda", compute_type=ctype
                )
                _whisper_device = "cuda"
                print(f"  GPU OK (compute_type={ctype})")
                break
            except Exception as e:
                last_error = e
                print(f"  CUDA compute_type={ctype} не сработал: {e}")

        if _whisper_model is None:
            print(f"  ! CUDA не работает. Переключаюсь на CPU...")
            print(f"  ! (если хочешь разобраться — установи CUDA Toolkit с nvidia.com)")
            print()
            try:
                _whisper_model = WhisperModel(
                    config.WHISPER_MODEL, device="cpu", compute_type="int8"
                )
                _whisper_device = "cpu"
                print(f"  CPU OK (медленнее, но стабильно)")
            except Exception as e2:
                print(f"  ! CPU тоже не загрузился: {e2}")
                print(f"  ! Удали папку {Path.home() / '.cache' / 'huggingface'} и попробуй снова")
                raise
    else:
        _whisper_model = WhisperModel(
            config.WHISPER_MODEL, device="cpu", compute_type="int8"
        )
        _whisper_device = "cpu"

    print(f"  Модель '{config.WHISPER_MODEL}' готова к работе.")
    return _whisper_model


def transcribe(video_path, language=None, progress_callback=None):
    """
    Transcribe video audio using faster-whisper (cached model).
    Returns list of segments: [{start, end, text, words}].
    Each word: {start, end, word, probability}.
    """
    if language is None:
        language = config.WHISPER_LANGUAGE

    video_path = str(video_path)
    temp_audio = str(Path(config.TEMP_DIR) / f"_temp_audio_{os.getpid()}.wav")

    os.makedirs(config.TEMP_DIR, exist_ok=True)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn",
             "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
             temp_audio],
            capture_output=True, check=True, timeout=300)
    except subprocess.CalledProcessError:
        warnings.warn("Could not extract audio from video (no audio track?)")
        return []
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Install: winget install FFmpeg")

    return _transcribe_audio_file(temp_audio, language, progress_callback)


def _transcribe_audio_file(audio_path, language, progress_callback=None):
    """Transcribe an already-extracted audio file using cached model."""
    model = _get_model()

    # Get audio duration from file size (16kHz, mono, 16-bit WAV = 32000 bytes/sec)
    audio_size = os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
    total_duration_est = audio_size / 32000.0

    if total_duration_est > 0:
        # GPU ~20x faster than CPU
        is_gpu = _whisper_device == "cuda"
        est_factor = {"tiny": 3, "base": 5, "small": 8, "medium": 20, "large": 50}
        factor = est_factor.get(config.WHISPER_MODEL, 15)
        if is_gpu:
            factor = max(1, factor / 20)  # GPU ~20x faster
        est_sec = total_duration_est * factor
        device_label = "GPU" if is_gpu else "CPU"
        print(f"  Audio duration: {_fmt_duration(total_duration_est)}")
        print(f"  Est. transcription time: ~{_fmt_duration(est_sec)} ({device_label}, {config.WHISPER_MODEL} model)")
        print()

    segments = []
    last_report_pct = 0
    transcribe_start = time_module.time()

    try:
        seg_gen, info = model.transcribe(
            audio_path, language=language,
            word_timestamps=True, beam_size=config.WHISPER_BEAM_SIZE)

        total_duration = info.duration if info else total_duration_est

        for seg in seg_gen:
            words_list = []
            if seg.words:
                for w in seg.words:
                    words_list.append({
                        "start": w.start,
                        "end": w.end,
                        "word": w.word,
                        "probability": w.probability,
                    })
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
                "words": words_list,
            })

            # Report progress every ~5% of audio
            if total_duration and total_duration > 0:
                pct = min(100, int(seg.end / total_duration * 100))
                if pct >= last_report_pct + 5:
                    last_report_pct = (pct // 5) * 5
                    wall_elapsed = time_module.time() - transcribe_start
                    processed = seg.end
                    if processed > 0 and wall_elapsed > 0:
                        speed = processed / wall_elapsed
                        remaining = total_duration - processed
                        eta = remaining / speed
                        eta_str = _fmt_duration(eta)
                    else:
                        eta_str = "?"
                    msg = f"  Transcribing: {last_report_pct}%  elapsed {_fmt_duration(wall_elapsed)}  ETA {eta_str}"
                    print(msg)
                    if progress_callback:
                        progress_callback(pct / 100.0, msg)

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

    print(f"  Done: {len(segments)} segments transcribed")
    return segments





def generate_srt(segments, output_path):
    """Convert segments to .srt subtitle file."""
    lines = []
    for i, seg in enumerate(segments, 1):
        start = _format_srt_time(seg["start"])
        end = _format_srt_time(seg["end"])
        text = seg.get("text", "").strip()
        if not text:
            continue
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")

    output_path = str(output_path)
    with open(output_path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines))

    print(f"SRT saved: {output_path}")


def _iter_word_groups(segments, max_chars=150):
    """Yield (start, end, text) word-groups — shared by SRT and ASS writers.

    Grouping logic is identical to the original generate_word_group_srt:
    words grouped up to max_chars; segments without word timestamps fall
    back to splitting segment text into max_chars chunks.
    """
    for seg in segments:
        words = seg.get("words", [])
        if not words:
            # Fallback: use segment text, split by max_chars
            text = seg.get("text", "").strip()
            if not text:
                continue
            while text:
                chunk = text[:max_chars]
                text = text[max_chars:]
                yield seg["start"], seg["end"], chunk
            continue

        # Group words by max_chars
        groups = []
        current_group = []
        current_len = 0

        for w in words:
            word_text = w.get("word", "").strip()
            if not word_text:
                continue
            # +1 for space before word (except first)
            added_len = len(word_text) + (1 if current_group else 0)
            if current_group and current_len + added_len > max_chars:
                groups.append(current_group)
                current_group = [w]
                current_len = len(word_text)
            else:
                current_group.append(w)
                current_len += added_len

        if current_group:
            groups.append(current_group)

        for group in groups:
            group_start = group[0].get("start", seg["start"])
            group_end = group[-1].get("end", seg["end"])
            group_text = " ".join(
                w.get("word", "").strip() for w in group if w.get("word", "").strip()
            )
            if not group_text:
                continue
            yield group_start, group_end, group_text


def generate_word_group_srt(segments, output_path, max_chars=150):
    """Generate SRT with word-groups (2-4 words per line, max 1 line).

    Uses word-level timestamps from Whisper for precise sync.
    Falls back to segment-level if no word timestamps available.
    """
    lines = []
    idx = 0

    for group_start, group_end, group_text in _iter_word_groups(segments, max_chars):
        idx += 1
        lines.append(f"{idx}")
        lines.append(f"{_format_srt_time(group_start)} --> {_format_srt_time(group_end)}")
        lines.append(group_text)
        lines.append("")

    output_path = str(output_path)
    with open(output_path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines))

    print(f"Word-group SRT saved: {output_path} ({idx} entries)")
    return idx


def save_segments_json(segments, output_path):
    """Save transcript segments to JSON for reuse across clips."""
    import json as json_mod
    output_path = str(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json_mod.dump(segments, f, ensure_ascii=False, indent=2)
    print(f"Segments JSON saved: {output_path} ({len(segments)} segments)")


def load_segments_json(input_path):
    """Load transcript segments from JSON."""
    import json as json_mod
    input_path = str(input_path)
    with open(input_path, "r", encoding="utf-8") as f:
        return json_mod.load(f)


def filter_segments_in_range(all_segments, start_sec, end_sec):
    """Filter segments that overlap with [start_sec, end_sec].

    Adjusts timestamps relative to start_sec for clip-level SRT.
    """
    filtered = []
    for seg in all_segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        # Check overlap
        if seg_start < end_sec and seg_end > start_sec:
            # Clip timestamps to range and offset
            clipped_start = max(0, seg_start - start_sec)
            clipped_end = min(end_sec - start_sec, seg_end - start_sec)
            new_seg = {
                "start": clipped_start,
                "end": clipped_end,
                "text": seg.get("text", ""),
                "words": [],
            }
            # Adjust word timestamps
            for w in seg.get("words", []):
                w_start = w.get("start", seg_start)
                w_end = w.get("end", seg_end)
                if w_start < end_sec and w_end > start_sec:
                    new_seg["words"].append({
                        "start": max(0, w_start - start_sec),
                        "end": min(end_sec - start_sec, w_end - start_sec),
                        "word": w.get("word", ""),
                        "probability": w.get("probability", 0),
                    })
            filtered.append(new_seg)
    return filtered


def _format_srt_time(seconds):
    """Convert seconds to SRT time format: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def find_external_subtitle(video_path, explicit_path=None):
    """
    Find an external subtitle file for a movie.

    Priority:
      1. Explicit subtitle path supplied by the user.
      2. Exact movie filename with .srt/.ass/.ssa/.vtt.
      3. Common language suffixes such as .en, .eng, .english.

    Returns:
        str path to subtitle file, or None if no subtitle was found.
    """
    video_path = Path(video_path)

    # 1. Explicit subtitle path
    if explicit_path:
        explicit = Path(str(explicit_path))
        if explicit.exists() and explicit.is_file():
            return str(explicit)

    # 2. Exact matching filename
    extensions = [".srt", ".ass", ".ssa", ".vtt"]

    for ext in extensions:
        candidate = video_path.with_suffix(ext)
        if candidate.exists() and candidate.is_file():
            return str(candidate)

    # 3. Common language suffixes
    language_suffixes = [
        ".en",
        ".eng",
        ".english",
        ".en-US",
        ".en-GB",
    ]

    for suffix in language_suffixes:
        for ext in extensions:
            candidate = video_path.with_name(video_path.stem + suffix + ext)
            if candidate.exists() and candidate.is_file():
                return str(candidate)

    return None


def _read_subtitle_file(path):
    """Read subtitle text with several common encodings."""
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
            with open(path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError as e:
            last_error = e
        except OSError:
            raise

    raise UnicodeDecodeError(
        "subtitle",
        b"",
        0,
        1,
        f"Could not decode subtitle file: {path} ({last_error})",
    )


def _parse_subtitle_timestamp(value):
    """
    Parse subtitle timestamp into seconds.

    Supports:
      00:01:23,456
      00:01:23.456
      0:01:23.45
      01:23.456
    """
    value = value.strip().replace(",", ".")

    parts = value.split(":")

    try:
        if len(parts) == 3:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds

        if len(parts) == 2:
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds

        return float(value)

    except ValueError:
        return None


def _clean_subtitle_text(text):
    """
    Clean subtitle formatting while preserving meaningful sound/action cues.

    Examples preserved:
        [door slams]
        (grunts)
        ♪ dramatic music ♪
        [phone ringing]

    Formatting tags such as <i>...</i> and ASS override tags are removed.
    """
    text = html.unescape(text)

    # ASS formatting overrides: {\i1}, {\an8}, {\pos(...)} etc.
    text = re.sub(r"\{[^}]*\}", "", text)

    # HTML subtitle formatting.
    text = re.sub(r"<[^>]+>", "", text)

    # ASS newline markers.
    text = text.replace(r"\N", "\n")
    text = text.replace(r"\n", "\n")

    # Remove excessive whitespace while preserving line breaks.
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            lines.append(line)

    return " ".join(lines).strip()


def _parse_srt(text):
    """Parse SRT subtitles into the internal MovieShort segment format."""
    segments = []

    # Normalize line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Split on blank lines between subtitle entries.
    entries = re.split(r"\n\s*\n", text)

    timestamp_re = re.compile(
        r"^\s*"
        r"(\d{1,2}:\d{2}:\d{2}[,.]\d+)"
        r"\s*-->\s*"
        r"(\d{1,2}:\d{2}:\d{2}[,.]\d+)"
    )

    for entry in entries:
        lines = [line.strip() for line in entry.split("\n") if line.strip()]

        if not lines:
            continue

        timestamp_index = None

        for i, line in enumerate(lines):
            if timestamp_re.match(line):
                timestamp_index = i
                break

        if timestamp_index is None:
            continue

        match = timestamp_re.match(lines[timestamp_index])

        start = _parse_subtitle_timestamp(match.group(1))
        end = _parse_subtitle_timestamp(match.group(2))

        if start is None or end is None or end <= start:
            continue

        subtitle_text = " ".join(lines[timestamp_index + 1:])
        subtitle_text = _clean_subtitle_text(subtitle_text)

        if not subtitle_text:
            continue

        segments.append({
            "start": start,
            "end": end,
            "text": subtitle_text,
            # External subtitles normally do not contain word timestamps.
            "words": [],
        })

    return segments


def _parse_ass(text):
    """
    Parse ASS/SSA dialogue events.

    ASS timestamps use:
        H:MM:SS.cc

    We locate the [Events] section and read Dialogue lines.
    """
    segments = []

    in_events = False
    format_fields = None

    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()

        if not line:
            continue

        if line.startswith("["):
            in_events = line.lower() == "[events]"
            format_fields = None
            continue

        if not in_events:
            continue

        if line.lower().startswith("format:"):
            format_fields = [
                field.strip().lower()
                for field in line.split(":", 1)[1].split(",")
            ]
            continue

        if not line.lower().startswith("dialogue:"):
            continue

        data = line.split(":", 1)[1].lstrip()

        # ASS dialogue normally has 10 comma-separated fields, but the
        # Text field can itself contain commas. Split only the first
        # N-1 separators.
        if format_fields:
            field_count = len(format_fields)
        else:
            field_count = 10

        fields = data.split(",", field_count - 1)

        if len(fields) < 3:
            continue

        if format_fields:
            try:
                start_index = format_fields.index("start")
                end_index = format_fields.index("end")
                text_index = format_fields.index("text")
            except ValueError:
                start_index = 1
                end_index = 2
                text_index = len(fields) - 1
        else:
            # Standard ASS format:
            # Layer, Start, End, Style, Name, MarginL, MarginR,
            # MarginV, Effect, Text
            start_index = 1
            end_index = 2
            text_index = 9

        if max(start_index, end_index, text_index) >= len(fields):
            continue

        start = _parse_subtitle_timestamp(fields[start_index])
        end = _parse_subtitle_timestamp(fields[end_index])

        if start is None or end is None or end <= start:
            continue

        subtitle_text = _clean_subtitle_text(fields[text_index])

        if not subtitle_text:
            continue

        segments.append({
            "start": start,
            "end": end,
            "text": subtitle_text,
            "words": [],
        })

    return segments


def _parse_vtt(text):
    """Parse basic WebVTT subtitles."""
    segments = []

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    timestamp_re = re.compile(
        r"^\s*"
        r"(\d{1,2}:\d{2}(?::\d{2})?\.\d+)"
        r"\s*-->\s*"
        r"(\d{1,2}:\d{2}(?::\d{2})?\.\d+)"
    )

    entries = re.split(r"\n\s*\n", text)

    for entry in entries:
        lines = [line.strip() for line in entry.split("\n")]

        match = None
        timestamp_index = None

        for i, line in enumerate(lines):
            m = timestamp_re.match(line)
            if m:
                match = m
                timestamp_index = i
                break

        if not match:
            continue

        start = _parse_subtitle_timestamp(match.group(1))
        end = _parse_subtitle_timestamp(match.group(2))

        if start is None or end is None or end <= start:
            continue

        subtitle_text = " ".join(
            line for line in lines[timestamp_index + 1:] if line
        )
        subtitle_text = _clean_subtitle_text(subtitle_text)

        if not subtitle_text:
            continue

        segments.append({
            "start": start,
            "end": end,
            "text": subtitle_text,
            "words": [],
        })

    return segments


def load_external_subtitles(subtitle_path):
    """
    Load an external subtitle file.

    Supported:
      .srt
      .ass
      .ssa
      .vtt

    Returns the same segment structure used by Whisper.
    """
    subtitle_path = Path(subtitle_path)

    if not subtitle_path.exists():
        raise FileNotFoundError(
            f"Missing subtitles: {subtitle_path}"
        )

    text = _read_subtitle_file(subtitle_path)
    extension = subtitle_path.suffix.lower()

    if extension == ".srt":
        segments = _parse_srt(text)
    elif extension in (".ass", ".ssa"):
        segments = _parse_ass(text)
    elif extension == ".vtt":
        segments = _parse_vtt(text)
    else:
        raise ValueError(
            f"Unsupported subtitle format: {extension}"
        )

    # Sort by timestamp.
    segments.sort(key=lambda s: (s["start"], s["end"]))

    if not segments:
        raise ValueError(
            f"Subtitle file contains no usable subtitles: {subtitle_path.name}"
        )

    print(
        f"  External subtitles loaded: "
        f"{subtitle_path.name} ({len(segments)} cues)"
    )

    return segments