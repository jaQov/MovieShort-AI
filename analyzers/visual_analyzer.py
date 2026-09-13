"""
MovieShort AI — Visual Analyzer

Captions a single sample frame per scene block using a local Ollama vision
model (qwen3-vl:8b by default, see config.OLLAMA_VISION_MODEL). This runs
ALONGSIDE the subtitle-based text analysis in analyzers/text_analyzer.py —
not instead of it.

Why this exists: judging a scene purely by its dialogue means a silent
fight, a chase, or a wordless emotional beat looks identical to an empty
room to the text model, because both have no subtitle text. A short visual
caption ("two men fighting in an alley" vs. "an empty hallway, nothing
happening") gives the text model — and the pre-filter that runs before it —
something to go on even when there's no dialogue at all.

Never raises: any failure (ffmpeg missing, Ollama unreachable, the vision
model not pulled, a bad frame) degrades to an empty caption so the rest of
the pipeline keeps working exactly as it did with text-only analysis.
"""

from __future__ import annotations

import base64
import re
import subprocess

import httpx

import config


# A 64x64 solid-gray JPEG, used only to verify the vision model accepts an
# image and responds at all — no real video frame is needed for the
# availability check. Deliberately NOT a 1x1/2x2 pixel image: some vision
# models (qwen3-vl included) reject images below their patch-embedding
# minimum size with a hard 400 error rather than just describing it.
_PROBE_IMAGE_B64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJ"
    "BwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoK"
    "CgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCABAAEADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAA"
    "AAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAk"
    "M2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKT"
    "lJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QA"
    "HwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdh"
    "cRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hp"
    "anN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk"
    "5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAo"
    "oooAKKKKACiiigAooooAKKKKACiiigAooooA/9k="
)

_VISION_PROMPT = (
    "Describe this movie frame in ONE short sentence, focused on WHAT IS "
    "HAPPENING (action, danger, emotion, tension) rather than visual style "
    "or composition. Say plainly if it looks like a fight, chase, argument, "
    "romantic moment, dramatic reveal, calm conversation, or an empty/"
    "static shot (e.g. scenery, black screen, credits, a title card)."
)

# Caption phrases that mean "nothing worth cutting is on screen" — used to
# decide whether a visual caption should rescue an otherwise-silent block.
_STATIC_VISUAL_RE = re.compile(
    r"\b(black screen|blank screen|nothing (is |much )?happening|no action|"
    r"static shot|title card|end credits|credits (roll|rolling|scene)|"
    r"studio logo|copyright notice|opening logo)\b",
    re.I,
)

# One-shot cache: whether the vision model looked reachable/usable the last
# time it was checked this process. Re-checked once per process, not once
# per block — calling out to Ollama for every single block just to find out
# it's unavailable would be dozens of wasted round-trips per movie.
_availability = {"checked": False, "ok": False, "reason": ""}


def is_visually_interesting(caption: str) -> bool:
    """True if a caption describes something worth keeping, not a static/
    empty shot. Empty captions (analysis failed or wasn't run) are not
    considered interesting — they carry no information either way."""
    caption = (caption or "").strip()
    return bool(caption) and not _STATIC_VISUAL_RE.search(caption)


def _extract_frame_jpeg(video_path, timestamp_sec: float, timeout: float) -> bytes:
    """Grab one JPEG frame at timestamp_sec via ffmpeg, piped straight to
    memory (no temp file). Raises on any failure — callers must catch."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{max(0.0, timestamp_sec):.3f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-vf", "scale=512:-2",
        "-q:v", "4",
        "-f", "image2",
        "-vcodec", "mjpeg",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if result.returncode != 0 or not result.stdout:
        stderr = (result.stderr or b"").decode(errors="replace").strip()
        raise RuntimeError(f"ffmpeg couldn't grab a frame at {timestamp_sec:.0f}s: {stderr[:200]}")
    return result.stdout


def _call_vision_model(image_b64: str, model: str, timeout: float, prompt: str = _VISION_PROMPT,
                        num_predict: int | None = None) -> tuple[str, str]:
    """Send one image to the local Ollama vision model.

    Returns (content, thinking) — both stripped strings. Some vision models
    (qwen3-vl included) "think" before answering, and empirically /no_think
    does NOT suppress that for image inputs the way it does for text-only
    qwen3 — so num_predict must stay generous (config.VISION_NUM_PREDICT)
    or the reasoning trace alone can consume the whole budget and leave
    content empty even though the model is working fine.
    """
    base = getattr(config, "OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    if num_predict is None:
        num_predict = getattr(config, "VISION_NUM_PREDICT", 500)
    resp = httpx.post(
        f"{base}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "user", "content": prompt, "images": [image_b64]},
            ],
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": num_predict},
        },
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    message = data.get("message") or {}
    content = (message.get("content") or "").strip()
    thinking = (message.get("thinking") or "").strip()
    return content, thinking


def check_vision_model() -> dict:
    """Check that Ollama is running, the vision model is pulled, and it can
    actually accept an image and respond. Mirrors text_analyzer.check_ollama().

    Returns {"ok": True} or {"ok": False, "error": "..."}.
    """
    base = getattr(config, "OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = getattr(config, "OLLAMA_VISION_MODEL", "qwen3-vl:8b")

    try:
        tags_resp = httpx.get(f"{base}/api/tags", timeout=15)
        tags_resp.raise_for_status()
        model_names = {str(m.get("name", "")) for m in tags_resp.json().get("models", [])}

        if model not in model_names:
            base_model = model.split(":")[0]
            if not any(name.split(":")[0] == base_model for name in model_names):
                return {"ok": False, "error": f"Ollama: vision model {model} is not installed "
                                               f"(run: ollama pull {model})"}

        # num_predict generous enough to survive this model's "thinking"
        # trace (see module docstring) — a small budget here would look
        # like a dead/broken model when it's actually just mid-thought.
        content, thinking = _call_vision_model(_PROBE_IMAGE_B64, model, timeout=30,
                                                prompt="Reply with exactly: OK", num_predict=200)
        if content or thinking:
            return {"ok": True}
        return {"ok": False, "error": "Ollama: vision model gave an empty response"}

    except httpx.TimeoutException:
        return {"ok": False, "error": "Ollama: vision model check timed out"}
    except httpx.ConnectError:
        return {"ok": False, "error": "Ollama: server is not running"}
    except httpx.HTTPStatusError as e:
        return {"ok": False, "error": f"Ollama: HTTP {e.response.status_code}"}
    except Exception as e:
        return {"ok": False, "error": f"Ollama: {str(e)[:100]}"}


def visual_analysis_available(force_recheck: bool = False) -> bool:
    """Cached: is visual analysis usable right now? Checked once per process
    unless force_recheck is set (used by the GUI's manual "check" button)."""
    if not getattr(config, "VISUAL_ANALYSIS_ENABLED", True):
        return False
    if _availability["checked"] and not force_recheck:
        return _availability["ok"]
    result = check_vision_model()
    _availability["checked"] = True
    _availability["ok"] = result["ok"]
    _availability["reason"] = result.get("error", "")
    return result["ok"]


def describe_frame(video_path, timestamp_sec: float, model: str | None = None) -> tuple[str, str | None]:
    """Caption one frame of video_path at timestamp_sec.

    Returns (caption, error) — caption is "" and error is set on any
    failure; never raises.
    """
    model = model or getattr(config, "OLLAMA_VISION_MODEL", "qwen3-vl:8b")
    timeout = getattr(config, "VISUAL_ANALYSIS_TIMEOUT_SECONDS", 30)

    try:
        frame_bytes = _extract_frame_jpeg(video_path, timestamp_sec, timeout=timeout)
    except Exception as e:
        return "", f"frame extraction failed: {e}"

    try:
        image_b64 = base64.b64encode(frame_bytes).decode("ascii")
        caption, _thinking = _call_vision_model(image_b64, model, timeout=timeout)
        if not caption:
            return "", "vision model produced no caption (reasoning trace may have used the whole budget)"
        return caption, None
    except Exception as e:
        return "", f"vision model call failed: {e}"


def describe_blocks(video_path, blocks: list, progress_every: int = 10) -> list:
    """Caption one representative (mid-point) frame per block, in place.

    Sets block["visual"] on every block. If the vision model isn't
    available (not enabled, not pulled, Ollama unreachable), this sets ""
    on every block and returns immediately — no frame grabs are attempted,
    so a missing vision model costs nothing but a single fast check.
    """
    if not visual_analysis_available():
        model = getattr(config, "OLLAMA_VISION_MODEL", "qwen3-vl:8b")
        reason = _availability.get("reason") or "disabled in config"
        print(f"  ⚠ Visual analysis skipped ({reason}) — continuing with "
              f"dialogue-only scoring. To enable it: ollama pull {model}")
        for block in blocks:
            block["visual"] = ""
        return blocks

    model = getattr(config, "OLLAMA_VISION_MODEL", "qwen3-vl:8b")
    print(f"  Looking at one sample frame per section with the local vision "
          f"model ({model}), so scenes with real action/emotion but little "
          "dialogue don't get skipped as \"nothing happening\"...")

    described = 0
    failed = 0
    for i, block in enumerate(blocks):
        mid = (block["start"] + block["end"]) / 2
        caption, error = describe_frame(video_path, mid, model=model)
        block["visual"] = caption
        if caption:
            described += 1
        elif error:
            failed += 1
        if (i + 1) % progress_every == 0 or (i + 1) == len(blocks):
            print(f"    Captioned {i + 1}/{len(blocks)} section(s)")

    print(f"  ✓ Got a visual description for {described}/{len(blocks)} "
          f"section(s){f' ({failed} failed)' if failed else ''}")
    return blocks
