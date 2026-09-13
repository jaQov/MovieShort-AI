"""
MovieShort AI — Text Analyzer
Scene/clip scoring via the self-hosted local Ollama model (Qwen).
"""

from __future__ import annotations

import json
import re
import time

import httpx

import config


def call_llm(prompt_text: str, max_tokens: int = 256) -> str:
    """Send a prompt to the local Ollama model and return its reply text."""
    base = getattr(config, "OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = getattr(config, "OLLAMA_MODEL", "qwen3:8b")

    print(f"  Local Ollama model: {model}")

    url = f"{base}/api/chat"

    # Qwen3 supports /no_think. Scene scoring only needs the final structured
    # answer, so disabling thinking makes the response faster and easier to parse.
    ollama_prompt = "/no_think\n\n" + prompt_text

    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": ollama_prompt,
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": max_tokens,
        },
    }

    last_exc: Exception | None = None
    attempts = 0

    for attempt in range(3):
        try:
            resp = httpx.post(
                url,
                json=body,
                headers={"Content-Type": "application/json"},
                timeout=120,
            )

            if resp.status_code in (408, 429, 500, 502, 503, 504):
                if attempt < 2:
                    wait = 2 * (attempt + 1)
                    print(f"  Ollama temporary error ({resp.status_code}), retrying in {wait}s...")
                    time.sleep(wait)
                    continue

            resp.raise_for_status()

            data = resp.json()

            # Ollama /api/chat returns:
            # {
            #   "message": {
            #       "role": "assistant",
            #       "content": "..."
            #   }
            # }
            content = data["message"]["content"]

            if content is None:
                raise RuntimeError("Ollama returned null content")

            return content.strip()

        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            attempts = attempt + 1
            last_exc = exc

            if attempt < 2:
                time.sleep(2)

    msg = (
        f"Local Ollama API failed after {attempts} attempt"
        f"{'s' if attempts != 1 else ''}: {last_exc}"
    )

    raise RuntimeError(msg)


def check_ollama() -> dict:
    """Check that the local Ollama server is running and the model responds.

    Returns:
        {"ok": True} if the model answers,
        {"ok": False, "error": "message"} otherwise.
    """
    base = getattr(config, "OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = getattr(config, "OLLAMA_MODEL", "qwen3:8b")

    try:
        # First check that Ollama is alive.
        tags_resp = httpx.get(f"{base}/api/tags", timeout=15)
        tags_resp.raise_for_status()

        data = tags_resp.json()
        models = data.get("models", [])

        model_names = {str(model_info.get("name", "")) for model_info in models}

        if model not in model_names:
            # Ollama may return the model without an explicit tag in some
            # situations, so also accept the base model name.
            base_model = model.split(":")[0]
            if not any(name.split(":")[0] == base_model for name in model_names):
                return {
                    "ok": False,
                    "error": f"Ollama: model {model} is not installed",
                }

        # Actually test inference so the GUI check confirms the model can
        # respond, not merely that Ollama is running.
        resp = httpx.post(
            f"{base}/api/chat",
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": "/no_think\nReply with exactly: OK",
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 4,
                },
            },
            headers={"Content-Type": "application/json"},
            timeout=30,
        )

        resp.raise_for_status()

        result = resp.json()
        content = result.get("message", {}).get("content", "")

        if content:
            return {"ok": True}

        return {"ok": False, "error": "Ollama: empty response"}

    except httpx.TimeoutException:
        return {"ok": False, "error": "Ollama: timed out"}

    except httpx.ConnectError:
        return {"ok": False, "error": "Ollama: server is not running"}

    except httpx.HTTPStatusError as e:
        return {"ok": False, "error": f"Ollama: HTTP {e.response.status_code}"}

    except Exception as e:
        return {"ok": False, "error": f"Ollama: {str(e)[:100]}"}


# ---------------------------------------------------------------------------
# Batch block-to-clips prompt — multiple blocks in one LLM call
# ---------------------------------------------------------------------------

PROMPT_BATCH_TO_CLIPS = (
    "You are an expert at cutting movie blocks into YouTube Shorts.\n"
    "Movie: «{movie_name}»\n\n"
    "Below are scene blocks. For EACH block, determine which clips "
    "are suitable for Shorts.\n"
    "---\n"
    "{blocks_text}\n"
    "---\n\n"
    "Rules (per block):\n"
    "- Each clip: 30-75 seconds. Prefer ~60 seconds.\n"
    "- Clips < 20 seconds are allowed ONLY with reason=\"self_contained\"\n"
    "  (complete joke, quotable line, self-sufficient moment).\n"
    "- start/end are ABSOLUTE seconds from MOVIE START (not block-relative).\n"
    "- Clips within ONE block must NOT overlap.\n"
    "- Decide how many clips per block. 1-3 is usually enough.\n"
    "- If a block is uninteresting — don't include its clips.\n\n"
    "Output format (strict JSON array, no explanations):\n"
    '[{{"start": 0.0, "end": 60.0, "title": "Title", "score": 8.5,\n'
    '  "reason": "description", "block": 0}}]\n\n'
    'Field "block" is 0-based block index (0, 1, 2, 3...).\n'
    "Example:\n"
    '[{{"start": 5.0, "end": 60.0, "title": "Tony suits up", "score": 9.0,\n'
    '  "reason": "iconic scene", "block": 0}},\n'
    ' {{"start": 130.0, "end": 185.0, "title": "Tower talk", "score": 7.0,\n'
    '  "reason": "dialogue", "block": 1}}]'
)


def _parse_batch_response(raw: str, block_start_times: list[float]) -> dict[int, list[dict]]:
    """Parse batch LLM response into per-block clip lists.

    Expects a JSON array with each item having "block" field (int).

    Args:
        raw: raw LLM response text
        block_start_times: list of block start times (seconds from film start)

    Returns:
        dict mapping block_index -> list of clip dicts (with absolute timestamps)
    """
    if not raw or not isinstance(raw, str):
        return {}
    content = raw.strip()
    if not content:
        return {}

    # 1) extract from ```json fence if present, else use raw content
    fence_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', content, re.DOTALL)
    json_text = fence_match.group(1) if fence_match else content

    # 2) try json.loads on extracted text
    try:
        clips = json.loads(json_text)
        if isinstance(clips, list):
            result: dict[int, list[dict]] = {}
            for item in clips:
                if not isinstance(item, dict):
                    continue
                block_raw = item.get("block")
                try:
                    block_idx = int(block_raw) if isinstance(block_raw, str) else block_raw
                except (ValueError, TypeError):
                    continue
                if not isinstance(block_idx, int):
                    continue
                if block_idx < 0 or block_idx >= len(block_start_times):
                    continue
                result.setdefault(block_idx, []).append(item)
            return result
        # if not list, fall through to regex fallback
    except (json.JSONDecodeError, TypeError):
        pass

    # also try extracting array substring from json_text if direct loads failed
    arr_match = re.search(r'\[.*\]', json_text, re.DOTALL)
    if arr_match:
        try:
            clips2 = json.loads(arr_match.group(0))
            if isinstance(clips2, list):
                result: dict[int, list[dict]] = {}
                for item in clips2:
                    if not isinstance(item, dict):
                        continue
                    block_raw = item.get("block")
                    try:
                        block_idx = int(block_raw) if isinstance(block_raw, str) else block_raw
                    except (ValueError, TypeError):
                        continue
                    if not isinstance(block_idx, int):
                        continue
                    if block_idx < 0 or block_idx >= len(block_start_times):
                        continue
                    result.setdefault(block_idx, []).append(item)
                return result
        except (json.JSONDecodeError, TypeError):
            pass

    # 3) regex fallback for truncated / malformed JSON — collect objects with "block"
    result: dict[int, list[dict]] = {}
    for m in re.finditer(r'\{[^{}]*"block"[^{}]*\}', content):
        obj_text = m.group(0)
        try:
            item = json.loads(obj_text)
        except (json.JSONDecodeError, TypeError):
            # try manual field extraction for block str/int
            bm = re.search(r'"block"\s*:\s*"?(\d+)"?', obj_text)
            if not bm:
                continue
            try:
                item = {"block": int(bm.group(1))}
                # also try to extract start/end/title/score if present for completeness
                sm = re.search(r'"start"\s*:\s*([\d.]+)', obj_text)
                em = re.search(r'"end"\s*:\s*([\d.]+)', obj_text)
                if sm:
                    try:
                        item["start"] = float(sm.group(1))
                    except ValueError:
                        pass
                if em:
                    try:
                        item["end"] = float(em.group(1))
                    except ValueError:
                        pass
                tm = re.search(r'"title"\s*:\s*"([^"]*)"', obj_text)
                if tm:
                    item["title"] = tm.group(1)
                sc = re.search(r'"score"\s*:\s*([\d.]+)', obj_text)
                if sc:
                    try:
                        item["score"] = float(sc.group(1))
                    except ValueError:
                        pass
            except (ValueError, TypeError):
                continue
        if not isinstance(item, dict):
            continue
        block_raw = item.get("block")
        try:
            block_idx = int(block_raw) if isinstance(block_raw, str) else block_raw
        except (ValueError, TypeError):
            continue
        if not isinstance(block_idx, int):
            continue
        if block_idx < 0 or block_idx >= len(block_start_times):
            continue
        result.setdefault(block_idx, []).append(item)

    return result
