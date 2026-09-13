"""
MovieShort AI — Configuration
"""
from pathlib import Path

# App version
APP_VERSION = "2.1.1"

# Paths
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = OUTPUT_DIR / "temp"
CACHE_DIR = OUTPUT_DIR / "cache"  # reusable caches (transcripts, person/RMS) — survives auto_cleanup

# Whisper settings
WHISPER_MODEL = "medium"        # tiny/base/small/medium/large-v3
WHISPER_LANGUAGE = "ru"          # Default language for transcription (ru/en)
WHISPER_DEVICE = "auto"          # "auto", "cpu", or "cuda"
FORCE_CPU = False                # True = force CPU even if GPU available
WHISPER_BEAM_SIZE = 5            # Beam size for transcription accuracy

# Video processing defaults — fixed rule, not user-configurable: every
# generated clip must be 1-3 minutes long.
DEFAULT_MAX_CLIP_DURATION = 180  # seconds
DEFAULT_MIN_CLIP_DURATION = 60   # seconds
VERTICAL_WIDTH = 1080
VERTICAL_HEIGHT = 1920

# Banner padding (top/bottom space for banners in shorts)
BANNER_TOP = 300                 # pixels
BANNER_BOTTOM = 300              # pixels

# Face tracking
FACE_TRACKING_INTERVAL = 5       # Analyze every Nth frame
PERSON_SCAN_TIMEOUT_SECONDS = 120  # Give up and center-crop if the per-clip
                                    # face/person scan takes longer than this

# Scene detection
SCENE_THRESHOLD = 27.0
SCENE_FRAME_SKIP = 2             # Process every N+1th frame (0=all, 2=every 3rd)
MIN_SCENE_DURATION = 15.0        # seconds — merge raw scenes shorter than this
MAX_MERGE_DURATION = 120         # seconds — don't merge beyond this (prevents giant scenes)
DIALOGUE_PAUSE_THRESHOLD = 2.0   # seconds — gap > this = scene boundary

# Local Ollama settings — self-hosted local model only, no API key required.
#
# Measured on an RTX 3080 10GB — model + OLLAMA_NUM_CTX combinations that
# stayed 100% GPU-resident (`ollama ps`) vs. spilling to CPU:
#   qwen3:8b          ->  OLLAMA_NUM_CTX = 16000  (fits; 32000 spills to CPU)
#   mistral-nemo:12b  ->  OLLAMA_NUM_CTX = 8192   (fits; 10000 spills to CPU)
# Re-measure with `ollama ps` after loading if you change either — a context
# window Ollama can't fit alongside the model's weights forces a CPU/GPU
# split, which is dramatically slower, not just "a bit slower".
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "qwen3:8b"
OLLAMA_NUM_CTX = 16000

# Visual analysis (local vision model) — runs alongside the text model above
# so a scene with real action or emotion but little/no dialogue (a fight, a
# chase, a silent reveal) doesn't get treated as "nothing happening" just
# because the subtitle track is quiet there. One sample frame per scene
# block gets captioned by this model; the caption is added to the same
# prompt the text model sees, and also rescues otherwise-silent blocks from
# being filtered out before the text model even gets to see them.
#
# qwen3-vl:8b (~6.1GB on disk, `ollama pull qwen3-vl:8b`) — a real
# vision-language model rather than a lightweight captioner (moondream was
# tried first; qwen3-vl's scene/action understanding is meaningfully
# better, which is the entire point of this feature). It only needs to fit
# in VRAM on its own: this runs as its own pass BEFORE the text model's
# batch calls (see core/batch.py), so Ollama loads/unloads each model in
# turn rather than needing both resident at once on the 10GB card.
#
# Like qwen3:8b, this model "thinks" before answering — measured on an
# RTX 3080, and confirmed /no_think does NOT suppress it for image inputs
# the way it does for the text model — so num_predict must stay generous
# (see VISION_NUM_PREDICT below) or the reasoning trace eats the whole
# budget and no actual caption comes back.
#
# If the model isn't pulled or Ollama can't reach it, this auto-disables
# itself and the pipeline falls back to dialogue-only scoring, exactly
# like before this feature existed.
OLLAMA_VISION_MODEL = "qwen3-vl:8b"
VISUAL_ANALYSIS_ENABLED = True
VISUAL_ANALYSIS_TIMEOUT_SECONDS = 60   # per block: ffmpeg frame grab + vision inference (incl. thinking)
VISION_NUM_PREDICT = 500                # generous — most of it is the model's hidden reasoning trace

# Anti-copyright measures (slight transformations to avoid Content ID)
ANTI_COPYRIGHT = True           # master toggle
AC_MIRROR = True                # horizontal flip
AC_CONTRAST = 1.05              # 1.0 = no change
AC_BRIGHTNESS = 0.02            # 0.0 = no change
AC_SATURATION = 1.05            # 1.0 = no change

# Processing options (defaults for GUI)
DEFAULT_BANNER_TOP = 300
DEFAULT_BANNER_BOTTOM = 300
DEFAULT_BLUR_BACKGROUND = True
DEFAULT_ANTI_COPYRIGHT = True
DEFAULT_SUBTITLES = True
DEFAULT_FACE_TRACKING = True
DEFAULT_NUM_CLIPS = 10

# Subtitle editor defaults
SUBTITLE_FONT = "Arial"
SUBTITLE_SIZE = 13
SUBTITLE_COLOR = "&H00FFFFFF"
SUBTITLE_OUTLINE = 1
SUBTITLE_BOLD = True
SUBTITLE_ITALIC = False
SUBTITLE_SHADOW = False
SUBTITLE_POSITION_Y = 400       # px from bottom

# LLM batching — how many scene blocks go into one LLM call, and how the
# prompt/output token budget is split. See OLLAMA_NUM_CTX above: the actual
# context window is the single source of truth for both budgets (see
# core/batch.py's _max_prompt_chars / _max_tokens_for).
DEFAULT_LLM_BATCH_SIZE = 2
PROMPT_CHARS_PER_TOKEN = 2.5     # estimate for English dialogue
PROMPT_INPUT_BUDGET = 0.5        # share of OLLAMA_NUM_CTX reserved for input (rest: output)

# YouTube Shorts output filename hashtags (empty since R7b-9 — no hashtags in clip names)
HASHTAGS = ""

# Gradio
GRADIO_PORT = 7860
GRADIO_SHARE = False
