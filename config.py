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

# Video processing defaults
DEFAULT_MAX_CLIP_DURATION = 60   # seconds
DEFAULT_MIN_CLIP_DURATION = 15   # seconds
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

# LLM — self-hosted local model only. No internet API or API key is required.
LLM_BASE_URL = "http://127.0.0.1:11434"
LLM_MODEL = "qwen3:8b"

# Local Ollama settings
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "qwen3:8b"
# Context window sent to Ollama on every request. Without this, Ollama uses
# its own default (commonly 4096) regardless of what the batching logic
# above assumes is available. 16000 was measured to keep qwen3:8b fully
# resident on a 10GB GPU (RTX 3080); going to the full 32000 pushed total
# memory to ~10GB and forced a slow CPU/GPU split. Lower this if your GPU
# has less VRAM, or if you switch to a larger model.
OLLAMA_NUM_CTX = 16000

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

# LLM batching for the local model's context window — fewer/larger calls
# for models with more room, smaller calls for the default local model.
MODEL_BATCH_SIZES = {"deepseek-v4-flash": 4, "nemotron-3-ultra-free": 4, "big-pickle": 3, "mimo-v2.5-free": 3, "hy3-free": 3, "nemotron-3.5-lightning-free": 3}
DEFAULT_LLM_BATCH_SIZE = 2
MODEL_CONTEXT_TOKENS = {"deepseek-v4-flash": 1000000, "nemotron-3-ultra-free": 1000000, "big-pickle": 200000, "mimo-v2.5-free": 200000, "hy3-free": 190000, "nemotron-3.5-lightning-free": 262000}
DEFAULT_CONTEXT_TOKENS = 32000
PROMPT_CHARS_PER_TOKEN = 2.5     # estimate for English dialogue
PROMPT_INPUT_BUDGET = 0.5        # half the context reserved for input (rest: 4096 output + overhead)

# YouTube Shorts output filename hashtags (empty since R7b-9 — no hashtags in clip names)
HASHTAGS = ""

# Gradio
GRADIO_PORT = 7860
GRADIO_SHARE = False
