"""
MovieShort AI — User config persistence (saves/loads user_config.json).
"""
import json
import os
import shutil

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "user_config.json"
)

DEFAULT_CONFIG = {
    "movie_title": "",
    "num_clips": 10,
    "min_duration": 15,
    "max_duration": 60,
    "subtitles": True,
    "face_tracking": True,
    # Processing options
    "banner_top": 300,
    "banner_bottom": 300,
    "blur_background": True,
    "anti_copyright": True,
    "score_threshold": 7.0,
    "auto_cleanup": True,
    # Subtitle editor
    "subtitle_font": "Arial",
    "subtitle_font_name": "Bebas Neue",  # display name from utils/font_manager.POPULAR_FONTS
    "subtitle_size": 13,
    "subtitle_color": "&H00FFFFFF",
    "subtitle_outline": 1,
    "subtitle_bold": True,
    "subtitle_italic": False,
    "subtitle_shadow": False,
    "subtitle_position_y": 400,
    # Cost tracking (rub per minute of film with DeepSeek V4 Flash)
    "cost_per_minute": 0.0,
}


def load():
    """Load user config from JSON file, returns dict."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        merged = dict(DEFAULT_CONFIG)
        merged.update(data)
        return merged
    except (FileNotFoundError, json.JSONDecodeError):
        return dict(DEFAULT_CONFIG)


def save(config_dict):
    """Save user config dict to JSON file (atomic tmp+replace, .bak rotation)."""
    bak_path = CONFIG_PATH + ".bak"
    tmp_path = CONFIG_PATH + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
        if os.path.exists(CONFIG_PATH):
            shutil.copyfile(CONFIG_PATH, bak_path)
        os.replace(tmp_path, CONFIG_PATH)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
