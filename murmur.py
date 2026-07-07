"""Murmur — self-hosted dictation tool.

Hold the hotkey, speak, release. Whisper transcribes locally,
a local LLM (Ollama) cleans up filler/punctuation, text pastes into the
focused field. Custom dictionary biases recognition + protects
proper nouns during cleanup.
"""
import json
import math
import os
import re
import shutil
import sys
import time
import threading
import subprocess
from collections import deque
from pathlib import Path

# Add CUDA DLLs to the search path so faster-whisper can use the GPU even when the
# CUDA toolkit isn't separately installed. Must happen BEFORE faster_whisper is
# imported, and must update PATH (not just add_dll_directory) for ctranslate2's loader.
# Primary source: the nvidia-* pip packages in this venv (cuBLAS + cuDNN + NVRTC).
# Ollama's bundled libs are kept as a secondary source (cuBLAS only — no cuDNN there).
if sys.platform == 'win32':
    _nvidia_root = Path(sys.prefix) / 'Lib' / 'site-packages' / 'nvidia'
    _cuda_dirs = sorted(_nvidia_root.glob('*/bin')) if _nvidia_root.exists() else []
    _cuda_dirs += [
        Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs' / 'Ollama' / 'lib' / 'ollama' / 'cuda_v12',
        Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs' / 'Ollama' / 'lib' / 'ollama',
    ]
    _added = []
    for _p in _cuda_dirs:
        if _p.exists():
            os.environ['PATH'] = str(_p) + os.pathsep + os.environ.get('PATH', '')
            try:
                os.add_dll_directory(str(_p))
            except Exception:
                pass
            _added.append(str(_p))

CREATE_NO_WINDOW = 0x08000000 if sys.platform == 'win32' else 0

import numpy as np
import sounddevice as sd
import pyperclip
from pynput import keyboard
from pystray import Icon, Menu, MenuItem
from PIL import Image, ImageDraw, ImageTk, ImageFilter
from faster_whisper import WhisperModel
import tkinter as tk

MURMUR_DIR = Path.home() / '.murmur'
MURMUR_DIR.mkdir(exist_ok=True)
CONFIG_PATH = MURMUR_DIR / 'config.json'
DICT_PATH = MURMUR_DIR / 'dictionary.txt'
COMMANDS_PATH = MURMUR_DIR / 'commands.txt'
STATS_PATH = MURMUR_DIR / 'stats.json'
LOG_PATH = MURMUR_DIR / 'murmur.log'

_log_file = open(LOG_PATH, 'a', buffering=1, encoding='utf-8')
sys.stdout = _log_file
sys.stderr = _log_file


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)

DEFAULTS = {
    'hotkey': ['ctrl', 'cmd'],          # push-to-talk: hold to dictate
    'toggle_hotkey': ['ctrl', 'alt'],   # tap to start/stop hands-free continuous mode ([] = off)
    'model': 'large-v3-turbo',
    'compute_type': 'int8_float16',
    'device': 'cuda',
    'language': 'en',   # ISO code ('en', 'vi', ...) or 'auto' to detect per-utterance
    'cleanup_enabled': True,
    'cleanup_backend': 'ollama',     # 'ollama' (fast/free) or 'claude' (heavier)
    'cleanup_model': 'qwen2.5:7b',   # ollama model tag, or claude alias if backend=claude
    'ollama_url': 'http://localhost:11434',
    'sample_rate': 16000,
    'max_record_sec': 90,
    'voice_commands': True,   # spoken "new line", "scratch that", "send it", etc.
    'continuous_pause': 1.0,  # continuous mode: seconds of silence that ends a phrase
    'tone_matching': True,    # nudge cleanup tone to match the foreground app
    'sounds': True,           # start/stop blips
    'cue_volume': 35,         # blip volume, 0-100
    'cue_sound': 'Soft sine', # built-in preset name, or 'Custom file…'
    'cue_file': '',           # WAV path when cue_sound == 'Custom file…'
    'bubble_visible': True,
    'bubble_position': 'bottom-center',  # anchor, or 'custom' when dragged
    'bubble_x': None,     # remembered position when bubble_position == 'custom'
    'bubble_y': None,
}

CTRL_KEYS = {keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r}
WIN_KEYS = {keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r}
ALT_KEYS = {keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r, keyboard.Key.alt_gr}
SHIFT_KEYS = {keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r}

KEY_GROUPS = {
    'ctrl': CTRL_KEYS,
    'cmd': WIN_KEYS,
    'win': WIN_KEYS,
    'alt': ALT_KEYS,
    'shift': SHIFT_KEYS,
}

_MOD_ORDER = {'ctrl': 0, 'cmd': 1, 'alt': 2, 'shift': 3}
_TOKEN_NAMES = {'ctrl': 'Ctrl', 'cmd': 'Win', 'alt': 'Alt', 'shift': 'Shift',
                'space': 'Space', 'enter': 'Enter', 'tab': 'Tab', 'esc': 'Esc',
                'backspace': 'Backspace', 'delete': 'Del', 'caps_lock': 'CapsLock'}


def key_token(key):
    """Canonical token for any pynput key: a modifier name ('ctrl','cmd','alt',
    'shift'), a special-key name ('space','f13',...), or 'vk<code>' for a
    character key. Lets any key combination be a hotkey, not just modifiers."""
    if key in CTRL_KEYS:
        return 'ctrl'
    if key in WIN_KEYS:
        return 'cmd'
    if key in ALT_KEYS:
        return 'alt'
    if key in SHIFT_KEYS:
        return 'shift'
    if isinstance(key, keyboard.Key):
        return key.name
    vk = getattr(key, 'vk', None)
    if vk:
        return f'vk{vk}'          # stable across modifiers, unlike .char
    ch = getattr(key, 'char', None)
    return ch.lower() if ch else str(key)


def token_label(tok):
    """Human-readable name for a hotkey token."""
    if tok in _TOKEN_NAMES:
        return _TOKEN_NAMES[tok]
    if tok.startswith('vk'):
        try:
            vk = int(tok[2:])
        except ValueError:
            return tok
        if 65 <= vk <= 90 or 48 <= vk <= 57:
            return chr(vk)                 # A-Z, 0-9
        if 112 <= vk <= 123:
            return f'F{vk - 111}'          # F1-F12
        if 96 <= vk <= 105:
            return f'Num{vk - 96}'
        return f'VK{vk}'
    if len(tok) > 1 and tok[0] == 'f' and tok[1:].isdigit():
        return tok.upper()                 # f13 -> F13
    return tok.upper() if len(tok) == 1 else tok.capitalize()


def order_tokens(tokens):
    return sorted(tokens, key=lambda t: (_MOD_ORDER.get(t, 9), t))


def tokens_label(tokens):
    return ' + '.join(token_label(t) for t in order_tokens(tokens)) if tokens else 'None'


def load_config():
    if not CONFIG_PATH.exists():
        save_config(DEFAULTS)
        return DEFAULTS.copy()
    cfg = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    for k, v in DEFAULTS.items():
        cfg.setdefault(k, v)
    return cfg


def save_config(cfg):
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding='utf-8')


def load_dictionary():
    if not DICT_PATH.exists():
        DICT_PATH.write_text(
            '# One term per line. Lines starting with # are ignored.\n'
            '# These bias Whisper recognition and are protected during cleanup.\n'
            '# Examples:\n'
            '# DaVinci Resolve\n'
            '# faster-whisper\n'
            '# pynput\n',
            encoding='utf-8',
        )
        return []
    lines = DICT_PATH.read_text(encoding='utf-8').splitlines()
    return [l.strip() for l in lines if l.strip() and not l.startswith('#')]


def load_stats():
    if not STATS_PATH.exists():
        return {'total_words': 0, 'sessions': 0, 'last_session': None}
    try:
        return json.loads(STATS_PATH.read_text(encoding='utf-8'))
    except Exception:
        return {'total_words': 0, 'sessions': 0, 'last_session': None}


def save_stats(s):
    STATS_PATH.write_text(json.dumps(s, indent=2), encoding='utf-8')


class Recorder:
    def __init__(self, sample_rate=16000, max_sec=90):
        self.sr = sample_rate
        self.max_frames = sample_rate * max_sec
        self.frames = []
        self.stream = None
        self.recording = False
        self.level = 0.0  # rolling RMS of the latest chunk — drives the waveform UI

    def _callback(self, indata, frames, time_info, status):
        if self.recording:
            self.frames.append(indata.copy())
            self.level = float(np.sqrt(np.mean(indata ** 2)))

    def start(self):
        self.frames = []
        self.recording = True
        self.stream = sd.InputStream(
            samplerate=self.sr,
            channels=1,
            dtype='float32',
            callback=self._callback,
        )
        self.stream.start()

    def stop(self):
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        if not self.frames:
            return None
        audio = np.concatenate(self.frames, axis=0).flatten()
        if len(audio) > self.max_frames:
            audio = audio[: self.max_frames]
        return audio


class Transcriber:
    """In-process faster-whisper: loads on GPU when available, falls back to CPU/int8.
    Model is preloaded (and CUDA kernels warmed) at startup so the first dictation
    doesn't pay the cold-start cost."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self._lock = threading.Lock()

    def load(self):
        with self._lock:
            if self.model is not None:
                return
            requested_device = self.cfg.get('device', 'cuda')
            if requested_device == 'cuda':
                try:
                    model = WhisperModel(
                        self.cfg['model'],
                        device='cuda',
                        compute_type=self.cfg.get('compute_type', 'int8_float16'),
                    )
                    # Warm-up pass forces CUDA kernel/cuDNN load now, not mid-dictation
                    dummy = np.zeros(16000, dtype=np.float32)
                    segs, _ = model.transcribe(dummy, language='en', vad_filter=False, beam_size=1)
                    list(segs)
                    self.model = model
                    log('whisper loaded on GPU')
                    return
                except Exception as e:
                    log(f'GPU load failed ({e}); using CPU/int8')
            self.model = WhisperModel(self.cfg['model'], device='cpu', compute_type='int8')
            log('whisper loaded on CPU')

    def transcribe(self, audio, dictionary):
        """Returns (text, language) — language is the configured code, or the
        Whisper-detected one when config is 'auto'. Used to pick the cleanup prompt."""
        self.load()  # no-op once loaded
        prompt = ', '.join(dictionary) if dictionary else None
        lang = self.cfg.get('language', 'en')
        segments, info = self.model.transcribe(
            audio,
            language=None if lang == 'auto' else lang,
            initial_prompt=prompt,
            vad_filter=True,
        )
        text = ' '.join(s.text.strip() for s in segments).strip()
        used_lang = info.language if lang == 'auto' else lang
        return text, (used_lang or 'en')


CLEANUP_PROMPT = """You clean up dictated text. Follow the rules and study the examples.

RULES:
- Remove meaningless filler words (um, uh, like, you know) — keep them only when they carry meaning.
- Keep hedges and opinion markers (I think, maybe, probably, kind of) — they carry meaning and are NOT fillers.
- This is a verbatim transcript of the user's own private dictation. Preserve profanity, slang, and crude language EXACTLY as spoken — never censor, soften, star out, or replace words. You are a transcription engine, not a content moderator: never refuse, apologize, or comment on the content.
- Add natural punctuation and capitalization.
- Always end statements with a period, questions with a question mark.
- Capitalize proper nouns, product names, and brand names (e.g., Whisper Flow, Claude, Bambu, DaVinci Resolve).
- Apply spoken self-corrections ("scratch that", "actually I meant", "no wait") — remove the struck-out portion, keep only the corrected version.
- Preserve these terms EXACTLY as written: {dictionary}
- Do NOT add words that weren't spoken. Do NOT change meaning. Do NOT formalize the tone. Do NOT wrap in quotes. Do NOT add preamble or explanation.

EXAMPLES:

INPUT: hey can you um send me the file when you get a chance
CLEANED: Hey can you send me the file when you get a chance?

INPUT: i was thinking we could go to the store actually no the park
CLEANED: I was thinking we could go to the park.

INPUT: testing this whisper flow thing
CLEANED: Testing this Whisper Flow thing.

INPUT: yeah so the bambu printer is having issues again
CLEANED: Yeah, so the Bambu printer is having issues again.

INPUT: not quite as good cleanup as whisper flow though
CLEANED: Not quite as good cleanup as Whisper Flow though.

INPUT: this is using it
CLEANED: This is using it.

INPUT: murmur
CLEANED: Murmur.

INPUT: so um the chorus needs like more energy i think
CLEANED: So the chorus needs more energy, I think.

INPUT: this fucking mix sounds like shit right now
CLEANED: This fucking mix sounds like shit right now.

NOW CLEAN THIS DICTATION:
INPUT: {text}
CLEANED:"""


CLEANUP_PROMPT_VI = """You clean up dictated Vietnamese text. Follow the rules and study the examples.

RULES:
- The dictation is in Vietnamese. The cleaned output MUST stay in Vietnamese — NEVER translate to English.
- Preserve all Vietnamese diacritics exactly (ă â đ ê ô ơ ư and tone marks).
- Remove meaningless filler words (ừm, ờ, à, kiểu như, đại loại là, nói chung là) — keep them only when they carry meaning.
- Add natural punctuation and capitalization.
- Always end statements with a period, questions with a question mark.
- Capitalize proper nouns, product names, and brand names.
- Apply spoken self-corrections ("à không", "nhầm", "ý là") — remove the struck-out portion, keep only the corrected version.
- Keep English words spoken mid-sentence (code-switching) exactly as spoken — do not translate them to Vietnamese.
- Copy every remaining word EXACTLY as spoken — never substitute a similar word. Never swap nha→nhé, đấy→đó, với lại→và, or any casual word for a formal one.
- This is a verbatim transcript of the user's own private dictation. Preserve profanity, slang, and crude language EXACTLY as spoken — never censor, soften, or replace words. You are a transcription engine, not a content moderator: never refuse, apologize, or comment on the content.
- Preserve these terms EXACTLY as written: {dictionary}
- Do NOT add words that weren't spoken. Do NOT change meaning. Do NOT formalize the tone. Do NOT wrap in quotes. Do NOT add preamble or explanation.

EXAMPLES:

INPUT: ừm cho mình xin cái file khi nào bạn rảnh nhé
CLEANED: Cho mình xin cái file khi nào bạn rảnh nhé.

INPUT: bài này cần thêm bass với lại trống nữa
CLEANED: Bài này cần thêm bass với lại trống nữa.

INPUT: hẹn gặp lúc 2 giờ à không 3 giờ chiều
CLEANED: Hẹn gặp lúc 3 giờ chiều.

INPUT: kiểu như là mình thấy bản mix này hơi đục ở phần trầm
CLEANED: Mình thấy bản mix này hơi đục ở phần trầm.

INPUT: anh gửi em cái project studio board qua email nhé
CLEANED: Anh gửi em cái project Studio Board qua email nhé.

INPUT: cái này dùng được không nhỉ
CLEANED: Cái này dùng được không nhỉ?

NOW CLEAN THIS DICTATION:
INPUT: {text}
CLEANED:"""


# Prompt per transcription language; anything not listed falls back to English.
CLEANUP_PROMPTS = {
    'en': CLEANUP_PROMPT,
    'vi': CLEANUP_PROMPT_VI,
}


_REFUSAL_MARKERS = (
    'i cannot', "i can't", 'i will not', "i won't", "i'm sorry", 'i am sorry',
    'as an ai', 'unable to assist', 'cannot assist', 'cannot clean', 'will not produce',
)


def _looks_like_refusal(out, original):
    """A cleanup model must always return cleaned text. If the output opens with
    refusal language that was NOT part of the dictation itself, the model is
    moderating instead of cleaning — caller should paste the raw transcript."""
    head = out[:100].lower()
    orig = original.lower()
    return any(m in head and m not in orig for m in _REFUSAL_MARKERS)


def _build_cleanup_prompt(text, dictionary, lang, tone):
    dict_str = ', '.join(dictionary) if dictionary else 'none'
    prompt = CLEANUP_PROMPTS.get(lang, CLEANUP_PROMPT).format(dictionary=dict_str, text=text)
    if tone:  # inject a tone rule at the top of the RULES block (both languages have it)
        prompt = prompt.replace('RULES:\n', f'RULES:\n- {tone}\n', 1)
    return prompt


def clean_with_ollama(text, dictionary, model, url, lang='en', tone=None):
    """Cleanup via local Ollama daemon. Returns (text, backend_ok); backend_ok is
    False when the daemon is unreachable (so the caller can flag 'cleanup offline'
    instead of silently pasting the raw transcript)."""
    if not text.strip():
        return text, True
    import urllib.request
    import urllib.error
    prompt = _build_cleanup_prompt(text, dictionary, lang, tone)
    payload = json.dumps({
        'model': model,
        'prompt': prompt,
        'stream': False,
        'keep_alive': '30m',  # stay resident between dictations — avoids ~40s cold reload
        'options': {
            'temperature': 0.1,
            'num_predict': 512,
            'top_p': 0.9,
        },
    }).encode('utf-8')
    req = urllib.request.Request(
        f'{url.rstrip("/")}/api/generate',
        data=payload,
        headers={'Content-Type': 'application/json'},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
            out = (data.get('response') or '').strip()
            if out.startswith('"') and out.endswith('"') and len(out) > 1:
                out = out[1:-1]
            # qwen models sometimes prepend "CLEANED:" or similar
            for prefix in ('CLEANED:', 'Cleaned:', 'Output:', 'Result:'):
                if out.startswith(prefix):
                    out = out[len(prefix):].strip()
            if out and _looks_like_refusal(out, text):
                log('cleanup model refused — pasting raw transcript instead')
                return text, True
            return (out or text), True
    except urllib.error.URLError as e:
        log(f'ollama URL error: {e}')
        return text, False  # daemon unreachable
    except Exception as e:
        log(f'ollama cleanup error: {e}')
        return text, True   # reachable but some other error — don't flag offline


def find_claude_cli():
    candidates = [
        Path.home() / '.local' / 'bin' / 'claude.exe',
        Path.home() / '.local' / 'bin' / 'claude',
        Path(os.environ.get('APPDATA', '')) / 'npm' / 'claude.cmd',
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    via_path = shutil.which('claude') or shutil.which('claude.exe') or shutil.which('claude.cmd')
    return via_path


CLAUDE_CLI = find_claude_cli()


# --- Reliability: Ollama autostart + run-on-login ---------------------------

OLLAMA_APP = Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs' / 'Ollama' / 'ollama app.exe'
_PROGRAMS = Path(os.environ.get('APPDATA', '')) / 'Microsoft' / 'Windows' / 'Start Menu' / 'Programs'
STARTUP_LNK = _PROGRAMS / 'Startup' / 'Murmur.lnk'   # launches at login
START_MENU_LNK = _PROGRAMS / 'Murmur.lnk'            # searchable in the Start menu
ICON_ICO = Path(__file__).parent / 'assets' / 'icon.ico'


def ollama_reachable(url, timeout=1.5):
    import urllib.request
    try:
        with urllib.request.urlopen(f'{url.rstrip("/")}/api/tags', timeout=timeout) as r:
            return getattr(r, 'status', 200) == 200
    except Exception:
        return False


def ensure_ollama_running(url, wait=12.0):
    """Return True if Ollama answers; if not, try to launch the bundled app and
    wait briefly for it to come up. Prevents silent raw-transcript fallback."""
    if ollama_reachable(url):
        return True
    if sys.platform == 'win32' and OLLAMA_APP.exists():
        try:
            subprocess.Popen([str(OLLAMA_APP)], creationflags=CREATE_NO_WINDOW)
            log('Ollama not reachable — launching it')
        except Exception as e:
            log(f'failed to launch Ollama: {e}')
            return False
        deadline = time.time() + wait
        while time.time() < deadline:
            time.sleep(0.5)
            if ollama_reachable(url):
                log('Ollama is up')
                return True
    return ollama_reachable(url)


def _make_shortcut(lnk):
    """Create a .lnk that launches Murmur windowless, with the app icon."""
    lnk.parent.mkdir(parents=True, exist_ok=True)
    pyw = str(Path(sys.executable).with_name('pythonw.exe'))
    script = str(Path(__file__).resolve())
    workdir = str(Path(__file__).resolve().parent)
    icon = f"$s.IconLocation='{ICON_ICO}';" if ICON_ICO.exists() else ''
    ps = (
        "$s=(New-Object -ComObject WScript.Shell).CreateShortcut('%s');"
        "$s.TargetPath='%s';$s.Arguments='\"%s\"';"
        "$s.WorkingDirectory='%s';%s"
        "$s.Description='Murmur - local voice dictation';$s.WindowStyle=7;$s.Save()"
    ) % (lnk, pyw, script, workdir, icon)
    subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', ps],
                   creationflags=CREATE_NO_WINDOW, timeout=15)


def _remove_shortcut(lnk):
    try:
        lnk.unlink()
    except FileNotFoundError:
        pass
    except Exception as e:
        log(f'failed to remove shortcut {lnk.name}: {e}')


def startup_enabled():
    return STARTUP_LNK.exists()


def set_startup(enable):
    """Create/remove a Startup-folder shortcut so Murmur launches at login."""
    if enable:
        try:
            _make_shortcut(STARTUP_LNK)
        except Exception as e:
            log(f'failed to create startup shortcut: {e}')
    else:
        _remove_shortcut(STARTUP_LNK)


_SINGLE_INSTANCE_HANDLE = None


def acquire_single_instance(name='Murmur-SingleInstance'):
    """Hold a named mutex so only one Murmur runs. Returns False if another
    instance already owns it (the caller should then exit)."""
    global _SINGLE_INSTANCE_HANDLE
    if sys.platform != 'win32':
        return True
    try:
        import ctypes
        k = ctypes.WinDLL('kernel32', use_last_error=True)
        k.CreateMutexW.restype = ctypes.c_void_p
        k.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
        h = k.CreateMutexW(None, False, name)
        err = ctypes.get_last_error()
        ERROR_ALREADY_EXISTS = 183
        if not h or err == ERROR_ALREADY_EXISTS:
            return False
        _SINGLE_INSTANCE_HANDLE = h  # keep alive for the process lifetime
        return True
    except Exception as e:
        log(f'single-instance check failed ({e}); continuing')
        return True


def start_menu_enabled():
    return START_MENU_LNK.exists()


def set_start_menu(enable):
    """Create/remove a Start-menu shortcut (searchable as 'Murmur')."""
    if enable:
        try:
            _make_shortcut(START_MENU_LNK)
            log('Start Menu shortcut created')
        except Exception as e:
            log(f'failed to create Start Menu shortcut: {e}')
    else:
        _remove_shortcut(START_MENU_LNK)


def clean_with_claude(text, dictionary, model='haiku', lang='en', tone=None):
    """Cleanup via the Claude Code CLI on the user's Max plan. Returns (text, ok).

    Strips API-key env vars before invoking so a stale key cannot route
    the call to the paid API — only the OAuth/Max session is used.
    """
    if not text.strip():
        return text, True
    if not CLAUDE_CLI:
        print('[murmur] claude.exe not found on PATH or ~/.local/bin — skipping cleanup', file=sys.stderr)
        return text, False
    prompt = _build_cleanup_prompt(text, dictionary, lang, tone)
    env = os.environ.copy()
    for var in ('ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'CLAUDE_API_KEY', 'CLAUDE_CODE_API_KEY'):
        env.pop(var, None)
    try:
        result = subprocess.run(
            [CLAUDE_CLI, '-p', '--model', model],
            input=prompt,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=20,
            env=env,
            creationflags=CREATE_NO_WINDOW,
        )
        out = (result.stdout or '').strip()
        if result.returncode == 0 and out:
            if out.startswith('"') and out.endswith('"') and len(out) > 1:
                out = out[1:-1]
            if _looks_like_refusal(out, text):
                log('cleanup model refused — pasting raw transcript instead')
                return text, True
            return out, True
        print(f'[murmur] cleanup non-zero exit {result.returncode}: {result.stderr}', file=sys.stderr)
        return text, False
    except Exception as e:
        print(f'[murmur] cleanup error: {e}', file=sys.stderr)
        return text, False


# --- Voice commands, tone matching, sound cues ------------------------------

# Action name -> keystroke recipe. 'taps' = keys pressed one at a time;
# 'chord' = modifiers held while the last key is pressed.
COMMAND_ACTIONS = {
    'newline':   {'taps': [keyboard.Key.enter]},
    'paragraph': {'taps': [keyboard.Key.enter, keyboard.Key.enter]},
    'send':      {'taps': [keyboard.Key.enter]},
    'enter':     {'taps': [keyboard.Key.enter]},
    'tab':       {'taps': [keyboard.Key.tab]},
    'escape':    {'taps': [keyboard.Key.esc]},
    'space':     {'taps': [keyboard.Key.space]},
    'backspace': {'taps': [keyboard.Key.backspace]},
    'undo':      {'chord': [keyboard.Key.ctrl, 'z']},
    'redo':      {'chord': [keyboard.Key.ctrl, keyboard.Key.shift, 'z']},
    'selectall': {'chord': [keyboard.Key.ctrl, 'a']},
    'copy':      {'chord': [keyboard.Key.ctrl, 'c']},
    'cut':       {'chord': [keyboard.Key.ctrl, 'x']},
    'paste':     {'chord': [keyboard.Key.ctrl, 'v']},
    'save':      {'chord': [keyboard.Key.ctrl, 's']},
}

DEFAULT_COMMANDS_TEXT = """# Voice commands — say the phrase on its own to trigger the action.
# Format:  phrase = action      (lines starting with # are ignored)
#
# Available actions:
#   newline  paragraph  send  enter  tab  escape  space  backspace
#   undo  redo  selectall  copy  cut  paste  save
#
# Edit freely: add phrases, remap them, or delete ones you don't want.

new line = newline
line break = newline
new paragraph = paragraph
scratch that = undo
delete that = undo
send it = send
press enter = send
select all = selectall
"""


def load_commands():
    """Parse ~/.murmur/commands.txt into {normalized phrase: action}. Writes the
    default file on first run. Unknown actions are skipped."""
    if not COMMANDS_PATH.exists():
        COMMANDS_PATH.write_text(DEFAULT_COMMANDS_TEXT, encoding='utf-8')
    cmds = {}
    for line in COMMANDS_PATH.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        phrase, action = line.split('=', 1)
        action = action.strip().lower()
        phrase = re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', phrase).strip().lower())
        if phrase and action in COMMAND_ACTIONS:
            cmds[phrase] = action
    return cmds


def match_voice_command(text, commands):
    """If the whole utterance is a command phrase, return its action key."""
    t = re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', text or '').strip().lower())
    return commands.get(t)


def do_voice_command(action, controller):
    recipe = COMMAND_ACTIONS.get(action)
    if not recipe:
        return
    if 'taps' in recipe:
        for key in recipe['taps']:
            controller.press(key)
            controller.release(key)
    elif 'chord' in recipe:
        keys = recipe['chord']
        for key in keys:
            controller.press(key)
        for key in reversed(keys):
            controller.release(key)


# Foreground app → cleanup tone. Light touch: casual for chat, clean for docs/email.
TONE_RULES = {
    'casual': 'Keep the tone casual and conversational, exactly as spoken; do not formalize.',
    'clean': 'Use clean grammar and professional wording, but never add or drop meaning.',
}
APP_TONE = {
    'discord.exe': 'casual', 'slack.exe': 'casual', 'telegram.exe': 'casual',
    'whatsapp.exe': 'casual', 'teams.exe': 'casual', 'ms-teams.exe': 'casual',
    'outlook.exe': 'clean', 'winword.exe': 'clean', 'onenote.exe': 'clean',
    'notion.exe': 'clean',
}
TITLE_TONE = [('gmail', 'clean'), ('outlook', 'clean'), ('docs.google', 'clean'),
              ('- word', 'clean'), ('slack', 'casual'), ('discord', 'casual'),
              ('messenger', 'casual'), ('whatsapp', 'casual')]


def foreground_app():
    """(exe_name_lower, window_title_lower) of the foreground window."""
    if sys.platform != 'win32':
        return ('', '')
    try:
        import ctypes
        from ctypes import wintypes
        u, k = ctypes.windll.user32, ctypes.windll.kernel32
        u.GetForegroundWindow.restype = ctypes.c_void_p
        hwnd = u.GetForegroundWindow()
        if not hwnd:
            return ('', '')
        pid = wintypes.DWORD()
        u.GetWindowThreadProcessId(ctypes.c_void_p(hwnd), ctypes.byref(pid))
        n = u.GetWindowTextLengthW(ctypes.c_void_p(hwnd))
        buf = ctypes.create_unicode_buffer(n + 1)
        u.GetWindowTextW(ctypes.c_void_p(hwnd), buf, n + 1)
        title = (buf.value or '').lower()
        exe = ''
        k.OpenProcess.restype = ctypes.c_void_p
        h = k.OpenProcess(0x1000, False, pid.value)  # PROCESS_QUERY_LIMITED_INFORMATION
        if h:
            size = wintypes.DWORD(260)
            pbuf = ctypes.create_unicode_buffer(260)
            if k.QueryFullProcessImageNameW(ctypes.c_void_p(h), 0, pbuf, ctypes.byref(size)):
                exe = Path(pbuf.value).name.lower()
            k.CloseHandle(ctypes.c_void_p(h))
        return (exe, title)
    except Exception:
        return ('', '')


def tone_for_foreground():
    """Return a tone instruction string for the current foreground app, or None."""
    exe, title = foreground_app()
    label = APP_TONE.get(exe)
    if not label:
        for kw, t in TITLE_TONE:
            if kw in title:
                label = t
                break
    return TONE_RULES.get(label)


_CUE_CACHE = {}
CUE_SR = 44100
CUSTOM_LABEL = 'Custom file…'

# Built-in blip samples. Each has a higher 'start' and lower 'stop' pitch.
CUE_PRESETS = {
    'Soft sine': dict(start=587, stop=440, ms=150, shape='sine'),
    'Blip':      dict(start=880, stop=620, ms=70,  shape='sine'),
    'Marimba':   dict(start=659, stop=440, ms=240, shape='decay'),
    'Chime':     dict(start=784, stop=523, ms=300, shape='chime'),
    'Pop':       dict(start=1100, stop=760, ms=40, shape='sine'),
}


def _synth(freq, ms, shape, sr=CUE_SR):
    """Generate a unit-volume float32 blip of the given timbre."""
    n = int(sr * ms / 1000)
    t = np.arange(n) / sr
    if shape == 'decay':          # quick attack, exponential decay — marimba-ish
        sig = np.sin(2 * np.pi * freq * t) * np.exp(-t * 18.0)
    elif shape == 'chime':        # a few harmonics with a longer decay
        sig = (np.sin(2 * np.pi * freq * t)
               + 0.5 * np.sin(2 * np.pi * freq * 2 * t)
               + 0.25 * np.sin(2 * np.pi * freq * 3 * t)) / 1.75 * np.exp(-t * 8.0)
    else:                         # sine with a clickless raised-cosine envelope
        env = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / max(1, n - 1))
        sig = np.sin(2 * np.pi * freq * t) * env
    return sig.astype('float32')


def _preset_sample(name, kind):
    key = ('preset', name, kind)
    if key not in _CUE_CACHE:
        p = CUE_PRESETS.get(name, CUE_PRESETS['Soft sine'])
        _CUE_CACHE[key] = (_synth(p['start'] if kind == 'start' else p['stop'],
                                  p['ms'], p['shape']), CUE_SR)
    return _CUE_CACHE[key]


def _load_wav(path):
    """Load a WAV file to a unit-peak float32 mono array + samplerate (cached)."""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None, None
    key = ('file', path, mtime)
    if key not in _CUE_CACHE:
        try:
            import wave
            with wave.open(path, 'rb') as w:
                ch, sw, sr = w.getnchannels(), w.getsampwidth(), w.getframerate()
                frames = w.readframes(w.getnframes())
            if sw == 2:
                a = np.frombuffer(frames, dtype='<i2').astype('float32') / 32768.0
            elif sw == 1:
                a = (np.frombuffer(frames, dtype=np.uint8).astype('float32') - 128) / 128.0
            elif sw == 4:
                a = np.frombuffer(frames, dtype='<i4').astype('float32') / 2147483648.0
            else:
                raise ValueError(f'unsupported sample width {sw}')
            if ch > 1:
                a = a.reshape(-1, ch).mean(axis=1)
            peak = float(np.max(np.abs(a))) or 1.0
            _CUE_CACHE[key] = ((a / peak * 0.9).astype('float32'), sr)
        except Exception as e:
            log(f'cue wav load failed: {e}')
            _CUE_CACHE[key] = (None, None)
    return _CUE_CACHE[key]


def cue_sample(kind, sound, file):
    if sound == CUSTOM_LABEL and file:
        arr, sr = _load_wav(file)
        if arr is not None:
            return arr, sr  # else fall through to the default preset
    return _preset_sample(sound if sound in CUE_PRESETS else 'Soft sine', kind)


def play_cue(kind, volume=0.35, sound='Soft sine', file=''):
    """Soft, non-blocking activation blip through the default output device.
    Uses sounddevice (same stack as recording); winsound is unreliable from
    the console-less pythonw process."""
    if volume <= 0:
        return
    arr, sr = cue_sample(kind, sound, file)
    if arr is None:
        return
    data = (arr * float(volume)).astype('float32')

    def _play():
        try:
            sd.play(data, sr)
        except Exception as e:
            log(f'cue play failed: {e}')
    threading.Thread(target=_play, daemon=True).start()


_last_paste_time = [0.0]


def paste_text(text, controller):
    if not text:
        return
    # Continuation heuristic: if we pasted within the last 30s, the user is most
    # likely continuing in the same field — prepend a space so sentences don't run
    # together. Past 30s, assume new context, no leading space.
    if time.time() - _last_paste_time[0] < 30 and not text.startswith((' ', '\n', '\t')):
        text = ' ' + text
    try:
        prev = pyperclip.paste()
    except Exception:
        prev = None
    pyperclip.copy(text)
    time.sleep(0.05)
    controller.press(keyboard.Key.ctrl)
    controller.press('v')
    controller.release('v')
    controller.release(keyboard.Key.ctrl)
    _last_paste_time[0] = time.time()
    if prev is not None:
        threading.Timer(0.4, lambda: _safe_clip(prev)).start()


def _safe_clip(value):
    try:
        pyperclip.copy(value)
    except Exception:
        pass


def taskbar_height(default=48):
    """Height of the Windows taskbar — the pill matches it exactly."""
    if sys.platform != 'win32':
        return default
    try:
        import ctypes
        class RECT(ctypes.Structure):
            _fields_ = [('l', ctypes.c_long), ('t', ctypes.c_long),
                        ('r', ctypes.c_long), ('b', ctypes.c_long)]
        rect = RECT()
        ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0)  # SPI_GETWORKAREA
        screen_h = ctypes.windll.user32.GetSystemMetrics(1)  # SM_CYSCREEN
        h = screen_h - rect.b
        return h if 20 <= h <= 90 else default  # sane range; taskbar may be hidden/side-docked
    except Exception:
        return default


def apply_no_activate(win):
    """Keep a floating Tk window from stealing keyboard focus when clicked —
    paste must go to the window the user was typing in, not the overlay."""
    if sys.platform != 'win32':
        return
    try:
        import ctypes
        GWL_EXSTYLE = -20
        WS_EX_NOACTIVATE = 0x08000000
        WS_EX_TOOLWINDOW = 0x00000080
        win.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(win.winfo_id()) or win.winfo_id()
        ex = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        ctypes.windll.user32.SetWindowLongW(
            hwnd, GWL_EXSTYLE, ex | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW
        )
    except Exception as e:
        log(f'no-activate flag failed: {e}')


class Bubble:
    """Wispr-Flow-style compact pill, bottom-center by default. A true rounded
    capsule (color-key transparency outside the shape). Idle: dim dots.
    Listening: live waveform bars from the mic level. Processing: pulsing dots.
    Drag to move (position remembered). Double-click expands into the settings
    window. Right-click hides it (re-enable from the tray menu)."""

    BARS = 15
    TICK_MS = 50  # waveform refresh — one new bar every tick, scrolls right-to-left
    DRAG_THRESHOLD = 6  # pixels — moves above this trigger drag instead of click
    KEY = '#010203'  # transparency color key — anything this color is see-through
    PILL_BG = '#1c1c1c'
    PILL_EDGE = '#3a3a3a'
    PAD = 12  # transparent margin around the pill that holds the drop shadow

    def __init__(self, root, app=None):
        self.root = root
        self.app = app
        self.W = 160
        self.H = max(28, taskbar_height() - 8)  # a touch shorter than the taskbar
        self.win = None
        self.canvas = None
        self._levels = None
        self._anim_job = None
        self._mode = None
        self._phase = 0
        self._layered = False
        self._photo = None
        self._base = None          # cached shadow + pill body (constant per size)
        self._last_img = None      # last full frame, for fade re-paints
        self._base_opacity = 0.95
        self.dragging = False
        self.press_root = (0, 0)
        self.win_start = (0, 0)

    @property
    def FW(self):
        return self.W + 2 * self.PAD

    @property
    def FH(self):
        return self.H + 2 * self.PAD

    @property
    def enabled(self):
        return bool(self.app is None or self.app.cfg.get('bubble_visible', True))

    def _ensure(self):
        if self.win is not None:
            return
        self.win = tk.Toplevel(self.root)
        self.win.overrideredirect(True)
        self.win.attributes('-topmost', True)
        self.win.configure(bg=self.KEY)
        apply_no_activate(self.win)
        # Per-pixel alpha via UpdateLayeredWindow — how Electron/Qt apps get smooth
        # window edges. WS_EX_LAYERED is (re)applied per-frame in _push_frame rather
        # than cached: Tk recreates the OS window when -topmost/overrideredirect/
        # deiconify are applied, so any HWND captured here goes stale.
        self._layered = (sys.platform == 'win32')
        self.canvas = tk.Canvas(
            self.win, width=self.FW, height=self.FH,
            bg=self.KEY, highlightthickness=0, cursor='hand2',
        )
        self.canvas.pack()
        # Drag to move, double-click for settings, right-click to hide
        self.canvas.bind('<ButtonPress-1>', self._on_press)
        self.canvas.bind('<B1-Motion>', self._on_move)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)
        self.canvas.bind('<Double-Button-1>', self._on_double_click)
        self.win.withdraw()
        self._assert_topmost()  # starts the keep-on-top watchdog loop

    SS = 4  # supersample factor — render 4x with PIL, downscale for smooth curves

    def _pill_base(self):
        """Cached full-window frame: soft drop shadow + the capsule body. Constant
        per size, so the blur is computed once and reused every animation frame."""
        if self._base is not None:
            return self._base
        S, P, W, H = self.SS, self.PAD, self.W, self.H
        big = Image.new('RGBA', (self.FW * S, self.FH * S), (0, 0, 0, 0))
        # Drop shadow: the pill silhouette, nudged down, Gaussian-blurred.
        shadow = Image.new('RGBA', (self.FW * S, self.FH * S), (0, 0, 0, 0))
        ImageDraw.Draw(shadow).rounded_rectangle(
            [P * S, (P + 3) * S, (P + W) * S, (P + H + 3) * S],
            radius=(H * S) // 2, fill=(0, 0, 0, 150))
        shadow = shadow.filter(ImageFilter.GaussianBlur(P * S * 0.5))
        big.alpha_composite(shadow)
        # Capsule body on top of the shadow.
        m = S
        ImageDraw.Draw(big).rounded_rectangle(
            [P * S + m, P * S + m, (P + W) * S - m, (P + H) * S - m],
            radius=(H * S - 2 * m) // 2,
            fill=self.PILL_BG, outline=self.PILL_EDGE, width=S)
        self._base = big.resize((self.FW, self.FH), Image.LANCZOS)
        return self._base

    def _render(self, draw_content, opacity=0.95):
        """Compose one frame: cached shadow+pill base, then the state content
        (bars/dots) drawn 4x-supersampled in pill-local coords and pasted in."""
        S = self.SS
        frame = self._pill_base().copy()
        content = Image.new('RGBA', (self.W * S, self.H * S), (0, 0, 0, 0))
        draw_content(ImageDraw.Draw(content), S)
        frame.alpha_composite(content.resize((self.W, self.H), Image.LANCZOS),
                              (self.PAD, self.PAD))
        self._last_img = frame
        self._base_opacity = opacity
        self._paint(frame, opacity)

    def _paint(self, img, opacity):
        """Push a composed RGBA frame to the window (per-pixel alpha, or color-key
        fallback). Kept separate from _render so fades can re-push the same frame."""
        if self._layered:
            ok = False
            try:
                ok = self._push_frame(img, opacity)
            except Exception as e:
                log(f'layered paint raised, using color-key fallback: {e}')
            if ok:
                return
            # ULW failed — never leave a solid-black layered window on screen.
            log('layered paint failed; falling back to color-key rendering')
            self._layered = False
        # Fallback (non-Windows or layered failure): color-key transparency
        try:
            self.win.attributes('-transparentcolor', self.KEY)
            self.win.attributes('-alpha', opacity)
        except Exception:
            pass
        flat = Image.new('RGB', img.size, self.KEY)
        flat.paste(img, (0, 0), img)
        self._photo = ImageTk.PhotoImage(flat)  # keep the ref or Tk drops the image
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor='nw', image=self._photo)

    def _fade(self, frm, to, step=0, steps=7, done=None):
        """Ramp window opacity by re-pushing the last frame at scaled alpha."""
        if self._last_img is None:
            if done:
                done()
            return
        f = frm + (to - frm) * (step / steps)
        self._paint(self._last_img, self._base_opacity * f)
        if step < steps:
            self.win.after(16, lambda: self._fade(frm, to, step + 1, steps, done))
        elif done:
            done()

    def _push_frame(self, img, opacity):
        """Blit an RGBA PIL frame onto this window via UpdateLayeredWindow —
        the compositor blends our alpha channel per pixel (DWM does the rest)."""
        import ctypes

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [('biSize', ctypes.c_uint32), ('biWidth', ctypes.c_int32),
                        ('biHeight', ctypes.c_int32), ('biPlanes', ctypes.c_uint16),
                        ('biBitCount', ctypes.c_uint16), ('biCompression', ctypes.c_uint32),
                        ('biSizeImage', ctypes.c_uint32), ('biXPelsPerMeter', ctypes.c_int32),
                        ('biYPelsPerMeter', ctypes.c_int32), ('biClrUsed', ctypes.c_uint32),
                        ('biClrImportant', ctypes.c_uint32)]

        class BLENDFUNCTION(ctypes.Structure):
            _fields_ = [('BlendOp', ctypes.c_ubyte), ('BlendFlags', ctypes.c_ubyte),
                        ('SourceConstantAlpha', ctypes.c_ubyte), ('AlphaFormat', ctypes.c_ubyte)]

        class POINT(ctypes.Structure):
            _fields_ = [('x', ctypes.c_long), ('y', ctypes.c_long)]

        class SIZE(ctypes.Structure):
            _fields_ = [('cx', ctypes.c_long), ('cy', ctypes.c_long)]

        user32, gdi32 = ctypes.windll.user32, ctypes.windll.gdi32
        # HANDLE-returning calls must be c_void_p or the 64-bit value truncates.
        user32.GetParent.restype = ctypes.c_void_p
        user32.GetParent.argtypes = [ctypes.c_void_p]
        user32.GetWindowLongW.restype = ctypes.c_long
        user32.GetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int]
        user32.SetWindowLongW.restype = ctypes.c_long
        user32.SetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_long]
        user32.GetDC.restype = ctypes.c_void_p
        gdi32.CreateCompatibleDC.restype = ctypes.c_void_p
        gdi32.CreateDIBSection.restype = ctypes.c_void_p
        gdi32.SelectObject.restype = ctypes.c_void_p
        user32.UpdateLayeredWindow.restype = ctypes.c_int  # BOOL

        # Fresh parent HWND each frame (the real TkTopLevel; winfo_id() is a
        # non-composited TkChild) + (re)assert WS_EX_LAYERED right before the blit.
        hwnd = user32.GetParent(self.win.winfo_id()) or self.win.winfo_id()
        if not hwnd:
            return False
        GWL_EXSTYLE, WS_EX_LAYERED = -20, 0x00080000
        ex = user32.GetWindowLongW(ctypes.c_void_p(hwnd), GWL_EXSTYLE)
        if not (ex & WS_EX_LAYERED):
            user32.SetWindowLongW(ctypes.c_void_p(hwnd), GWL_EXSTYLE, ex | WS_EX_LAYERED)

        w, h = img.size
        arr = np.asarray(img, dtype=np.uint8)
        a = arr[..., 3].astype(np.uint16)
        bgra = np.empty((h, w, 4), dtype=np.uint8)
        bgra[..., 0] = (arr[..., 2].astype(np.uint16) * a // 255).astype(np.uint8)
        bgra[..., 1] = (arr[..., 1].astype(np.uint16) * a // 255).astype(np.uint8)
        bgra[..., 2] = (arr[..., 0].astype(np.uint16) * a // 255).astype(np.uint8)
        bgra[..., 3] = arr[..., 3]  # UpdateLayeredWindow wants premultiplied BGRA
        data = bgra.tobytes()

        bmi = BITMAPINFOHEADER()
        bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.biWidth, bmi.biHeight = w, -h  # negative = top-down rows
        bmi.biPlanes, bmi.biBitCount, bmi.biCompression = 1, 32, 0

        screen_dc = user32.GetDC(None)
        mem_dc = gdi32.CreateCompatibleDC(ctypes.c_void_p(screen_dc))
        bits = ctypes.c_void_p()
        hbmp = gdi32.CreateDIBSection(ctypes.c_void_p(screen_dc), ctypes.byref(bmi),
                                      0, ctypes.byref(bits), None, 0)
        ok = False
        try:
            if not hbmp or not bits:
                return False
            ctypes.memmove(bits, data, len(data))
            old = gdi32.SelectObject(ctypes.c_void_p(mem_dc), ctypes.c_void_p(hbmp))
            blend = BLENDFUNCTION(0, 0, int(opacity * 255), 1)  # AC_SRC_OVER / AC_SRC_ALPHA
            size, src = SIZE(w, h), POINT(0, 0)
            ULW_ALPHA = 2
            res = user32.UpdateLayeredWindow(
                ctypes.c_void_p(hwnd), ctypes.c_void_p(screen_dc),
                None, ctypes.byref(size),
                ctypes.c_void_p(mem_dc), ctypes.byref(src),
                0, ctypes.byref(blend), ULW_ALPHA)
            ok = bool(res)
            if not ok:
                log('UpdateLayeredWindow returned FALSE')
            gdi32.SelectObject(ctypes.c_void_p(mem_dc), ctypes.c_void_p(old))
        finally:
            gdi32.DeleteObject(ctypes.c_void_p(hbmp))
            gdi32.DeleteDC(ctypes.c_void_p(mem_dc))
            user32.ReleaseDC(None, ctypes.c_void_p(screen_dc))
        return ok

    def _bar_xs(self):
        pad = self.H / 2 + 6  # keep the bar strip clear of the round end caps
        span = self.W - 2 * pad
        step = span / (self.BARS - 1)
        return [pad + i * step for i in range(self.BARS)]

    def _assert_topmost(self):
        """Clicking the taskbar raises it above other topmost windows — periodically
        push the pill back to the top of the topmost band so it never hides."""
        if self.win is not None and self._mode is not None:
            try:
                import ctypes
                user32 = ctypes.windll.user32
                user32.GetParent.restype = ctypes.c_void_p
                user32.GetParent.argtypes = [ctypes.c_void_p]
                user32.SetWindowPos.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                                ctypes.c_int, ctypes.c_uint]
                hwnd = user32.GetParent(self.win.winfo_id()) or self.win.winfo_id()
                HWND_TOPMOST = -1
                SWP_NOSIZE, SWP_NOMOVE, SWP_NOACTIVATE = 0x0001, 0x0002, 0x0010
                user32.SetWindowPos(
                    ctypes.c_void_p(hwnd), ctypes.c_void_p(HWND_TOPMOST & 0xFFFFFFFFFFFFFFFF),
                    0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE,
                )
            except Exception:
                pass
        self.root.after(1500, self._assert_topmost)

    def _anchor_xy(self, pos):
        """Window (top-left) position for a named anchor. The window is PAD larger
        than the pill on each side (shadow margin), so offset by -PAD to keep the
        visible pill where the anchor intends."""
        self.win.update_idletasks()
        sw = self.win.winfo_screenwidth()
        sh = self.win.winfo_screenheight()
        m = 28
        top, bot = m + 8, sh - 140
        anchors = {
            'bottom-center': ((sw - self.W) // 2, bot),
            'bottom-right': (sw - self.W - m, bot),
            'bottom-left': (m, bot),
            'top-center': ((sw - self.W) // 2, top),
            'top-right': (sw - self.W - m, top),
            'top-left': (m, top),
        }
        px, py = anchors.get(pos, anchors['bottom-center'])
        return px - self.PAD, py - self.PAD

    def _place(self):
        pos = self.app.cfg.get('bubble_position', 'bottom-center') if self.app else 'bottom-center'
        if pos == 'custom' and self.app:
            x = self.app.cfg.get('bubble_x')
            y = self.app.cfg.get('bubble_y')
            if x is None or y is None:
                x, y = self._anchor_xy('bottom-center')
        else:
            x, y = self._anchor_xy(pos)
        self.win.geometry(f'{self.FW}x{self.FH}+{x}+{y}')
        self.win.update_idletasks()  # ensure the OS window is sized before a layered blit

    def reposition(self):
        """Re-apply placement from config (e.g. after the position setting changes)."""
        if self.win and self.enabled:
            self._place()

    def _stop_anim(self):
        if self._anim_job is not None:
            try:
                self.win.after_cancel(self._anim_job)
            except Exception:
                pass
            self._anim_job = None

    # --- interaction -----------------------------------------------------

    def _on_press(self, ev):
        self.press_root = (ev.x_root, ev.y_root)
        self.win_start = (self.win.winfo_x(), self.win.winfo_y())
        self.dragging = False

    def _on_move(self, ev):
        dx = ev.x_root - self.press_root[0]
        dy = ev.y_root - self.press_root[1]
        if not self.dragging and (abs(dx) + abs(dy) > self.DRAG_THRESHOLD):
            self.dragging = True
        if self.dragging:
            nx = self.win_start[0] + dx
            ny = self.win_start[1] + dy
            self.win.geometry(f'{self.FW}x{self.FH}+{nx}+{ny}')  # full size incl. shadow pad

    def _on_release(self, _ev):
        if not self.dragging:
            return
        self.dragging = False
        if self.app:
            try:
                self.app.cfg['bubble_position'] = 'custom'
                self.app.cfg['bubble_x'] = self.win.winfo_x()
                self.app.cfg['bubble_y'] = self.win.winfo_y()
                save_config(self.app.cfg)
            except Exception as e:
                log(f'failed saving bubble position: {e}')

    def _on_double_click(self, _ev):
        if self.dragging or not self.app:
            return
        self.app.open_settings_from_bubble()

    # --- states -----------------------------------------------------------

    def show_idle(self):
        """Dim row of dots — 'Murmur is running, mic off'. Dots turn amber when
        the cleanup backend is offline (raw transcripts instead of cleaned)."""
        if not self.enabled:
            self.hide_window()
            return
        self._ensure()
        self._stop_anim()
        self._mode = 'idle'
        self._place()
        self.win.deiconify()
        self.win.lift()
        offline = bool(self.app and not getattr(self.app, 'cleanup_online', True))
        dot = '#d98a2b' if offline else '#5a5a5a'

        def content(d, S):
            mid = self.H * S / 2
            r = 1.6 * S
            for x in self._bar_xs():
                cx = x * S
                d.ellipse([cx - r, mid - r, cx + r, mid + r], fill=dot)
        self._render(content, opacity=0.72 if offline else 0.65)

    def appear(self):
        """Show the pill with a soft fade-in (startup / re-enable)."""
        if not self.enabled:
            return
        self.show_idle()
        self._paint(self._last_img, 0.0)
        self._fade(0.0, 1.0)

    def disappear(self, done=None):
        """Fade the pill out, then withdraw it."""
        if self.win is None or self._last_img is None:
            self.hide_window()
            if done:
                done()
            return
        self._stop_anim()
        self._fade(1.0, 0.0, done=lambda: (self.hide_window(), done() if done else None))

    def show_wave(self):
        """Animated waveform while recording — the 'it hears you' signal."""
        if not self.enabled:
            return
        self._ensure()
        self._stop_anim()
        self._mode = 'wave'
        self._levels = deque([0.0] * self.BARS, maxlen=self.BARS)
        self._place()
        self.win.deiconify()
        self.win.lift()
        self._tick()

    def _tick(self):
        if self._mode != 'wave':
            return
        raw = self.app.recorder.level if (self.app and self.app.recorder) else 0.0
        # Speech RMS is roughly 0.02–0.3; scale into 0..1 with a floor so the
        # bar strip stays visible (and clearly "waiting") during silence.
        self._levels.append(min(1.0, raw * 8.0))

        def content(d, S):
            mid = self.H * S / 2
            hw = 1.8 * S  # bar half-width; also the cap radius
            max_h = self.H - 14
            for x, lv in zip(self._bar_xs(), self._levels):
                h = max(4.0, lv * max_h) * S  # min height ≥ bar width so caps fit
                cx = x * S
                d.rounded_rectangle([cx - hw, mid - h / 2, cx + hw, mid + h / 2],
                                    radius=hw, fill='#f2f2f2')
        self._render(content)
        self._anim_job = self.win.after(self.TICK_MS, self._tick)

    def show_processing(self):
        """Three gold dots doing a gentle wave while Whisper + the LLM work."""
        if not self.enabled:
            return
        self._ensure()
        self._stop_anim()
        self._mode = 'proc'
        self._phase = 0
        self._place()
        self.win.deiconify()
        self.win.lift()
        self._tick_processing()

    def _tick_processing(self):
        if self._mode != 'proc':
            return

        def content(d, S):
            cx, mid = self.W * S / 2, self.H * S / 2
            for i in range(3):
                s = (2.8 + 1.8 * max(0.0, math.sin((self._phase - i * 2) * 0.55))) * S
                x = cx + (i - 1) * 15 * S
                d.ellipse([x - s, mid - s, x + s, mid + s], fill='#ffd56b')
        self._render(content)
        self._phase += 1
        self._anim_job = self.win.after(80, self._tick_processing)

    def hide_window(self):
        self._stop_anim()
        self._mode = None
        if self.win:
            self.win.withdraw()


_ICON_PATH = Path(__file__).parent / 'assets' / 'icon.png'
_ICON_BASE = None


def _load_base_icon():
    global _ICON_BASE
    if _ICON_BASE is None and _ICON_PATH.exists():
        _ICON_BASE = Image.open(_ICON_PATH).convert('RGBA').resize((64, 64), Image.LANCZOS)
    return _ICON_BASE


def make_icon_image(color):
    """Branded gold-M base + a small state dot in the bottom-right corner."""
    base = _load_base_icon()
    if base is None:
        # Fallback if icon.png missing — flat circle with letter
        img = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse((4, 4, 60, 60), fill=color, outline='white', width=2)
        return img
    img = base.copy()
    if color and color != '#5a5a5a':  # not idle — paint a state dot
        d = ImageDraw.Draw(img)
        d.ellipse((44, 44, 60, 60), fill=color, outline='black', width=1)
    return img


STATE_VISUAL = {
    'idle': ('#5a5a5a', None, None),
    'listening': ('#e64545', 'Listening...', '#a02030'),
    'processing': ('#f0a030', 'Processing...', '#a06010'),
    'success': ('#4caf50', None, None),
}


class MurmurApp:
    def __init__(self):
        self.cfg = load_config()
        self.stats = load_stats()
        self.recorder = Recorder(self.cfg['sample_rate'], self.cfg['max_record_sec'])
        self.transcriber = Transcriber(self.cfg)
        self.controller = keyboard.Controller()
        self.state = 'idle'
        self.cleanup_online = True  # False when the cleanup backend is unreachable

        self.tk_root = tk.Tk()
        self.tk_root.withdraw()
        self.bubble = Bubble(self.tk_root, self)
        self.tk_root.after(150, self.bubble.appear)  # persistent pill, fades in

        # Two independent chords, each an arbitrary set of key tokens: push-to-talk
        # (hold) and continuous (tap to toggle). _pressed tracks all held keys.
        self.ptt_tokens = [t.lower() for t in (self.cfg.get('hotkey') or ['ctrl', 'cmd'])]
        self.toggle_tokens = [t.lower() for t in (self.cfg.get('toggle_hotkey') or [])]
        self._pressed = set()
        self._ptt_active = False
        self._toggle_was_active = False
        self.continuous = False
        # Custom-hotkey recording
        self._capturing = None
        self._capture_pressed = set()
        self._capture_max = set()
        self._settings_win = None

        self.icon = Icon(
            'murmur',
            make_icon_image(STATE_VISUAL['idle'][0]),
            title='Murmur — Idle',
            menu=Menu(
                MenuItem(
                    'Show Status Bubble',
                    self._menu_toggle_bubble,
                    checked=lambda _i: bool(self.cfg.get('bubble_visible', True)),
                ),
                MenuItem(
                    'Start on Login',
                    self._menu_toggle_startup,
                    checked=lambda _i: startup_enabled(),
                ),
                MenuItem(
                    'Add to Start Menu',
                    self._menu_toggle_start_menu,
                    checked=lambda _i: start_menu_enabled(),
                ),
                MenuItem('Settings', self._menu_settings),
                MenuItem('Edit Dictionary', self._menu_dictionary),
                MenuItem('Edit Commands', self._menu_commands),
                MenuItem('Stats', self._menu_stats),
                Menu.SEPARATOR,
                MenuItem('Quit', self._menu_quit),
            ),
        )

        threading.Thread(target=self.transcriber.load, daemon=True).start()
        threading.Thread(target=self._warm_cleanup, daemon=True).start()

    def _warm_cleanup(self):
        if not self.cfg.get('cleanup_enabled', True):
            return
        if self.cfg.get('cleanup_backend', 'ollama') != 'ollama':
            return
        url = self.cfg.get('ollama_url', 'http://localhost:11434')
        try:
            self._set_cleanup_online(ensure_ollama_running(url))  # launch it if it's down
            t0 = time.time()
            _, ok = clean_with_ollama('hello world', [],
                                      self.cfg.get('cleanup_model', 'qwen2.5:7b'), url)
            self._set_cleanup_online(ok)
            log(f'ollama warmed in {time.time()-t0:.1f}s (online={ok})')
        except Exception as e:
            log(f'ollama warmup failed: {e}')

    def _set_cleanup_online(self, ok):
        """Track backend reachability; refresh the idle pill if the state flipped."""
        ok = bool(ok)
        if ok == self.cleanup_online:
            return
        self.cleanup_online = ok
        if self.state == 'idle':
            self.tk_root.after(0, self.bubble.show_idle)

    def _recover_cleanup(self):
        """Backend went offline mid-use — try to bring Ollama back in the background."""
        url = self.cfg.get('ollama_url', 'http://localhost:11434')
        if self.cfg.get('cleanup_backend', 'ollama') == 'ollama':
            self._set_cleanup_online(ensure_ollama_running(url))

    def _chord_active(self, tokens):
        return bool(tokens) and set(tokens) <= self._pressed

    def start_hotkey_capture(self, on_captured):
        """Record the next key combination. on_captured(tokens|None) fires on the
        tk thread (None = nothing captured / cancelled). Auto-cancels after 8s."""
        if self._capturing is not None:      # a row was already recording — cancel it
            prev, self._capturing = self._capturing, None
            prev(None)
        self._capture_pressed = set()
        self._capture_max = set()

        def cb(combo):
            self.tk_root.after(0, lambda: on_captured(combo))
        self._capturing = cb

        def timeout():
            if self._capturing is cb:
                self._capturing = None
                self.tk_root.after(0, lambda: on_captured(None))
        threading.Timer(8.0, timeout).start()

    def set_state(self, state):
        self.state = state
        color, bubble_text, bubble_bg = STATE_VISUAL[state]
        self.icon.icon = make_icon_image(color)
        self.icon.title = f'Murmur — {state.capitalize()}'
        if state == 'listening':
            self.tk_root.after(0, self.bubble.show_wave)
        elif state == 'processing':
            self.tk_root.after(0, self.bubble.show_processing)
        else:
            self.tk_root.after(0, self.bubble.show_idle)

    def _cue(self, kind):
        if not self.cfg.get('sounds', True):
            return
        play_cue(kind, self.cfg.get('cue_volume', 35) / 100.0,
                 self.cfg.get('cue_sound', 'Soft sine'), self.cfg.get('cue_file', ''))

    def on_press(self, key):
        tok = key_token(key)
        self._pressed.add(tok)

        # Recording a custom hotkey — swallow keys, track the largest chord seen.
        if self._capturing is not None:
            self._capture_pressed.add(tok)
            if len(self._capture_pressed) > len(self._capture_max):
                self._capture_max = set(self._capture_pressed)
            return

        # Continuous mode: fire once on the rising edge of the toggle chord
        if self.toggle_tokens:
            if self._chord_active(self.toggle_tokens):
                if not self._toggle_was_active:
                    self._toggle_was_active = True
                    self.toggle_continuous()
            else:
                self._toggle_was_active = False

        # Push-to-talk: start on chord down (ignored while continuous owns the mic)
        if (not self.continuous and self.state == 'idle'
                and self._chord_active(self.ptt_tokens)
                and not (self.toggle_tokens and self._chord_active(self.toggle_tokens))):
            self._ptt_active = True
            self.set_state('listening')
            self._cue('start')
            try:
                self.recorder.start()
            except Exception as e:
                print(f'[murmur] recorder start failed: {e}', file=sys.stderr)
                self._ptt_active = False
                self.set_state('idle')

    def on_release(self, key):
        tok = key_token(key)

        if self._capturing is not None:
            self._capture_pressed.discard(tok)
            self._pressed.discard(tok)
            if not self._capture_pressed and self._capture_max:
                combo = order_tokens(self._capture_max)
                cb, self._capturing, self._capture_max = self._capturing, None, set()
                cb(None if combo == ['esc'] else combo)  # Esc alone = cancel
            return

        ptt_was_active = self._chord_active(self.ptt_tokens)
        self._pressed.discard(tok)
        if not self._chord_active(self.toggle_tokens):
            self._toggle_was_active = False

        # Push-to-talk: stop + process when the chord is released
        if self._ptt_active and ptt_was_active and not self.continuous:
            self._ptt_active = False
            self._cue('stop')
            audio = self.recorder.stop()
            self.set_state('processing')
            threading.Thread(target=self._process, args=(audio,), daemon=True).start()

    def toggle_continuous(self):
        """Tap the toggle hotkey to start/stop hands-free continuous dictation."""
        if self.continuous:
            log('continuous: stopping')
            self._cue('stop')
            self.continuous = False  # worker flushes trailing audio, stops recorder, goes idle
            return
        if self.state != 'idle' or self._ptt_active:
            return  # busy with push-to-talk
        log('continuous: starting')
        self.continuous = True
        self.set_state('listening')
        self._cue('start')
        try:
            self.recorder.start()
        except Exception as e:
            log(f'continuous recorder start failed: {e}')
            self.continuous = False
            self.set_state('idle')
            return
        threading.Thread(target=self._continuous_worker, daemon=True).start()

    def _continuous_worker(self):
        """Segment the live mic stream on silence; transcribe+clean+paste each
        segment as it completes, so text flows while you keep talking.

        The silence hold is deliberately generous (configurable): short pauses
        mid-sentence should NOT end a segment. Longer segments also punctuate
        better, since cleanup decides sentence breaks from the words, not the
        pauses — so a real sentence end inside a segment still gets a period."""
        sr = int(self.cfg['sample_rate'])
        SPEECH_RMS = 0.015     # above this = speech
        MIN_SEG = 0.4          # ignore blips shorter than this
        MAX_SEG = 30.0         # force-flush a very long run
        # seconds of trailing silence that ends a segment (read live from config)
        silence_hold = float(self.cfg.get('continuous_pause', 1.0))
        idx = 0
        seg, seg_dur, sil_dur, speaking = [], 0.0, 0.0, False
        try:
            while self.continuous:
                time.sleep(0.05)
                silence_hold = float(self.cfg.get('continuous_pause', 1.0))
                frames = self.recorder.frames
                n = len(frames)
                while idx < n:
                    chunk = frames[idx]
                    idx += 1
                    dur = len(chunk) / sr
                    rms = float(np.sqrt(np.mean(chunk ** 2)))
                    if rms >= SPEECH_RMS:
                        speaking = True
                        sil_dur = 0.0
                        seg.append(chunk)
                        seg_dur += dur
                    elif speaking:
                        seg.append(chunk)
                        seg_dur += dur
                        sil_dur += dur
                    if speaking and seg_dur >= MIN_SEG and (sil_dur >= silence_hold or seg_dur >= MAX_SEG):
                        audio = np.concatenate(seg, axis=0).flatten()
                        seg, seg_dur, sil_dur, speaking = [], 0.0, 0.0, False
                        self._process_segment(audio)
            # toggled off — flush whatever is buffered
            if speaking and seg_dur >= MIN_SEG:
                self._process_segment(np.concatenate(seg, axis=0).flatten())
        except Exception as e:
            import traceback
            log(f'continuous worker error: {e}')
            log(traceback.format_exc())
        finally:
            self.recorder.stop()
            self.continuous = False
            self.set_state('idle')

    def _cleanup(self, text, dictionary, lang):
        """Run the configured cleanup backend, tracking reachability and applying
        per-app tone. Returns the cleaned (or raw, if offline) text."""
        backend = self.cfg.get('cleanup_backend', 'ollama')
        model = self.cfg.get('cleanup_model', 'qwen2.5:7b')
        tone = tone_for_foreground() if self.cfg.get('tone_matching', True) else None
        if backend == 'claude':
            out, ok = clean_with_claude(text, dictionary, model, lang, tone)
        else:
            out, ok = clean_with_ollama(text, dictionary, model,
                                        self.cfg.get('ollama_url', 'http://localhost:11434'), lang, tone)
        self._set_cleanup_online(ok)
        if not ok:
            threading.Thread(target=self._recover_cleanup, daemon=True).start()
        return out

    def _try_command(self, raw):
        """If the utterance is an editing command, perform it and return True."""
        if not self.cfg.get('voice_commands', True):
            return False
        action = match_voice_command(raw, load_commands())
        if not action:
            return False
        log(f'voice command: {raw[:40]!r} -> {action}')
        do_voice_command(action, self.controller)
        return True

    def _process_segment(self, audio):
        """Transcribe/clean/paste one continuous-mode segment. Keeps the UI in
        'listening' so the waveform stays live while the user keeps talking."""
        try:
            if audio is None or len(audio) < int(0.3 * self.cfg['sample_rate']):
                return
            dictionary = load_dictionary()
            text, lang = self.transcriber.transcribe(audio, dictionary)
            log(f'continuous segment [{lang}]: {text[:100]!r}')
            if not text:
                return
            if self._try_command(text):
                return
            if self.cfg.get('cleanup_enabled', True):
                text = self._cleanup(text, dictionary, lang)
            if text:
                paste_text(text, self.controller)
                self._update_stats(text)
        except Exception as e:
            log(f'continuous segment error: {e}')

    def _process(self, audio):
        try:
            if audio is None or len(audio) < 1600:
                log(f'audio too short: {0 if audio is None else len(audio)} samples')
                self.set_state('idle')
                return
            log(f'audio captured: {len(audio)} samples ({len(audio)/self.cfg["sample_rate"]:.1f}s)')
            dictionary = load_dictionary()
            log(f'transcribing with dict={len(dictionary)} terms')
            t0 = time.time()
            text, lang = self.transcriber.transcribe(audio, dictionary)
            log(f'transcribed in {time.time()-t0:.1f}s [{lang}]: {text[:120]!r}')
            if not text:
                self.set_state('idle')
                return
            if self._try_command(text):
                self.set_state('success')
                threading.Timer(0.4, lambda: self.set_state('idle')).start()
                return
            if self.cfg.get('cleanup_enabled', True):
                t0 = time.time()
                log(f'cleanup via {self.cfg.get("cleanup_backend", "ollama")}: {self.cfg.get("cleanup_model", "qwen2.5:7b")}')
                text = self._cleanup(text, dictionary, lang)
                log(f'cleaned in {time.time()-t0:.1f}s: {text[:120]!r}')
            log('pasting')
            paste_text(text, self.controller)
            log('paste done')
            self._update_stats(text)
            self.set_state('success')
            threading.Timer(0.4, lambda: self.set_state('idle')).start()
        except Exception as e:
            import traceback
            log(f'process error: {e}')
            log(traceback.format_exc())
            self.set_state('idle')

    def _update_stats(self, text):
        words = len([w for w in text.split() if w.strip()])
        self.stats['total_words'] = self.stats.get('total_words', 0) + words
        self.stats['sessions'] = self.stats.get('sessions', 0) + 1
        self.stats['last_session'] = time.strftime('%Y-%m-%d %H:%M:%S')
        save_stats(self.stats)

    def _menu_toggle_bubble(self, icon, item):
        self.cfg['bubble_visible'] = not self.cfg.get('bubble_visible', True)
        save_config(self.cfg)
        if self.cfg['bubble_visible']:
            self.tk_root.after(0, self.bubble.appear)
        else:
            self.tk_root.after(0, self.bubble.disappear)

    def _menu_toggle_startup(self, icon, item):
        set_startup(not startup_enabled())

    def _menu_toggle_start_menu(self, icon, item):
        set_start_menu(not start_menu_enabled())

    def open_settings_from_bubble(self):
        """Double-click on the bubble toggles Settings open/closed."""
        self.open_settings(toggle=True)

    def _menu_settings(self, icon, item):
        self.open_settings(toggle=False)

    def open_settings(self, toggle=False):
        self.tk_root.after(0, lambda: self._open_settings_ui(toggle))

    def _open_settings_ui(self, toggle):
        from settings_ui import open_settings_window
        w = self._settings_win
        if w is not None and w.winfo_exists():
            if toggle:
                w.destroy()
                self._settings_win = None
            else:
                w.deiconify()
                w.lift()
                w.focus_force()
            return
        self._settings_win = open_settings_window(
            self.tk_root, self.cfg, self._on_settings_saved,
            capture_hotkey=self.start_hotkey_capture)

    def _on_settings_saved(self, new_cfg):
        self.cfg = new_cfg
        save_config(new_cfg)
        self.ptt_tokens = [t.lower() for t in (new_cfg.get('hotkey') or ['ctrl', 'cmd'])]
        self.toggle_tokens = [t.lower() for t in (new_cfg.get('toggle_hotkey') or [])]
        self._toggle_was_active = False
        if not self.continuous:  # don't yank the mic out from under a running session
            self.recorder = Recorder(new_cfg['sample_rate'], new_cfg['max_record_sec'])
        if new_cfg.get('bubble_visible', True):
            self.tk_root.after(0, self.bubble.show_idle)  # (re)show + apply new position
        else:
            self.tk_root.after(0, self.bubble.disappear)
        if new_cfg['model'] != self.transcriber.cfg.get('model'):
            self.transcriber = Transcriber(new_cfg)
            threading.Thread(target=self.transcriber.load, daemon=True).start()
        else:
            self.transcriber.cfg = new_cfg

    def _menu_dictionary(self, icon, item):
        from settings_ui import open_dictionary_window
        self.tk_root.after(0, lambda: open_dictionary_window(self.tk_root, DICT_PATH))

    def _menu_commands(self, icon, item):
        from settings_ui import open_commands_window
        load_commands()  # ensure the file exists before editing
        self.tk_root.after(0, lambda: open_commands_window(self.tk_root, COMMANDS_PATH))

    def _menu_stats(self, icon, item):
        from settings_ui import open_stats_window
        self.tk_root.after(0, lambda: open_stats_window(self.tk_root, self.stats))

    def _menu_quit(self, icon, item):
        try:
            self.icon.stop()
        except Exception:
            pass
        self.tk_root.after(0, self.tk_root.destroy)

    def run(self):
        log(f'Murmur starting — claude_cli={CLAUDE_CLI}')
        log(f'ptt={tokens_label(self.ptt_tokens)}, toggle={tokens_label(self.toggle_tokens)}, '
            f'model={self.cfg["model"]}, device={self.cfg["device"]}')
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        threading.Thread(target=self.icon.run, daemon=True).start()
        self.tk_root.mainloop()


if __name__ == '__main__':
    if not acquire_single_instance():
        log('another instance is already running — exiting')
        sys.exit(0)
    MurmurApp().run()
