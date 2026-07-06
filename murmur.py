"""Murmur — self-hosted dictation tool.

Hold the hotkey, speak, release. Whisper transcribes locally,
a local LLM (Ollama) cleans up filler/punctuation, text pastes into the
focused field. Custom dictionary biases recognition + protects
proper nouns during cleanup.
"""
import json
import math
import os
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
from PIL import Image, ImageDraw, ImageTk
from faster_whisper import WhisperModel
import tkinter as tk

MURMUR_DIR = Path.home() / '.murmur'
MURMUR_DIR.mkdir(exist_ok=True)
CONFIG_PATH = MURMUR_DIR / 'config.json'
DICT_PATH = MURMUR_DIR / 'dictionary.txt'
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


def clean_with_ollama(text, dictionary, model, url, lang='en'):
    """Cleanup via local Ollama daemon — fast, free, offline, no quota impact."""
    if not text.strip():
        return text
    import urllib.request
    import urllib.error
    dict_str = ', '.join(dictionary) if dictionary else 'none'
    prompt = CLEANUP_PROMPTS.get(lang, CLEANUP_PROMPT).format(dictionary=dict_str, text=text)
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
                return text
            return out or text
    except urllib.error.URLError as e:
        log(f'ollama URL error: {e}')
        return text
    except Exception as e:
        log(f'ollama cleanup error: {e}')
        return text


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


def clean_with_claude(text, dictionary, model='haiku', lang='en'):
    """Cleanup via the Claude Code CLI on the user's Max plan.

    Strips API-key env vars before invoking so a stale key cannot route
    the call to the paid API — only the OAuth/Max session is used.
    """
    if not text.strip():
        return text
    if not CLAUDE_CLI:
        print('[murmur] claude.exe not found on PATH or ~/.local/bin — skipping cleanup', file=sys.stderr)
        return text
    dict_str = ', '.join(dictionary) if dictionary else 'none'
    prompt = CLEANUP_PROMPTS.get(lang, CLEANUP_PROMPT).format(dictionary=dict_str, text=text)
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
                return text
            return out
        print(f'[murmur] cleanup non-zero exit {result.returncode}: {result.stderr}', file=sys.stderr)
        return text
    except Exception as e:
        print(f'[murmur] cleanup error: {e}', file=sys.stderr)
        return text


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
        self.dragging = False
        self.press_root = (0, 0)
        self.win_start = (0, 0)

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
            self.win, width=self.W, height=self.H,
            bg=self.KEY, highlightthickness=0, cursor='hand2',
        )
        self.canvas.pack()
        # Drag to move, double-click for settings, right-click to hide
        self.canvas.bind('<ButtonPress-1>', self._on_press)
        self.canvas.bind('<B1-Motion>', self._on_move)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)
        self.canvas.bind('<Double-Button-1>', self._on_double_click)
        self.canvas.bind('<ButtonPress-3>', lambda _e: self._hide_forever())
        self.win.withdraw()
        self._assert_topmost()  # starts the keep-on-top watchdog loop

    SS = 4  # supersample factor — render 4x with PIL, downscale for smooth curves

    def _render(self, draw_content, opacity=0.95):
        """Draw the capsule + content into an RGBA frame (4x supersampled) and
        composite it with true per-pixel alpha. Edges blend against whatever is
        behind the window — actually smooth, unlike color-key transparency."""
        S = self.SS
        img = Image.new('RGBA', (self.W * S, self.H * S), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        m = S  # 1 logical px margin
        d.rounded_rectangle(
            [m, m, self.W * S - m, self.H * S - m],
            radius=(self.H * S - 2 * m) // 2,
            fill=self.PILL_BG, outline=self.PILL_EDGE, width=S,
        )
        draw_content(d, S)
        img = img.resize((self.W, self.H), Image.LANCZOS)
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
        return anchors.get(pos, anchors['bottom-center'])

    def _place(self):
        pos = self.app.cfg.get('bubble_position', 'bottom-center') if self.app else 'bottom-center'
        if pos == 'custom' and self.app:
            x = self.app.cfg.get('bubble_x')
            y = self.app.cfg.get('bubble_y')
            if x is None or y is None:
                x, y = self._anchor_xy('bottom-center')
        else:
            x, y = self._anchor_xy(pos)
        self.win.geometry(f'{self.W}x{self.H}+{x}+{y}')
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
            self.win.geometry(f'{self.W}x{self.H}+{nx}+{ny}')

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
        rect = (self.win.winfo_x(), self.win.winfo_y(),
                self.win.winfo_width(), self.win.winfo_height())
        self.app.open_settings_from_bubble(rect)

    def _hide_forever(self):
        if self.app:
            self.app.cfg['bubble_visible'] = False
            save_config(self.app.cfg)
        self.hide_window()

    # --- states -----------------------------------------------------------

    def show_idle(self):
        """Dim row of dots — 'Murmur is running, mic off'."""
        if not self.enabled:
            self.hide_window()
            return
        self._ensure()
        self._stop_anim()
        self._mode = 'idle'
        self._place()
        self.win.deiconify()
        self.win.lift()

        def content(d, S):
            mid = self.H * S / 2
            r = 1.6 * S
            for x in self._bar_xs():
                cx = x * S
                d.ellipse([cx - r, mid - r, cx + r, mid + r], fill='#5a5a5a')
        self._render(content, opacity=0.65)

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

        self.tk_root = tk.Tk()
        self.tk_root.withdraw()
        self.bubble = Bubble(self.tk_root, self)
        self.tk_root.after(150, self.bubble.show_idle)  # persistent pill (no-op if hidden)

        # Two independent chords: push-to-talk (hold) and continuous (tap to toggle)
        self.ptt_groups = self._parse_hotkey(self.cfg['hotkey'])
        self.ptt_down = {name: False for name in self.ptt_groups}
        self._ptt_active = False
        self.toggle_groups = self._parse_hotkey(self.cfg.get('toggle_hotkey', []), allow_empty=True)
        self.toggle_down = {name: False for name in self.toggle_groups}
        self._toggle_was_active = False
        self.continuous = False

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
                MenuItem('Settings', self._menu_settings),
                MenuItem('Edit Dictionary', self._menu_dictionary),
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
        try:
            t0 = time.time()
            clean_with_ollama(
                'hello world',
                [],
                self.cfg.get('cleanup_model', 'qwen2.5:7b'),
                self.cfg.get('ollama_url', 'http://localhost:11434'),
            )
            log(f'ollama warmed in {time.time()-t0:.1f}s')
        except Exception as e:
            log(f'ollama warmup failed: {e}')

    def _parse_hotkey(self, spec, allow_empty=False):
        if isinstance(spec, str):
            spec = [spec]
        groups = {}
        for name in (spec or []):
            name = name.lower().strip()
            if name in KEY_GROUPS:
                groups[name] = KEY_GROUPS[name]
        if not groups and not allow_empty:
            groups = {'ctrl': CTRL_KEYS, 'cmd': WIN_KEYS}
        return groups

    @staticmethod
    def _key_group(groups, key):
        for name, group in groups.items():
            if key in group:
                return name
        return None

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

    def _ptt_chord_active(self):
        return bool(self.ptt_down) and all(self.ptt_down.values())

    def _toggle_chord_active(self):
        return bool(self.toggle_down) and all(self.toggle_down.values())

    def on_press(self, key):
        pg = self._key_group(self.ptt_groups, key)
        if pg:
            self.ptt_down[pg] = True
        tg = self._key_group(self.toggle_groups, key)
        if tg:
            self.toggle_down[tg] = True

        # Continuous mode: fire once on the rising edge of the toggle chord
        if self.toggle_groups:
            if self._toggle_chord_active():
                if not self._toggle_was_active:
                    self._toggle_was_active = True
                    self.toggle_continuous()
            else:
                self._toggle_was_active = False

        # Push-to-talk: start on chord down (ignored while continuous mode owns the mic)
        if pg and not self.continuous and not self._toggle_chord_active() \
                and self._ptt_chord_active() and self.state == 'idle':
            self._ptt_active = True
            self.set_state('listening')
            try:
                self.recorder.start()
            except Exception as e:
                print(f'[murmur] recorder start failed: {e}', file=sys.stderr)
                self._ptt_active = False
                self.set_state('idle')

    def on_release(self, key):
        pg = self._key_group(self.ptt_groups, key)
        tg = self._key_group(self.toggle_groups, key)
        ptt_was_active = self._ptt_chord_active()
        if pg:
            self.ptt_down[pg] = False
        if tg:
            self.toggle_down[tg] = False
        if not self._toggle_chord_active():
            self._toggle_was_active = False

        # Push-to-talk: stop + process when the chord is released
        if pg and self._ptt_active and ptt_was_active and not self.continuous:
            self._ptt_active = False
            audio = self.recorder.stop()
            self.set_state('processing')
            threading.Thread(target=self._process, args=(audio,), daemon=True).start()

    def toggle_continuous(self):
        """Tap the toggle hotkey to start/stop hands-free continuous dictation."""
        if self.continuous:
            log('continuous: stopping')
            self.continuous = False  # worker flushes trailing audio, stops recorder, goes idle
            return
        if self.state != 'idle' or self._ptt_active:
            return  # busy with push-to-talk
        log('continuous: starting')
        self.continuous = True
        self.set_state('listening')
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
        segment as it completes, so text flows while you keep talking."""
        sr = int(self.cfg['sample_rate'])
        SPEECH_RMS = 0.015     # above this = speech
        SILENCE_HOLD = 0.6     # seconds of trailing silence that ends a segment
        MIN_SEG = 0.4          # ignore blips shorter than this
        MAX_SEG = 20.0         # force-flush a very long run
        idx = 0
        seg, seg_dur, sil_dur, speaking = [], 0.0, 0.0, False
        try:
            while self.continuous:
                time.sleep(0.05)
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
                    if speaking and seg_dur >= MIN_SEG and (sil_dur >= SILENCE_HOLD or seg_dur >= MAX_SEG):
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
            if self.cfg.get('cleanup_enabled', True):
                backend = self.cfg.get('cleanup_backend', 'ollama')
                model = self.cfg.get('cleanup_model', 'qwen2.5:7b')
                if backend == 'claude':
                    text = clean_with_claude(text, dictionary, model, lang)
                else:
                    text = clean_with_ollama(text, dictionary, model, self.cfg.get('ollama_url', 'http://localhost:11434'), lang)
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
            if self.cfg.get('cleanup_enabled', True):
                t0 = time.time()
                backend = self.cfg.get('cleanup_backend', 'ollama')
                model = self.cfg.get('cleanup_model', 'qwen2.5:7b')
                log(f'cleanup via {backend}: {model}')
                if backend == 'claude':
                    text = clean_with_claude(text, dictionary, model, lang)
                else:
                    text = clean_with_ollama(text, dictionary, model, self.cfg.get('ollama_url', 'http://localhost:11434'), lang)
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
            self.tk_root.after(0, self.bubble.show_idle)
        else:
            self.tk_root.after(0, self.bubble.hide_window)

    def open_settings_from_bubble(self, rect):
        """Double-click on the bubble — settings window expands out of it."""
        from settings_ui import open_settings_window
        open_settings_window(self.tk_root, self.cfg, self._on_settings_saved, origin=rect)

    def _menu_settings(self, icon, item):
        from settings_ui import open_settings_window
        self.tk_root.after(0, lambda: open_settings_window(self.tk_root, self.cfg, self._on_settings_saved))

    def _on_settings_saved(self, new_cfg):
        self.cfg = new_cfg
        save_config(new_cfg)
        self.ptt_groups = self._parse_hotkey(new_cfg['hotkey'])
        self.ptt_down = {name: False for name in self.ptt_groups}
        self.toggle_groups = self._parse_hotkey(new_cfg.get('toggle_hotkey', []), allow_empty=True)
        self.toggle_down = {name: False for name in self.toggle_groups}
        self._toggle_was_active = False
        if not self.continuous:  # don't yank the mic out from under a running session
            self.recorder = Recorder(new_cfg['sample_rate'], new_cfg['max_record_sec'])
        if new_cfg.get('bubble_visible', True):
            self.tk_root.after(0, self.bubble.show_idle)  # (re)show + apply new position
        else:
            self.tk_root.after(0, self.bubble.hide_window)
        if new_cfg['model'] != self.transcriber.cfg.get('model'):
            self.transcriber = Transcriber(new_cfg)
            threading.Thread(target=self.transcriber.load, daemon=True).start()
        else:
            self.transcriber.cfg = new_cfg

    def _menu_dictionary(self, icon, item):
        from settings_ui import open_dictionary_window
        self.tk_root.after(0, lambda: open_dictionary_window(self.tk_root, DICT_PATH))

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
        log(f'ptt={list(self.ptt_groups)}, toggle={list(self.toggle_groups)}, '
            f'model={self.cfg["model"]}, device={self.cfg["device"]}')
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        threading.Thread(target=self.icon.run, daemon=True).start()
        self.tk_root.mainloop()


if __name__ == '__main__':
    MurmurApp().run()
