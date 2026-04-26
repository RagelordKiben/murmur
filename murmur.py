"""Murmur — self-hosted dictation tool.

Hold the hotkey, speak, release. Whisper transcribes locally,
a local LLM (Ollama) cleans up filler/punctuation, text pastes into the
focused field. Custom dictionary biases recognition + protects
proper nouns during cleanup.
"""
import json
import os
import shutil
import sys
import time
import threading
import subprocess
from pathlib import Path

# Add Ollama's bundled CUDA DLLs to the search path so faster-whisper can use the GPU
# even when CUDA toolkit isn't separately installed. Must happen BEFORE faster_whisper
# is imported, and must update PATH (not just add_dll_directory) for ctranslate2's loader.
if sys.platform == 'win32':
    _cuda_dirs = [
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
from PIL import Image, ImageDraw
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
    'hotkey': ['ctrl', 'cmd'],
    'model': 'medium.en',
    'compute_type': 'int8_float16',
    'device': 'cuda',
    'cleanup_enabled': True,
    'cleanup_backend': 'ollama',     # 'ollama' (fast/free) or 'claude' (heavier)
    'cleanup_model': 'qwen2.5:7b',   # ollama model tag, or claude alias if backend=claude
    'ollama_url': 'http://localhost:11434',
    'sample_rate': 16000,
    'max_record_sec': 90,
    'floating_button': True,
    'floating_x': None,   # remembered position (None = default bottom-right)
    'floating_y': None,
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

    def _callback(self, indata, frames, time_info, status):
        if self.recording:
            self.frames.append(indata.copy())

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
    """Sends audio to the shared Wyoming faster-whisper server (managed by Jarvis tray)
    so we don't load a duplicate Whisper model into VRAM. Falls back to a local model
    only if cfg['fallback_local'] is True AND the Wyoming server is unreachable."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self._lock = threading.Lock()
        self.wyoming_host = cfg.get('wyoming_host', '127.0.0.1')
        self.wyoming_port = int(cfg.get('wyoming_port', 10301))

    def load(self):
        # Wyoming path is connection-based, no preload. Local fallback loads on demand.
        pass

    def _wyoming_transcribe(self, audio_np):
        import asyncio
        from wyoming.asr import Transcribe, Transcript
        from wyoming.audio import AudioChunk, AudioStart, AudioStop
        from wyoming.client import AsyncTcpClient

        audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        rate = int(self.cfg.get('sample_rate', 16000))

        async def _run():
            async with AsyncTcpClient(self.wyoming_host, self.wyoming_port) as client:
                await client.write_event(Transcribe(language='en').event())
                await client.write_event(AudioStart(rate=rate, width=2, channels=1).event())
                # 1-second chunks
                chunk_bytes = rate * 2  # 16-bit mono
                for i in range(0, len(audio_int16), chunk_bytes):
                    payload = audio_int16[i:i + chunk_bytes]
                    await client.write_event(
                        AudioChunk(rate=rate, width=2, channels=1, audio=payload).event()
                    )
                await client.write_event(AudioStop().event())
                while True:
                    event = await asyncio.wait_for(client.read_event(), timeout=30.0)
                    if event is None:
                        return ''
                    if Transcript.is_type(event.type):
                        return Transcript.from_event(event).text
        return asyncio.run(_run())

    def _local_load(self):
        with self._lock:
            if self.model is not None:
                return
            requested_device = self.cfg.get('device', 'cpu')
            if requested_device == 'cuda':
                try:
                    model = WhisperModel(
                        self.cfg['model'],
                        device='cuda',
                        compute_type=self.cfg.get('compute_type', 'int8_float16'),
                    )
                    dummy = np.zeros(16000, dtype=np.float32)
                    segs, _ = model.transcribe(dummy, language='en', vad_filter=False, beam_size=1)
                    list(segs)
                    self.model = model
                    log('whisper fallback loaded on GPU')
                    return
                except Exception as e:
                    log(f'GPU fallback load failed ({e}); using CPU/int8')
            self.model = WhisperModel(self.cfg['model'], device='cpu', compute_type='int8')
            log('whisper fallback loaded on CPU')

    def transcribe(self, audio, dictionary):
        # Primary path: shared Wyoming server (no local VRAM use)
        try:
            text = self._wyoming_transcribe(audio).strip()
            return text
        except Exception as e:
            log(f'Wyoming transcribe failed: {e}')
            if not self.cfg.get('fallback_local', False):
                return ''
        # Fallback path: in-process Whisper (only if explicitly enabled)
        self._local_load()
        prompt = ', '.join(dictionary) if dictionary else None
        segments, _ = self.model.transcribe(
            audio,
            language='en',
            initial_prompt=prompt,
            vad_filter=True,
        )
        return ' '.join(s.text.strip() for s in segments).strip()


CLEANUP_PROMPT = """You clean up dictated text. Follow the rules and study the examples.

RULES:
- Remove meaningless filler words (um, uh, like, you know) — keep them only when they carry meaning.
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

NOW CLEAN THIS DICTATION:
INPUT: {text}
CLEANED:"""


def clean_with_ollama(text, dictionary, model, url):
    """Cleanup via local Ollama daemon — fast, free, offline, no quota impact."""
    if not text.strip():
        return text
    import urllib.request
    import urllib.error
    dict_str = ', '.join(dictionary) if dictionary else 'none'
    prompt = CLEANUP_PROMPT.format(dictionary=dict_str, text=text)
    payload = json.dumps({
        'model': model,
        'prompt': prompt,
        'stream': False,
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


def clean_with_claude(text, dictionary, model='haiku'):
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
    prompt = CLEANUP_PROMPT.format(dictionary=dict_str, text=text)
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


class FloatingMic:
    """Always-on-top click-to-talk mic button. Press = start recording, release = stop.
    Mirrors the keyboard chord exactly so audio path is shared."""

    SIZE = 56  # pixels

    DRAG_THRESHOLD = 6  # pixels — moves above this trigger drag instead of click

    def __init__(self, root, app):
        self.root = root
        self.app = app
        self.win = None
        self.canvas = None
        self.dragging = False
        self.press_root = (0, 0)
        self.win_start = (0, 0)
        self.last_state_color = None

    def _ensure(self):
        if self.win is not None:
            return
        self.win = tk.Toplevel(self.root)
        self.win.overrideredirect(True)
        self.win.attributes('-topmost', True)
        self.win.attributes('-alpha', 0.92)
        self.win.configure(bg='#1a1a1a')
        # Don't steal foreground focus when clicked — paste needs to go to the
        # window the user was typing in, not to this mic button.
        self._apply_no_activate()
        self.canvas = tk.Canvas(
            self.win, width=self.SIZE, height=self.SIZE,
            bg='#1a1a1a', highlightthickness=0, cursor='hand2',
        )
        self.canvas.pack()
        self._draw('#5a5a5a')
        # Left-click: tap to toggle record/stop, hold-and-drag to move.
        # Right-click: hide.
        self.canvas.bind('<ButtonPress-1>', self._on_press)
        self.canvas.bind('<B1-Motion>', self._on_move)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)
        self.canvas.bind('<ButtonPress-3>', lambda _e: self.hide())
        # Restore last position or default to bottom-right
        x = self.app.cfg.get('floating_x')
        y = self.app.cfg.get('floating_y')
        if x is None or y is None:
            sw = self.win.winfo_screenwidth()
            sh = self.win.winfo_screenheight()
            x = sw - self.SIZE - 28
            y = sh - self.SIZE - 90
        self.win.geometry(f'{self.SIZE}x{self.SIZE}+{x}+{y}')

    def _apply_no_activate(self):
        if sys.platform != 'win32':
            return
        try:
            import ctypes
            GWL_EXSTYLE = -20
            WS_EX_NOACTIVATE = 0x08000000
            WS_EX_TOOLWINDOW = 0x00000080
            self.win.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(self.win.winfo_id()) or self.win.winfo_id()
            ex = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(
                hwnd, GWL_EXSTYLE, ex | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW
            )
        except Exception as e:
            log(f'no-activate flag failed: {e}')

    def _draw(self, color):
        if not self.canvas:
            return
        self.last_state_color = color
        self.canvas.delete('all')
        s = self.SIZE
        # Outer circle
        self.canvas.create_oval(2, 2, s - 2, s - 2, fill=color, outline='#ffd56b', width=2)
        # Mic glyph (rounded body + stand)
        cx = s // 2
        body_top = int(s * 0.28)
        body_bot = int(s * 0.62)
        body_w = int(s * 0.18)
        self.canvas.create_oval(cx - body_w, body_top, cx + body_w, body_bot, fill='white', outline='')
        # Arc under mic
        self.canvas.create_arc(
            int(s * 0.26), int(s * 0.42), int(s * 0.74), int(s * 0.78),
            start=200, extent=140, style='arc', outline='white', width=2,
        )
        # Stand
        self.canvas.create_line(cx, int(s * 0.72), cx, int(s * 0.82), fill='white', width=2)
        self.canvas.create_line(int(s * 0.42), int(s * 0.82), int(s * 0.58), int(s * 0.82), fill='white', width=2)

    def show(self):
        self._ensure()
        self.win.deiconify()
        self.win.lift()

    def hide(self):
        if self.win:
            self.win.withdraw()

    def set_state_color(self, color):
        if self.win and self.canvas:
            self._draw(color)

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
            self.win.geometry(f'{self.SIZE}x{self.SIZE}+{nx}+{ny}')

    def _on_release(self, _ev):
        if self.dragging:
            # Was a drag — persist new position, no recording
            try:
                self.app.cfg['floating_x'] = self.win.winfo_x()
                self.app.cfg['floating_y'] = self.win.winfo_y()
                save_config(self.app.cfg)
            except Exception as e:
                log(f'failed saving floating position: {e}')
            self.dragging = False
            return
        # Was a click — toggle record/stop
        if self.app.state == 'idle':
            self.app.set_state('listening')
            try:
                self.app.recorder.start()
            except Exception as e:
                print(f'[murmur] recorder start failed: {e}', file=sys.stderr)
                self.app.set_state('idle')
        elif self.app.state == 'listening':
            audio = self.app.recorder.stop()
            self.app.set_state('processing')
            threading.Thread(target=self.app._process, args=(audio,), daemon=True).start()


class Bubble:
    def __init__(self, root):
        self.root = root
        self.win = None
        self.label = None

    def _ensure(self):
        if self.win is not None:
            return
        self.win = tk.Toplevel(self.root)
        self.win.overrideredirect(True)
        self.win.attributes('-topmost', True)
        self.win.attributes('-alpha', 0.92)
        self.label = tk.Label(
            self.win,
            text='',
            font=('Segoe UI', 11, 'bold'),
            fg='white',
            bg='#202020',
            padx=16,
            pady=8,
        )
        self.label.pack()
        self.win.withdraw()

    def show(self, text, color='#202020'):
        self._ensure()
        self.label.config(text=text, bg=color)
        self.win.configure(bg=color)
        self.win.update_idletasks()
        sw = self.win.winfo_screenwidth()
        sh = self.win.winfo_screenheight()
        w = self.win.winfo_width()
        self.win.geometry(f'+{(sw - w) // 2}+{sh - 140}')
        self.win.deiconify()
        self.win.lift()

    def hide(self):
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
        self.bubble = Bubble(self.tk_root)
        self.floating_mic = FloatingMic(self.tk_root, self) if self.cfg.get('floating_button', True) else None
        if self.floating_mic:
            self.tk_root.after(100, self.floating_mic.show)

        self.hotkey_groups = self._parse_hotkey(self.cfg['hotkey'])
        self.keys_down = {name: False for name in self.hotkey_groups}

        self.icon = Icon(
            'murmur',
            make_icon_image(STATE_VISUAL['idle'][0]),
            title='Murmur — Idle',
            menu=Menu(
                MenuItem(
                    'Show Floating Button',
                    self._menu_toggle_floating,
                    checked=lambda _i: self._floating_visible(),
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

    def _parse_hotkey(self, spec):
        if isinstance(spec, str):
            spec = [spec]
        groups = {}
        for name in spec:
            name = name.lower().strip()
            if name in KEY_GROUPS:
                groups[name] = KEY_GROUPS[name]
        if not groups:
            groups = {'ctrl': CTRL_KEYS, 'cmd': WIN_KEYS}
        return groups

    def _chord_active(self):
        return all(self.keys_down.values())

    def _which_group(self, key):
        for name, group in self.hotkey_groups.items():
            if key in group:
                return name
        return None

    def set_state(self, state):
        self.state = state
        color, bubble_text, bubble_bg = STATE_VISUAL[state]
        self.icon.icon = make_icon_image(color)
        self.icon.title = f'Murmur — {state.capitalize()}'
        if bubble_text:
            self.tk_root.after(0, lambda: self.bubble.show(bubble_text, bubble_bg))
        else:
            self.tk_root.after(0, self.bubble.hide)
        if self.floating_mic:
            self.tk_root.after(0, lambda c=color: self.floating_mic.set_state_color(c))

    def on_press(self, key):
        group = self._which_group(key)
        if group is None:
            return
        self.keys_down[group] = True
        if self._chord_active() and self.state == 'idle':
            self.set_state('listening')
            try:
                self.recorder.start()
            except Exception as e:
                print(f'[murmur] recorder start failed: {e}', file=sys.stderr)
                self.set_state('idle')

    def on_release(self, key):
        group = self._which_group(key)
        if group is None:
            return
        was_active = self._chord_active()
        self.keys_down[group] = False
        if was_active and self.state == 'listening':
            audio = self.recorder.stop()
            self.set_state('processing')
            threading.Thread(target=self._process, args=(audio,), daemon=True).start()

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
            text = self.transcriber.transcribe(audio, dictionary)
            log(f'transcribed in {time.time()-t0:.1f}s: {text[:120]!r}')
            if not text:
                self.set_state('idle')
                return
            if self.cfg.get('cleanup_enabled', True):
                t0 = time.time()
                backend = self.cfg.get('cleanup_backend', 'ollama')
                model = self.cfg.get('cleanup_model', 'qwen2.5:7b')
                log(f'cleanup via {backend}: {model}')
                if backend == 'claude':
                    text = clean_with_claude(text, dictionary, model)
                else:
                    text = clean_with_ollama(text, dictionary, model, self.cfg.get('ollama_url', 'http://localhost:11434'))
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

    def _floating_visible(self):
        return bool(self.floating_mic and self.floating_mic.win and self.floating_mic.win.winfo_viewable())

    def _menu_toggle_floating(self, icon, item):
        # Lazily create the floating mic if it was disabled at startup
        if self.floating_mic is None:
            self.floating_mic = FloatingMic(self.tk_root, self)
        if self._floating_visible():
            self.tk_root.after(0, self.floating_mic.hide)
            self.cfg['floating_button'] = False
        else:
            self.tk_root.after(0, self.floating_mic.show)
            self.cfg['floating_button'] = True
        save_config(self.cfg)

    def _menu_settings(self, icon, item):
        from settings_ui import open_settings_window
        self.tk_root.after(0, lambda: open_settings_window(self.tk_root, self.cfg, self._on_settings_saved))

    def _on_settings_saved(self, new_cfg):
        self.cfg = new_cfg
        save_config(new_cfg)
        self.hotkey_groups = self._parse_hotkey(new_cfg['hotkey'])
        self.keys_down = {name: False for name in self.hotkey_groups}
        self.recorder = Recorder(new_cfg['sample_rate'], new_cfg['max_record_sec'])
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
        log(f'hotkey={list(self.hotkey_groups)}, model={self.cfg["model"]}, device={self.cfg["device"]}')
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        threading.Thread(target=self.icon.run, daemon=True).start()
        self.tk_root.mainloop()


if __name__ == '__main__':
    MurmurApp().run()
