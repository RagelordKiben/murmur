"""Tkinter windows for Murmur settings, dictionary, and stats."""
import tkinter as tk
from tkinter import ttk, messagebox

HOTKEY_PRESETS = {
    'Ctrl + Win  (Wispr default)': ['ctrl', 'cmd'],
    'Ctrl + Alt': ['ctrl', 'alt'],
    'Ctrl + Shift': ['ctrl', 'shift'],
    'Alt + Shift': ['alt', 'shift'],
    'Ctrl + Win + Alt': ['ctrl', 'cmd', 'alt'],
}

# Continuous-mode toggle chord — includes an "Off" choice, and avoids reusing the
# push-to-talk default so the two don't collide.
TOGGLE_PRESETS = {
    'Off (disabled)': [],
    'Ctrl + Alt': ['ctrl', 'alt'],
    'Ctrl + Shift': ['ctrl', 'shift'],
    'Alt + Shift': ['alt', 'shift'],
    'Ctrl + Win + Shift': ['ctrl', 'cmd', 'shift'],
}

# Bubble default start location. 'custom' means "wherever I last dragged it".
POSITION_LABELS = {
    'bottom-center': 'Bottom center',
    'bottom-right': 'Bottom right',
    'bottom-left': 'Bottom left',
    'top-center': 'Top center',
    'top-right': 'Top right',
    'top-left': 'Top left',
    'custom': 'Custom (last dragged)',
}

MODEL_OPTIONS = ['tiny.en', 'base.en', 'small.en', 'medium.en', 'large-v3-turbo', 'large-v3']
DEVICE_OPTIONS = ['cuda', 'cpu']
# Whisper language: ISO code or 'auto' to detect per-utterance.
# Note: '.en' models are English-only — use large-v3-turbo / large-v3 for other languages.
LANGUAGE_OPTIONS = ['en', 'vi', 'auto']
BACKEND_OPTIONS = ['ollama', 'claude']
OLLAMA_MODEL_SUGGESTIONS = [
    'qwen2.5:3b',
    'qwen2.5:7b',
    'llama3.2:3b',
    'llama3.1:8b',
    'gemma2:2b',
]
CLAUDE_MODEL_OPTIONS = ['haiku', 'sonnet', 'opus']


def _match_preset(spec, presets=HOTKEY_PRESETS):
    spec_set = set(spec or [])
    for label, preset in presets.items():
        if set(preset) == spec_set:
            return label
    return list(presets.keys())[0]


def _animate_expand(win, origin, final, steps=18, interval=10):
    """Grow a window from the origin rect to the final rect (ease-out cubic)."""
    ox, oy, ow, oh = origin
    fx, fy, fw, fh = final

    def step(i):
        t = i / steps
        e = 1 - (1 - t) ** 3
        w = max(1, int(ow + (fw - ow) * e))
        h = max(1, int(oh + (fh - oh) * e))
        x = int(ox + (fx - ox) * e)
        y = int(oy + (fy - oy) * e)
        try:
            win.geometry(f'{w}x{h}+{x}+{y}')
        except tk.TclError:
            return  # window was closed mid-animation
        if i < steps:
            win.after(interval, lambda: step(i + 1))

    step(0)


def open_settings_window(root, cfg, on_save, origin=None):
    """origin: optional (x, y, w, h) rect to animate the window out of —
    used when opened by double-clicking the status bubble."""
    WIN_W, WIN_H = 520, 760
    win = tk.Toplevel(root)
    win.title('Murmur —Settings')
    win.attributes('-topmost', True)
    win.configure(bg='#1e1e1e')
    if origin:
        ox, oy, ow, oh = origin
        sw = win.winfo_screenwidth()
        sh = win.winfo_screenheight()
        # Final rect: centered on the bubble, clamped onto the screen
        fx = max(8, min(ox + ow // 2 - WIN_W // 2, sw - WIN_W - 8))
        fy = max(8, min(oy + oh // 2 - WIN_H // 2, sh - WIN_H - 48))
        win.geometry(f'{ow}x{oh}+{ox}+{oy}')
        win.after(10, lambda: _animate_expand(win, (ox, oy, ow, oh), (fx, fy, WIN_W, WIN_H)))
    else:
        win.geometry(f'{WIN_W}x{WIN_H}')

    label_args = {'bg': '#1e1e1e', 'fg': '#ddd', 'font': ('Segoe UI', 10), 'anchor': 'w'}
    head_args = {'bg': '#1e1e1e', 'fg': '#ffd56b', 'font': ('Segoe UI', 10, 'bold'), 'anchor': 'w'}
    pad = {'padx': 14, 'pady': 4}
    r = 0

    tk.Label(win, text='Hotkeys', **head_args).grid(row=r, column=0, columnspan=2, sticky='w', padx=14, pady=(12, 2)); r += 1

    tk.Label(win, text='Push-to-talk (hold to dictate):', **label_args).grid(row=r, column=0, sticky='w', **pad)
    hotkey_var = tk.StringVar(value=_match_preset(cfg.get('hotkey', ['ctrl', 'cmd']), HOTKEY_PRESETS))
    ttk.Combobox(win, textvariable=hotkey_var, values=list(HOTKEY_PRESETS.keys()), state='readonly', width=30).grid(row=r, column=1, **pad); r += 1

    tk.Label(win, text='Continuous transcription (tap to toggle):', **label_args).grid(row=r, column=0, sticky='w', **pad)
    toggle_var = tk.StringVar(value=_match_preset(cfg.get('toggle_hotkey', []), TOGGLE_PRESETS))
    ttk.Combobox(win, textvariable=toggle_var, values=list(TOGGLE_PRESETS.keys()), state='readonly', width=30).grid(row=r, column=1, **pad); r += 1

    tk.Label(win, text='Transcription', **head_args).grid(row=r, column=0, columnspan=2, sticky='w', padx=14, pady=(12, 2)); r += 1

    tk.Label(win, text='Whisper model:', **label_args).grid(row=r, column=0, sticky='w', **pad)
    model_var = tk.StringVar(value=cfg.get('model', 'small.en'))
    ttk.Combobox(win, textvariable=model_var, values=MODEL_OPTIONS, state='readonly', width=30).grid(row=r, column=1, **pad); r += 1

    tk.Label(win, text='Whisper device:', **label_args).grid(row=r, column=0, sticky='w', **pad)
    device_var = tk.StringVar(value=cfg.get('device', 'cuda'))
    ttk.Combobox(win, textvariable=device_var, values=DEVICE_OPTIONS, state='readonly', width=30).grid(row=r, column=1, **pad); r += 1

    tk.Label(win, text='Language:', **label_args).grid(row=r, column=0, sticky='w', **pad)
    lang_var = tk.StringVar(value=cfg.get('language', 'en'))
    ttk.Combobox(win, textvariable=lang_var, values=LANGUAGE_OPTIONS, width=30).grid(row=r, column=1, **pad); r += 1

    cleanup_var = tk.BooleanVar(value=cfg.get('cleanup_enabled', True))
    tk.Checkbutton(
        win,
        text='Enable AI cleanup (filler removal, punctuation)',
        variable=cleanup_var,
        bg='#1e1e1e', fg='#ddd', selectcolor='#1e1e1e',
        activebackground='#1e1e1e', activeforeground='#fff',
        font=('Segoe UI', 10),
    ).grid(row=r, column=0, columnspan=2, sticky='w', **pad); r += 1

    tk.Label(win, text='Cleanup backend:', **label_args).grid(row=r, column=0, sticky='w', **pad)
    backend_var = tk.StringVar(value=cfg.get('cleanup_backend', 'ollama'))
    ttk.Combobox(win, textvariable=backend_var, values=BACKEND_OPTIONS, state='readonly', width=30).grid(row=r, column=1, **pad); r += 1

    tk.Label(win, text='Cleanup model:', **label_args).grid(row=r, column=0, sticky='w', **pad)
    cm_var = tk.StringVar(value=cfg.get('cleanup_model', 'qwen2.5:7b'))
    cm_combo = ttk.Combobox(win, textvariable=cm_var, values=OLLAMA_MODEL_SUGGESTIONS, width=30)
    cm_combo.grid(row=r, column=1, **pad); r += 1

    def on_backend_change(*_):
        if backend_var.get() == 'claude':
            cm_combo['values'] = CLAUDE_MODEL_OPTIONS
            cm_combo['state'] = 'readonly'
            if cm_var.get() not in CLAUDE_MODEL_OPTIONS:
                cm_var.set('haiku')
        else:
            cm_combo['values'] = OLLAMA_MODEL_SUGGESTIONS
            cm_combo['state'] = 'normal'
            if cm_var.get() in CLAUDE_MODEL_OPTIONS:
                cm_var.set('qwen2.5:7b')
    backend_var.trace_add('write', on_backend_change)
    on_backend_change()

    tk.Label(win, text='Behavior', **head_args).grid(row=r, column=0, columnspan=2, sticky='w', padx=14, pady=(12, 2)); r += 1

    def _check(text_, key, default=True):
        nonlocal r
        var = tk.BooleanVar(value=cfg.get(key, default))
        tk.Checkbutton(
            win, text=text_, variable=var,
            bg='#1e1e1e', fg='#ddd', selectcolor='#1e1e1e',
            activebackground='#1e1e1e', activeforeground='#fff', font=('Segoe UI', 10),
        ).grid(row=r, column=0, columnspan=2, sticky='w', **pad); r += 1
        return var

    voice_cmd_var = _check('Voice commands ("new line", "scratch that", "send it")', 'voice_commands')
    tone_var = _check('Match tone to the app (casual in chat, clean in email/docs)', 'tone_matching')
    sounds_var = _check('Play start/stop sounds', 'sounds')

    tk.Label(win, text='Status bubble', **head_args).grid(row=r, column=0, columnspan=2, sticky='w', padx=14, pady=(12, 2)); r += 1

    bubble_visible_var = tk.BooleanVar(value=cfg.get('bubble_visible', True))
    tk.Checkbutton(
        win,
        text='Always show status bubble',
        variable=bubble_visible_var,
        bg='#1e1e1e', fg='#ddd', selectcolor='#1e1e1e',
        activebackground='#1e1e1e', activeforeground='#fff',
        font=('Segoe UI', 10),
    ).grid(row=r, column=0, columnspan=2, sticky='w', **pad); r += 1

    tk.Label(win, text='Default location:', **label_args).grid(row=r, column=0, sticky='w', **pad)
    pos_var = tk.StringVar(value=POSITION_LABELS.get(cfg.get('bubble_position', 'bottom-center'), 'Bottom center'))
    ttk.Combobox(win, textvariable=pos_var, values=list(POSITION_LABELS.values()), state='readonly', width=30).grid(row=r, column=1, **pad); r += 1

    note = tk.Label(
        win,
        text=(
            'ollama  →  fast, free, local, no quota.  Best for high-frequency dictation.\n'
            'claude  →  uses your Max plan window. Higher quality, slower (~7-15s).\n'
            "language: '.en' models are English-only — for vi/auto use large-v3-turbo.\n"
            'Tip: drag the bubble to reposition; double-click it to open settings.'
        ),
        bg='#1e1e1e', fg='#888', font=('Segoe UI', 9, 'italic'),
        justify='left',
    )
    note.grid(row=r, column=0, columnspan=2, sticky='w', padx=14, pady=(10, 4)); r += 1

    label_to_pos = {v: k for k, v in POSITION_LABELS.items()}

    def save():
        new_cfg = dict(cfg)
        new_cfg['hotkey'] = HOTKEY_PRESETS[hotkey_var.get()]
        new_cfg['toggle_hotkey'] = TOGGLE_PRESETS[toggle_var.get()]
        new_cfg['model'] = model_var.get()
        new_cfg['device'] = device_var.get()
        new_cfg['language'] = lang_var.get().strip() or 'en'
        new_cfg['compute_type'] = 'int8_float16' if device_var.get() == 'cuda' else 'int8'
        new_cfg['cleanup_enabled'] = cleanup_var.get()
        new_cfg['cleanup_backend'] = backend_var.get()
        new_cfg['cleanup_model'] = cm_var.get()
        new_cfg['voice_commands'] = voice_cmd_var.get()
        new_cfg['tone_matching'] = tone_var.get()
        new_cfg['sounds'] = sounds_var.get()
        new_cfg['bubble_visible'] = bubble_visible_var.get()
        new_cfg['bubble_position'] = label_to_pos.get(pos_var.get(), 'bottom-center')
        on_save(new_cfg)
        messagebox.showinfo('Murmur', 'Settings saved. Restart Murmur if Whisper model or device changed.')
        win.destroy()

    tk.Button(win, text='Save', command=save, width=14, bg='#3a8a3a', fg='white', font=('Segoe UI', 10, 'bold'), relief='flat', padx=8, pady=4).grid(row=r, column=0, columnspan=2, pady=16)


def open_dictionary_window(root, dict_path):
    win = tk.Toplevel(root)
    win.title('Murmur —Dictionary')
    win.geometry('480x520')
    win.attributes('-topmost', True)
    win.configure(bg='#1e1e1e')

    tk.Label(
        win,
        text='Custom words & phrases (one per line).\nThese bias Whisper recognition and are protected during cleanup.',
        bg='#1e1e1e', fg='#ccc', font=('Segoe UI', 10), justify='left',
    ).pack(padx=14, pady=10, anchor='w')

    text = tk.Text(win, font=('Consolas', 10), wrap='none', bg='#252525', fg='#ddd', insertbackground='white', relief='flat')
    text.pack(fill='both', expand=True, padx=14, pady=4)

    if dict_path.exists():
        text.insert('1.0', dict_path.read_text(encoding='utf-8'))
    else:
        text.insert('1.0', '# One term per line.\n')

    def save():
        dict_path.write_text(text.get('1.0', 'end-1c'), encoding='utf-8')
        messagebox.showinfo('Murmur', 'Dictionary saved.')
        win.destroy()

    tk.Button(win, text='Save', command=save, width=14, bg='#3a8a3a', fg='white', font=('Segoe UI', 10, 'bold'), relief='flat', padx=8, pady=4).pack(pady=12)


def open_stats_window(root, stats):
    win = tk.Toplevel(root)
    win.title('Murmur —Stats')
    win.geometry('340x200')
    win.attributes('-topmost', True)
    win.configure(bg='#1e1e1e')

    rows = [
        ('Total words dictated', f"{stats.get('total_words', 0):,}"),
        ('Sessions', f"{stats.get('sessions', 0):,}"),
        ('Last session', stats.get('last_session') or '—'),
    ]
    for r, (label, val) in enumerate(rows):
        tk.Label(win, text=label + ':', bg='#1e1e1e', fg='#aaa', font=('Segoe UI', 10)).grid(row=r, column=0, sticky='w', padx=14, pady=10)
        tk.Label(win, text=str(val), bg='#1e1e1e', fg='#ddd', font=('Segoe UI', 11, 'bold')).grid(row=r, column=1, sticky='w', padx=8, pady=10)
