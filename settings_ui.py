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


def _match_preset(spec):
    spec_set = set(spec)
    for label, preset in HOTKEY_PRESETS.items():
        if set(preset) == spec_set:
            return label
    return list(HOTKEY_PRESETS.keys())[0]


def open_settings_window(root, cfg, on_save):
    win = tk.Toplevel(root)
    win.title('Murmur —Settings')
    win.geometry('480x440')
    win.attributes('-topmost', True)
    win.configure(bg='#1e1e1e')

    label_args = {'bg': '#1e1e1e', 'fg': '#ddd', 'font': ('Segoe UI', 10), 'anchor': 'w'}
    pad = {'padx': 14, 'pady': 5}

    tk.Label(win, text='Hotkey (hold to dictate):', **label_args).grid(row=0, column=0, sticky='w', **pad)
    hotkey_var = tk.StringVar(value=_match_preset(cfg.get('hotkey', ['ctrl', 'cmd'])))
    ttk.Combobox(win, textvariable=hotkey_var, values=list(HOTKEY_PRESETS.keys()), state='readonly', width=30).grid(row=0, column=1, **pad)

    tk.Label(win, text='Whisper model:', **label_args).grid(row=1, column=0, sticky='w', **pad)
    model_var = tk.StringVar(value=cfg.get('model', 'small.en'))
    ttk.Combobox(win, textvariable=model_var, values=MODEL_OPTIONS, state='readonly', width=30).grid(row=1, column=1, **pad)

    tk.Label(win, text='Whisper device:', **label_args).grid(row=2, column=0, sticky='w', **pad)
    device_var = tk.StringVar(value=cfg.get('device', 'cuda'))
    ttk.Combobox(win, textvariable=device_var, values=DEVICE_OPTIONS, state='readonly', width=30).grid(row=2, column=1, **pad)

    tk.Label(win, text='Language:', **label_args).grid(row=3, column=0, sticky='w', **pad)
    lang_var = tk.StringVar(value=cfg.get('language', 'en'))
    ttk.Combobox(win, textvariable=lang_var, values=LANGUAGE_OPTIONS, width=30).grid(row=3, column=1, **pad)

    cleanup_var = tk.BooleanVar(value=cfg.get('cleanup_enabled', True))
    tk.Checkbutton(
        win,
        text='Enable AI cleanup (filler removal, punctuation)',
        variable=cleanup_var,
        bg='#1e1e1e', fg='#ddd', selectcolor='#1e1e1e',
        activebackground='#1e1e1e', activeforeground='#fff',
        font=('Segoe UI', 10),
    ).grid(row=4, column=0, columnspan=2, sticky='w', **pad)

    tk.Label(win, text='Cleanup backend:', **label_args).grid(row=5, column=0, sticky='w', **pad)
    backend_var = tk.StringVar(value=cfg.get('cleanup_backend', 'ollama'))
    ttk.Combobox(win, textvariable=backend_var, values=BACKEND_OPTIONS, state='readonly', width=30).grid(row=5, column=1, **pad)

    tk.Label(win, text='Cleanup model:', **label_args).grid(row=6, column=0, sticky='w', **pad)
    cm_var = tk.StringVar(value=cfg.get('cleanup_model', 'qwen2.5:7b'))
    cm_combo = ttk.Combobox(win, textvariable=cm_var, values=OLLAMA_MODEL_SUGGESTIONS, width=30)
    cm_combo.grid(row=6, column=1, **pad)

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

    note = tk.Label(
        win,
        text=(
            'ollama  →  fast, free, local, no quota.  Best for high-frequency dictation.\n'
            'claude  →  uses your Max plan window. Higher quality, slower (~7-15s).\n'
            "language: '.en' Whisper models are English-only — for vi/auto use large-v3-turbo."
        ),
        bg='#1e1e1e', fg='#888', font=('Segoe UI', 9, 'italic'),
        justify='left',
    )
    note.grid(row=7, column=0, columnspan=2, sticky='w', padx=14, pady=(12, 4))

    def save():
        new_cfg = dict(cfg)
        new_cfg['hotkey'] = HOTKEY_PRESETS[hotkey_var.get()]
        new_cfg['model'] = model_var.get()
        new_cfg['device'] = device_var.get()
        new_cfg['language'] = lang_var.get().strip() or 'en'
        new_cfg['compute_type'] = 'int8_float16' if device_var.get() == 'cuda' else 'int8'
        new_cfg['cleanup_enabled'] = cleanup_var.get()
        new_cfg['cleanup_backend'] = backend_var.get()
        new_cfg['cleanup_model'] = cm_var.get()
        on_save(new_cfg)
        messagebox.showinfo('Murmur', 'Settings saved. Restart Murmur if Whisper model or device changed.')
        win.destroy()

    tk.Button(win, text='Save', command=save, width=14, bg='#3a8a3a', fg='white', font=('Segoe UI', 10, 'bold'), relief='flat', padx=8, pady=4).grid(row=10, column=0, columnspan=2, pady=18)


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
