"""Tkinter windows for Murmur settings, dictionary, commands, and stats."""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

MODEL_OPTIONS = ['tiny.en', 'base.en', 'small.en', 'medium.en', 'large-v3-turbo', 'large-v3']
DEVICE_OPTIONS = ['cuda', 'cpu']
# Whisper language: ISO code or 'auto' to detect per-utterance.
# Note: '.en' models are English-only — use large-v3-turbo / large-v3 for other languages.
LANGUAGE_OPTIONS = ['en', 'vi', 'auto']
BACKEND_OPTIONS = ['ollama', 'claude']
OLLAMA_MODEL_SUGGESTIONS = [
    'qwen2.5:3b', 'qwen2.5:7b', 'llama3.2:3b', 'llama3.1:8b', 'gemma2:2b',
]
CLAUDE_MODEL_OPTIONS = ['haiku', 'sonnet', 'opus']

DARK = '#1e1e1e'


def open_settings_window(root, cfg, on_save, capture_hotkey=None):
    """capture_hotkey(on_captured): start recording a custom key combination;
    on_captured(tokens|None) is called when done. Returns the created window."""
    from murmur import tokens_label
    WIN_W, WIN_H = 520, 840
    win = tk.Toplevel(root)
    win.title('Murmur — Settings')
    win.attributes('-topmost', True)
    win.configure(bg=DARK)
    win.geometry(f'{WIN_W}x{WIN_H}')

    label_args = {'bg': DARK, 'fg': '#ddd', 'font': ('Segoe UI', 10), 'anchor': 'w'}
    head_args = {'bg': DARK, 'fg': '#ffd56b', 'font': ('Segoe UI', 10, 'bold'), 'anchor': 'w'}
    pad = {'padx': 14, 'pady': 4}
    r = 0

    def header(text_):
        nonlocal r
        tk.Label(win, text=text_, **head_args).grid(row=r, column=0, columnspan=2, sticky='w', padx=14, pady=(12, 2)); r += 1

    def check(text_, key, default=True):
        nonlocal r
        var = tk.BooleanVar(value=cfg.get(key, default))
        tk.Checkbutton(
            win, text=text_, variable=var,
            bg=DARK, fg='#ddd', selectcolor=DARK,
            activebackground=DARK, activeforeground='#fff', font=('Segoe UI', 10),
        ).grid(row=r, column=0, columnspan=2, sticky='w', **pad); r += 1
        return var

    def hotkey_row(text_, initial, allow_off):
        """A "record any key combination" row. Returns {'tokens': [...]}."""
        nonlocal r
        state = {'tokens': list(initial or [])}
        tk.Label(win, text=text_, **label_args).grid(row=r, column=0, sticky='w', **pad)
        frame = tk.Frame(win, bg=DARK)
        frame.grid(row=r, column=1, sticky='w', padx=14, pady=4); r += 1
        lbl = tk.Label(frame, text=tokens_label(state['tokens']), bg='#252525', fg='#8fb7dc',
                       font=('Segoe UI', 10), width=16, anchor='w', padx=6, pady=2)
        lbl.pack(side='left', padx=(0, 6))

        def record():
            if not capture_hotkey:
                return
            btn.config(text='Press keys…', state='disabled')
            lbl.config(text='…')

            def done(tokens):
                if not btn.winfo_exists():
                    return
                btn.config(text='Record', state='normal')
                if tokens:
                    state['tokens'] = tokens
                lbl.config(text=tokens_label(state['tokens']))
            capture_hotkey(done)
        btn = tk.Button(frame, text='Record', command=record, bg='#333', fg='#ddd',
                        font=('Segoe UI', 9), relief='flat', padx=10, pady=2)
        btn.pack(side='left')

        if allow_off:
            def off():
                state['tokens'] = []
                lbl.config(text=tokens_label([]))
            tk.Button(frame, text='Off', command=off, bg='#333', fg='#ddd',
                      font=('Segoe UI', 9), relief='flat', padx=8, pady=2).pack(side='left', padx=(6, 0))
        return state

    header('Hotkeys')
    ptt_state = hotkey_row('Push-to-talk (hold to dictate):', cfg.get('hotkey') or ['ctrl', 'cmd'], allow_off=False)
    toggle_state = hotkey_row('Continuous transcription (tap to toggle):', cfg.get('toggle_hotkey') or [], allow_off=True)

    header('Transcription')
    tk.Label(win, text='Whisper model:', **label_args).grid(row=r, column=0, sticky='w', **pad)
    model_var = tk.StringVar(value=cfg.get('model', 'small.en'))
    ttk.Combobox(win, textvariable=model_var, values=MODEL_OPTIONS, state='readonly', width=30).grid(row=r, column=1, **pad); r += 1

    tk.Label(win, text='Whisper device:', **label_args).grid(row=r, column=0, sticky='w', **pad)
    device_var = tk.StringVar(value=cfg.get('device', 'cuda'))
    ttk.Combobox(win, textvariable=device_var, values=DEVICE_OPTIONS, state='readonly', width=30).grid(row=r, column=1, **pad); r += 1

    tk.Label(win, text='Language:', **label_args).grid(row=r, column=0, sticky='w', **pad)
    lang_var = tk.StringVar(value=cfg.get('language', 'en'))
    ttk.Combobox(win, textvariable=lang_var, values=LANGUAGE_OPTIONS, width=30).grid(row=r, column=1, **pad); r += 1

    cleanup_var = check('Enable AI cleanup (filler removal, punctuation)', 'cleanup_enabled')

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

    header('Behavior')
    voice_cmd_var = check('Voice commands (say a phrase on its own to run it)', 'voice_commands')

    def edit_commands():
        try:
            from murmur import COMMANDS_PATH, load_commands
            load_commands()  # ensure the file exists
            open_commands_window(root, COMMANDS_PATH)
        except Exception as e:
            messagebox.showerror('Murmur', f'Could not open commands: {e}')
    tk.Button(win, text='Edit Commands…', command=edit_commands, bg='#333', fg='#ddd',
              font=('Segoe UI', 9), relief='flat', padx=8, pady=2).grid(
        row=r, column=0, sticky='w', padx=(34, 14), pady=(0, 6)); r += 1

    # Continuous mode: how long a pause ends a phrase. Higher = pauses within a
    # sentence won't split it (but text appears a bit later).
    tk.Label(win, text='Continuous mode — pause before ending a phrase:', **label_args).grid(
        row=r, column=0, sticky='w', **pad)
    pause_var = tk.DoubleVar(value=float(cfg.get('continuous_pause', 1.0)))
    tk.Scale(win, from_=0.4, to=2.5, resolution=0.1, orient='horizontal', variable=pause_var,
             bg=DARK, fg='#ddd', troughcolor='#333', highlightthickness=0,
             length=200, showvalue=True).grid(row=r, column=1, sticky='w', **pad); r += 1

    tone_var = check('Match tone to the app (casual in chat, clean in email/docs)', 'tone_matching')
    sounds_var = check('Play start/stop sounds', 'sounds')

    # Blip sound: built-in presets + custom WAV, with a Test button and volume.
    try:
        from murmur import CUE_PRESETS, CUSTOM_LABEL, play_cue
        sound_values = list(CUE_PRESETS.keys()) + [CUSTOM_LABEL]
    except Exception:
        CUSTOM_LABEL = 'Custom file…'
        sound_values = ['Soft sine', CUSTOM_LABEL]
        play_cue = None

    tk.Label(win, text='Blip sound:', **label_args).grid(row=r, column=0, sticky='w', **pad)
    sound_var = tk.StringVar(value=cfg.get('cue_sound', 'Soft sine'))
    ttk.Combobox(win, textvariable=sound_var, values=sound_values, state='readonly', width=30).grid(row=r, column=1, **pad); r += 1

    cue_file_var = tk.StringVar(value=cfg.get('cue_file', ''))
    file_lbl = tk.Label(win, text='', bg=DARK, fg='#8fb7dc', font=('Segoe UI', 8), anchor='w')

    def refresh_file_lbl(*_):
        p = cue_file_var.get()
        if sound_var.get() == CUSTOM_LABEL and p:
            file_lbl.config(text='  ' + p.replace('\\', '/').split('/')[-1])
        else:
            file_lbl.config(text='')
    sound_var.trace_add('write', refresh_file_lbl)

    btns = tk.Frame(win, bg=DARK)
    btns.grid(row=r, column=1, sticky='w', padx=14, pady=(0, 2)); r += 1

    def browse():
        p = filedialog.askopenfilename(title='Choose a WAV file',
                                       filetypes=[('WAV audio', '*.wav')])
        if p:
            cue_file_var.set(p)
            sound_var.set(CUSTOM_LABEL)
            refresh_file_lbl()

    def test_sound():
        if play_cue:
            play_cue('start', vol_var.get() / 100.0, sound_var.get(), cue_file_var.get())
    tk.Button(btns, text='Choose WAV…', command=browse, bg='#333', fg='#ddd',
              font=('Segoe UI', 9), relief='flat', padx=8, pady=2).pack(side='left', padx=(0, 6))
    tk.Button(btns, text='▶ Test', command=test_sound, bg='#333', fg='#ddd',
              font=('Segoe UI', 9), relief='flat', padx=8, pady=2).pack(side='left')
    file_lbl.grid(row=r, column=0, columnspan=2, sticky='w', padx=(34, 14)); r += 1
    refresh_file_lbl()

    tk.Label(win, text='Blip volume:', **label_args).grid(row=r, column=0, sticky='w', **pad)
    vol_var = tk.IntVar(value=int(cfg.get('cue_volume', 35)))
    tk.Scale(win, from_=0, to=100, orient='horizontal', variable=vol_var,
             bg=DARK, fg='#ddd', troughcolor='#333', highlightthickness=0,
             length=200, showvalue=True).grid(row=r, column=1, sticky='w', **pad); r += 1

    header('Status bubble')
    bubble_visible_var = check('Always show status bubble', 'bubble_visible')

    note = tk.Label(
        win,
        text=(
            'ollama  →  fast, free, local, no quota.  Best for high-frequency dictation.\n'
            'claude  →  uses your Max plan window. Higher quality, slower (~7-15s).\n'
            "language: '.en' models are English-only — for vi/auto use large-v3-turbo.\n"
            'Tip: drag the bubble anywhere; it remembers where you leave it.'
        ),
        bg=DARK, fg='#888', font=('Segoe UI', 9, 'italic'), justify='left',
    )
    note.grid(row=r, column=0, columnspan=2, sticky='w', padx=14, pady=(10, 4)); r += 1

    def save():
        new_cfg = dict(cfg)
        new_cfg['hotkey'] = ptt_state['tokens'] or ['ctrl', 'cmd']  # ptt never empty
        new_cfg['toggle_hotkey'] = toggle_state['tokens']           # empty = disabled
        new_cfg['model'] = model_var.get()
        new_cfg['device'] = device_var.get()
        new_cfg['language'] = lang_var.get().strip() or 'en'
        new_cfg['compute_type'] = 'int8_float16' if device_var.get() == 'cuda' else 'int8'
        new_cfg['cleanup_enabled'] = cleanup_var.get()
        new_cfg['cleanup_backend'] = backend_var.get()
        new_cfg['cleanup_model'] = cm_var.get()
        new_cfg['voice_commands'] = voice_cmd_var.get()
        new_cfg['continuous_pause'] = round(float(pause_var.get()), 1)
        new_cfg['tone_matching'] = tone_var.get()
        new_cfg['sounds'] = sounds_var.get()
        new_cfg['cue_sound'] = sound_var.get()
        new_cfg['cue_file'] = cue_file_var.get()
        new_cfg['cue_volume'] = int(vol_var.get())
        new_cfg['bubble_visible'] = bubble_visible_var.get()
        on_save(new_cfg)
        messagebox.showinfo('Murmur', 'Settings saved. Restart Murmur if Whisper model or device changed.')
        win.destroy()

    tk.Button(win, text='Save', command=save, width=14, bg='#3a8a3a', fg='white',
              font=('Segoe UI', 10, 'bold'), relief='flat', padx=8, pady=4).grid(
        row=r, column=0, columnspan=2, pady=16)
    return win


def _text_editor_window(root, path, title, header_text, placeholder):
    win = tk.Toplevel(root)
    win.title(title)
    win.geometry('520x560')
    win.attributes('-topmost', True)
    win.configure(bg=DARK)

    tk.Label(win, text=header_text, bg=DARK, fg='#ccc',
             font=('Segoe UI', 10), justify='left').pack(padx=14, pady=10, anchor='w')

    text = tk.Text(win, font=('Consolas', 10), wrap='none', bg='#252525', fg='#ddd',
                   insertbackground='white', relief='flat')
    text.pack(fill='both', expand=True, padx=14, pady=4)
    text.insert('1.0', path.read_text(encoding='utf-8') if path.exists() else placeholder)

    def save():
        path.write_text(text.get('1.0', 'end-1c'), encoding='utf-8')
        messagebox.showinfo('Murmur', 'Saved.')
        win.destroy()
    tk.Button(win, text='Save', command=save, width=14, bg='#3a8a3a', fg='white',
              font=('Segoe UI', 10, 'bold'), relief='flat', padx=8, pady=4).pack(pady=12)


def open_dictionary_window(root, dict_path):
    _text_editor_window(
        root, dict_path, 'Murmur — Dictionary',
        'Custom words & phrases (one per line).\n'
        'These bias Whisper recognition and are protected during cleanup.',
        '# One term per line.\n')


def open_commands_window(root, commands_path):
    _text_editor_window(
        root, commands_path, 'Murmur — Voice Commands',
        'Voice commands — "phrase = action", one per line.\n'
        'Say a phrase on its own to run it. See the comments for available actions.\n'
        'Changes apply immediately — no restart needed.',
        '# phrase = action\nnew line = newline\n')


def open_stats_window(root, stats):
    win = tk.Toplevel(root)
    win.title('Murmur — Stats')
    win.geometry('340x200')
    win.attributes('-topmost', True)
    win.configure(bg=DARK)

    rows = [
        ('Total words dictated', f"{stats.get('total_words', 0):,}"),
        ('Sessions', f"{stats.get('sessions', 0):,}"),
        ('Last session', stats.get('last_session') or '—'),
    ]
    for r, (label, val) in enumerate(rows):
        tk.Label(win, text=label + ':', bg=DARK, fg='#aaa', font=('Segoe UI', 10)).grid(row=r, column=0, sticky='w', padx=14, pady=10)
        tk.Label(win, text=str(val), bg=DARK, fg='#ddd', font=('Segoe UI', 11, 'bold')).grid(row=r, column=1, sticky='w', padx=8, pady=10)
