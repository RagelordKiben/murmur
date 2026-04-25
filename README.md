# Murmur

> A free, local, open-source alternative to [Wispr Flow](https://wisprflow.ai). Hold a hotkey, talk, watch cleaned-up text appear in any app — running 100% on your own machine.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-green.svg)](https://www.python.org/)
[![Platform: Windows](https://img.shields.io/badge/platform-Windows-lightgrey.svg)]()

![Murmur tray icon](docs/tray.png)

## Why

Wispr Flow is great. Hold `Ctrl+Win`, talk, release, and clean punctuated text drops into whatever app you're focused on. No "comma. period. new paragraph" nonsense.

It's also $15/month, and your voice goes to their cloud.

**Murmur** does the same thing — locally, free, and offline. Your audio never leaves your machine.

## How It Works

```
Hold Ctrl+Win  →  faster-whisper transcribes locally
              →  qwen2.5:7b on Ollama cleans up filler / punctuation
              →  cleaned text pastes into the focused field
```

End-to-end: ~1.5 seconds on a mid-range GPU. No API keys. No subscriptions. No data leaves the box.

## Quickstart

**Prerequisites** (one-time):
1. **[Ollama](https://ollama.com)** installed, with a cleanup model pulled:
   ```
   ollama pull qwen2.5:7b
   ```
2. **Python 3.11+** with the dependencies in `requirements.txt`:
   ```
   pip install -r requirements.txt
   ```

**Install Murmur:**
```bat
git clone https://github.com/BrianDurand1/murmur.git
cd murmur
install.bat
start_murmur.bat
```

A gold "M" appears in your system tray. Hold **Ctrl+Win**, talk, release. Done.

## Features

- **Hold-to-talk hotkey** — default `Ctrl+Win`, configurable to any chord
- **Local transcription** — faster-whisper `small.en`, GPU when available, CPU fallback
- **AI cleanup** — Ollama HTTP (default `qwen2.5:7b`) removes fillers, adds punctuation, capitalizes proper nouns, applies spoken self-corrections
- **Custom dictionary** — bias Whisper recognition AND protect proper nouns during cleanup
- **Tray icon** with idle / listening / processing states
- **Floating status bubble** so you know it's listening
- **Settings UI** for hotkey, model, cleanup backend
- **Stats tracking** — total words dictated, sessions, last session
- **Auto-space** between consecutive dictations within 30 seconds
- **Optional Claude Code CLI cleanup backend** if you prefer Anthropic models on your Max plan

## The Cleanup Prompt (the secret sauce)

Local 7B models will paraphrase, formalize, and add words you didn't say if you just tell them to "clean this up." The fix is a strict prompt with worked examples:

```text
You clean up dictated text. Follow the rules and study the examples.

RULES:
- Remove meaningless filler words (um, uh, like, you know).
- Add natural punctuation and capitalization.
- Always end statements with a period, questions with a question mark.
- Capitalize proper nouns, product names, and brand names.
- Apply spoken self-corrections — remove the struck-out portion.
- Preserve these terms EXACTLY as written: {dictionary}
- Do NOT add words that weren't spoken. Do NOT change meaning.
  Do NOT formalize tone. Do NOT wrap in quotes or add preamble.

EXAMPLES:
INPUT:   hey can you um send me the file when you get a chance
CLEANED: Hey can you send me the file when you get a chance?

INPUT:   i was thinking we could go to the store actually no the park
CLEANED: I was thinking we could go to the park.

INPUT:   testing this whisper flow thing
CLEANED: Testing this Whisper Flow thing.

INPUT:   yeah so the bambu printer is having issues again
CLEANED: Yeah, so the Bambu printer is having issues again.

NOW CLEAN THIS DICTATION:
INPUT: {text}
CLEANED:
```

The full prompt with all examples lives in [`murmur.py`](murmur.py) — search for `CLEANUP_PROMPT`.

## The Build Prompt

If you want to recreate Murmur from scratch with [Claude Code](https://claude.com/code), paste this prompt in an empty folder:

```text
Build me Murmur — a free local replacement for Wispr Flow on Windows.

Hard requirements:
- Hold Ctrl+Win to record, release to paste cleaned text into the focused field
- faster-whisper for local transcription (small.en, GPU when available, CPU fallback)
- Ollama HTTP for cleanup (default qwen2.5:7b) — strict prompt with worked examples
- pystray tray icon with idle / listening / processing states
- Floating Tk bubble for status
- Custom dictionary file that biases Whisper AND is preserved during cleanup
- Settings UI (Tkinter) for hotkey, model, cleanup backend
- Stats: total words dictated, sessions
- No paid API. Strip ANTHROPIC_API_KEY before any subprocess call
- Single-file Python app, launch via pythonw.exe so no console window appears
- Add Ollama's bundled CUDA DLLs to PATH at startup
```

## Configuration

Murmur stores everything in `~/.murmur/`:

| File | Purpose |
|---|---|
| `config.json` | Hotkey, model, device, cleanup backend |
| `dictionary.txt` | Custom proper nouns, one per line |
| `stats.json` | Words dictated, session count |
| `murmur.log` | Runtime log (state transitions, errors) |

Right-click the tray icon for **Settings**, **Edit Dictionary**, or **Stats**.

## Performance

On an NVIDIA RTX 3060 12GB:

| Stage | Time |
|---|---|
| Whisper transcribe (small.en, GPU) | ~0.3–0.5s |
| Qwen 2.5 7B cleanup (warm) | ~0.7–1.2s |
| Paste | ~0.1s |
| **Total end-to-end** | **~1.5s** |

Wispr Flow on cloud GPU is faster (~1s), but Murmur is free, local, and unlimited.

## Comparison

| Feature | Wispr Flow ($15/mo) | Murmur (free) |
|---|---|---|
| Cost | $15/month | **$0/month** |
| Privacy | Cloud-only | **100% local** |
| End-to-end latency | ~1s | ~1.5s |
| Streaming transcription | Yes | Not yet |
| Custom dictionary | Yes | Yes |
| Filler removal & punctuation | Proprietary model | Qwen 2.5 7B + strict prompt |
| Tone-matching per app | Yes (Slack vs Gmail) | Single tone |
| Mobile apps | iOS + Android | Desktop only |
| Open source | No | **Yes — one Python file** |

## Roadmap

- [ ] Streaming transcription (transcribe-while-talking)
- [ ] App-context-aware tone matching (Slack vs Gmail vs IDE)
- [ ] Voice command mode (select text + "rewrite shorter")
- [ ] Snippets / voice-triggered text expansion
- [ ] macOS + Linux support

PRs welcome.

## Acknowledgments

- **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)** — local transcription
- **[Ollama](https://ollama.com)** — local LLM runtime
- **[Qwen 2.5](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)** — cleanup model
- **[Claude Code](https://claude.com/code)** — wrote the whole app in one session
- **[Wispr Flow](https://wisprflow.ai)** — for proving the UX is worth it

## License

MIT — see [LICENSE](LICENSE).

---

**Built by [Brian Durand](https://everydayaiwithbrian.com)** — read the full build story on [Everyday AI with Brian](https://everydayaiwithbrian.com/blog/replace-wispr-flow.html).
