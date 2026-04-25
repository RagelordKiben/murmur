# Short script — "Murmur: replaced Wispr Flow with $0 of local AI"

**Format:** ~55–60 seconds, vertical, voice-over + screen capture
**Hook density:** open within first 1.5s
**CTA:** link to blog post on EverydayAIwithBrian.com

---

## Hook (0–3s)

**ON SCREEN:** Wispr Flow's $15/mo pricing page, then close it.

**VO:** "Wispr Flow is fifteen bucks a month. I built Murmur, a free local clone, in fifteen minutes."

---

## Setup (3–12s)

**ON SCREEN:** Quick montage — hold key, talk, paste happens. Repeat in three different apps (browser, Slack, VS Code).

**VO:** "If you haven't tried Wispr Flow — hold a hotkey, talk, it pastes cleaned-up text into whatever app you're in. Faster than typing, by a lot. But your audio goes to their cloud, and the bill stacks up."

---

## The reveal (12–35s)

**ON SCREEN:** Side-by-side. Wispr Flow on the left, Murmur on the right. Same dictation. Both paste.

**VO:** "Murmur runs one hundred percent local. Whisper for the transcribing. Qwen 2.5 on Ollama for the cleanup pass — that's what removes the ums and adds punctuation. Both models open-source, both on my GPU, zero data leaves the machine."

**ON SCREEN:** Show the tray icon, the gold M. Show the cleanup prompt scrolling.

**VO:** "The whole thing is one Python file plus a system prompt. That prompt is the secret sauce — strict rules and a few examples teach a 7B model to clean up dictation as well as a cloud service."

---

## Speed proof (35–48s)

**ON SCREEN:** Stopwatch. Hold key, dictate one sentence, release, paste. Show 1.5 seconds end to end.

**VO:** "End to end, hold-to-paste — about a second and a half on my 3060. Wispr is faster on cloud GPUs, but for free, local, and unlimited? Good trade."

---

## Payoff + CTA (48–60s)

**ON SCREEN:** Pull up blog post on EverydayAIwithBrian.com.

**VO:** "Full code, the cleanup prompt, and the exact prompt I gave Claude Code to build it — link in the description. Try Murmur. Cancel Wispr."

---

## Build prompt (paste into Claude Code)

```
Build me Murmur — a free local replacement for Wispr Flow on Windows.

Hard requirements:
- Hold Ctrl+Win to record, release to paste cleaned text into the focused field
- faster-whisper for local transcription (small.en, GPU when available, fallback to CPU/int8)
- Ollama HTTP for cleanup (default qwen2.5:7b) — remove fillers, add punctuation, capitalize proper nouns, apply self-corrections
- pystray tray icon with idle/listening/processing states + a floating Tk bubble
- Custom dictionary file that biases Whisper recognition AND is preserved during cleanup
- Settings UI (Tkinter) for hotkey, model, cleanup backend
- Stats tracking: total words dictated, sessions
- No paid API. Strip ANTHROPIC_API_KEY before any subprocess call
- Single-file Python app, launch via pythonw.exe so no console window appears
- Add Ollama's bundled CUDA DLLs to PATH at startup so faster-whisper can use the GPU without a separate CUDA install
```

## Cleanup prompt (the secret sauce)

The cleanup prompt is in `%~dp0.\murmur.py` — search for `CLEANUP_PROMPT`. Strict rules + 7 worked examples covering filler removal, self-correction, proper-noun capitalization, and short utterances.
