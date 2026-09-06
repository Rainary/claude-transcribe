# claude-transcribe

🇷🇺 [Русская версия](README.md)

A Claude Code skill: audio or video → transcript → speakers → ready-to-use
memo. Fully local on Apple Silicon — no API keys, no tokens, no internet
after setup.

Drop in a meeting recording — a couple of minutes later you get a document:
decisions, action items with owners, open questions, and a full transcript
split into speaker turns with timestamps. The only thing the skill will ask
you is the speakers' names.

## What's inside

- **Transcription**: [mlx-whisper](https://github.com/ml-explore/mlx-examples)
  with `whisper-large-v3-turbo` (a flag switches to full large-v3 for
  difficult recordings).
- **Two-layer defense against Whisper hallucinations**: Silero VAD cuts long
  silences before they reach the model (hallucinations are born in silence),
  and a blacklist catches the classic phantom phrases Whisper picked up from
  YouTube subtitles.
- **Diarization**: [senko](https://github.com/narcotic-sh/senko) on
  CoreML/ANE — an hour of audio is speaker-labeled in seconds, no
  HuggingFace token required.
- **Deterministic speech cleanup** in scripts: filler interjections,
  stuttered function words, parasitic openers. Meaningful repetitions and
  personal speech markers are preserved.
- **Memo templates by recording type**: meeting (decisions / actions / open
  questions), interview (insights / pains / quotes), lecture (outline),
  voice note. The type is inferred from content — no questionnaires.
- **Editorial rules built in**: honesty-first memo guidelines (nothing that
  wasn't said makes it into the memo) plus a Russian typography pass.

## Speed

Measured on a MacBook M3 Pro / 18 GB, real recordings:

| Recording | Transcription | Diarization |
|---|---|---|
| 5:40 meeting (video, 2 speakers) | 20 s (×16 realtime) | 38 s |
| 44:39 lecture (video, monologue) | 2:18 (×19) | not needed |

The previous setup (large-v3, no VAD) ran at ×8.7 on the same files — the
current configuration is roughly 2–3× faster with the same quality.

The LLM never rewrites the transcript — all mechanical work is done by
scripts, Claude only writes the memo. That's both speed and token economy.

## Requirements

- Apple Silicon Mac (M1+), macOS 14+.
- Python 3.11 — if missing, `setup.sh` downloads the official installer and
  opens it (the only manual step; admin password required).
- ffmpeg — if missing, `setup.sh` installs it via Homebrew.
- Disk: ~1.6 GB for the Whisper model (downloaded on first run) and ~2 GB
  for the venv.

## Install

As a Claude Code plugin:

```text
/plugin marketplace add Rainary/claude-transcribe
/plugin install transcribe@claude-transcribe
```

Or with one command, without the plugin system:

```bash
curl -fsSL https://raw.githubusercontent.com/Rainary/claude-transcribe/main/install.sh | bash
```

After installing, tell Claude Code "check the transcribe environment" — it
runs `--check` and, if anything is missing, launches the setup doctor.

## Usage

In Claude Code, in plain language:

```text
transcribe ~/Desktop/meeting.mov
make an outline from ~/Downloads/lecture.mp4
```

Flags for special cases (Claude applies them if you ask): `--model large`
for maximum quality, `--language ru|en` if auto-detection guessed wrong,
`--precise` for word-level timestamps, `--no-vad`, `--no-cleanup`.

Formats: mp3, wav, m4a, ogg, webm, flac, aac, opus, mp4, mov, mkv, m4v, avi.

## How it works

```text
file → ffmpeg (16 kHz mono) → Silero VAD (speech regions)
     → mlx-whisper per region → hallucination blacklist → filler cleanup
     → senko (speakers) → timestamp matching → transcript.md
     → Claude: memo following references/memo-style.md → typography → document
```

Every run gets its own folder with `meta.json` (stats: speed, language,
hallucinations dropped), `segments.json`, `transcript.txt` and
`transcript.md`.

## Limitations

- Apple Silicon only: transcription rides on MLX, diarization on CoreML.
- Speech cleanup and the hallucination blacklist are tuned for Russian; the
  memo style guide is written in Russian. Whisper itself is multilingual and
  English works fine; other languages are untested.
- Overlapping speech (two voices at once) is the weak spot of any ASR —
  those spots may transcribe imprecisely.
- Diarization requires Python 3.11 (the senko macOS wheel is built for
  cp311).

## Roadmap

- Live mode: record a call (system audio + mic, no meeting bot) with rolling
  transcription — memo ready a minute after the call ends.
- Optional fast mode on parakeet-tdt (another ~5× over turbo).

## Credits

[mlx-whisper](https://github.com/ml-explore/mlx-examples) ·
[Silero VAD](https://github.com/snakers4/silero-vad) ·
[senko](https://github.com/narcotic-sh/senko) ·
[OpenAI Whisper](https://github.com/openai/whisper)

The editorial layer stands on the shoulders of Rodion Scryabin's
([beaverbeard](https://github.com/beaverbeard)) skills: the typography pass
is a bundled copy of his [milchin](https://github.com/beaverbeard/milchin)
(MIT, see [LICENSE-milchin](LICENSE-milchin)), and the memo style rules are
distilled from his [slopotron](https://github.com/beaverbeard/slopotron)
and [chukovsky](https://github.com/beaverbeard/chukovsky).

## License

[MIT](LICENSE)
