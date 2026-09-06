#!/usr/bin/env python3
"""Detect and repair hallucination zones in Whisper transcripts.

Usage:
    repair_segments.py <source_file> <segments_json> [language|auto] [--no-enhance]
                       [--model <hf-repo>]

Outputs patched JSON to stdout, progress to stderr.
"""

import sys
import os
import json
import re
import subprocess
import tempfile
from collections import Counter

# Russian and English filler words Whisper hallucinates under silence/noise
HALLUCINATION_PATTERNS = [
    r'(\bну\b[\s,]*){5,}',
    r'(\bвот\b[\s,]*){5,}',
    r'(\bда\b[\s,]*){5,}',
    r'(\bнет\b[\s,]*){5,}',
    r'(\bмм+\b[\s,]*){3,}',
    r'(\bэ+\b[\s,]*){3,}',
    r'(\buh\b[\s,]*){4,}',
    r'(\bum\b[\s,]*){4,}',
]

# If a single word makes up ≥60% of a segment with ≥6 words total → hallucination
WORD_RATIO_THRESHOLD = 0.60
MIN_WORDS_FOR_RATIO_CHECK = 6


def find_ffmpeg():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ai_hub = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    candidates = [
        os.path.join(ai_hub, ".venv", "bin", "ffmpeg"),
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "ffmpeg",
    ]
    for path in candidates:
        try:
            subprocess.run([path, "-version"], capture_output=True, check=True)
            return path
        except (FileNotFoundError, PermissionError, subprocess.CalledProcessError, OSError):
            continue
    return None


def is_hallucination(text):
    text = text.strip()
    if not text:
        return False

    for pattern in HALLUCINATION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) >= MIN_WORDS_FOR_RATIO_CHECK:
        top_word, top_count = Counter(words).most_common(1)[0]
        if top_count / len(words) >= WORD_RATIO_THRESHOLD:
            return True

    return False


def find_hallucination_zones(segments):
    """Group consecutive hallucination segments into zones.

    Returns list of (start_sec, end_sec, [segment_indices]).
    """
    bad = [i for i, s in enumerate(segments) if is_hallucination(s["text"])]
    if not bad:
        return []

    zones = []
    run_start = bad[0]
    run_end = bad[0]

    for idx in bad[1:]:
        if idx == run_end + 1:
            run_end = idx
        else:
            zones.append((run_start, run_end))
            run_start = run_end = idx
    zones.append((run_start, run_end))

    return [
        (segments[z[0]]["start"], segments[z[1]]["end"], list(range(z[0], z[1] + 1)))
        for z in zones
    ]


def extract_audio_segment(source_file, start, end, output_path, ffmpeg, enhance):
    """Extract a time range from source and write as 16kHz mono WAV."""
    pad = 0.5
    t_start = max(0.0, start - pad)
    t_end = end + pad

    af_filters = []
    if enhance:
        af_filters = [
            "highpass=f=150",
            "lowpass=f=8000",
            "afftdn=nf=-25",
            "loudnorm=I=-16:TP=-1.5:LRA=11",
        ]

    cmd = [
        ffmpeg, "-y",
        "-i", source_file,
        "-ss", f"{t_start:.3f}",
        "-to", f"{t_end:.3f}",
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-acodec", "pcm_s16le",
    ]
    if af_filters:
        cmd += ["-af", ",".join(af_filters)]
    cmd.append(output_path)

    subprocess.run(cmd, check=True, capture_output=True)


DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"


def retranscribe(audio_path, language, model=DEFAULT_MODEL):
    import mlx_whisper
    kwargs = {
        "path_or_hf_repo": model,
        "condition_on_previous_text": False,
    }
    if language:
        kwargs["language"] = language
    result = mlx_whisper.transcribe(audio_path, **kwargs)
    return result["text"].strip()


def repair(source_file, segments_json_path, language=None, enhance=True,
           model=DEFAULT_MODEL):
    with open(segments_json_path, encoding="utf-8") as f:
        data = json.load(f)
    segments = data["segments"]

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return None, "ffmpeg не найден. Проверьте .venv/bin/ffmpeg или установите ffmpeg."

    zones = find_hallucination_zones(segments)
    if not zones:
        return segments, "Галлюцинаций не обнаружено — транскрипт чистый."

    total_lost = sum(end - start for start, end, _ in zones)
    report = [
        f"Обнаружено зон с галлюцинациями: {len(zones)} (~{total_lost:.0f} сек потеряно)",
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, (start, end, indices) in enumerate(zones):
            dur = end - start
            mm, ss = divmod(int(start), 60)
            report.append(f"\n  Зона {i + 1}: {mm:02d}:{ss:02d} — {end:.1f}s ({dur:.0f}s)")

            seg_wav = os.path.join(tmpdir, f"zone_{i}.wav")
            try:
                extract_audio_segment(source_file, start, end, seg_wav, ffmpeg, enhance)
                new_text = retranscribe(seg_wav, language, model)

                if new_text and not is_hallucination(new_text):
                    preview = new_text[:100] + ("..." if len(new_text) > 100 else "")
                    report.append(f"    ✓ Восстановлено: «{preview}»")
                    merged = {
                        "start": start,
                        "end": end,
                        "text": new_text,
                        "repaired": True,
                    }
                    for idx in indices:
                        segments[idx]["_remove"] = True
                    segments[indices[0]] = merged
                else:
                    report.append("    ✗ Не удалось восстановить (тишина или шум)")
                    for idx in indices:
                        segments[idx]["text"] = "[неразборчиво]"
                        segments[idx]["repaired"] = False

            except subprocess.CalledProcessError as e:
                report.append(f"    ✗ Ошибка ffmpeg: {e.stderr.decode('utf-8', errors='replace')[-200:]}")
            except Exception as e:
                report.append(f"    ✗ Ошибка: {e}")

    segments = [s for s in segments if not s.get("_remove")]
    return segments, "\n".join(report)


def segments_to_text(segments):
    return " ".join(s["text"] for s in segments if s.get("text", "").strip())


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    source_file = sys.argv[1]
    segments_json = sys.argv[2]
    language = None
    enhance = True
    model = DEFAULT_MODEL

    rest = sys.argv[3:]
    i = 0
    while i < len(rest):
        arg = rest[i]
        if arg == "--no-enhance":
            enhance = False
        elif arg == "--model":
            i += 1
            model = rest[i] if i < len(rest) else model
        elif arg != "auto":
            language = arg
        i += 1

    if not os.path.exists(source_file):
        print(f"ОШИБКА: Файл не найден: {source_file}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(segments_json):
        print(f"ОШИБКА: JSON не найден: {segments_json}", file=sys.stderr)
        sys.exit(1)

    segments, report = repair(source_file, segments_json, language, enhance, model)
    print(report, file=sys.stderr)

    if segments is None:
        sys.exit(1)

    result = {
        "segments": segments,
        "text": segments_to_text(segments),
    }
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
