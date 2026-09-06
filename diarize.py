#!/usr/bin/env python3
"""Диаризация транскрипта: senko (CoreML/ANE, локально, без токенов).

Usage:
    diarize.py <media_file> <run_dir>
    diarize.py --rename "SPEAKER_00=Антон,SPEAKER_01=Мария" <run_dir>

Первый вызов: прогоняет senko по аудио, сопоставляет спикеров с сегментами
из <run_dir>/segments.json по пересечению интервалов, пишет:
    <run_dir>/diarized.json   — реплики со спикерами
    <run_dir>/transcript.md   — markdown с метками SPEAKER_NN

--rename: применяет имена к diarized.json и перегенерирует transcript.md.
Быстро, без ML — сегменты не пересчитываются.

Прогресс и сводка по спикерам — в stderr, путь к transcript.md — в stdout.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

# Реплики одного спикера, разделённые паузой короче порога, склеиваются
TURN_MERGE_GAP_SEC = 3.0


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def format_ts(sec):
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def to_wav_16k(media_file, wav_path):
    for ffmpeg in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "ffmpeg"):
        try:
            subprocess.run(
                [ffmpeg, "-y", "-i", media_file, "-vn", "-ac", "1",
                 "-ar", "16000", "-acodec", "pcm_s16le", wav_path],
                check=True, capture_output=True,
            )
            return
        except FileNotFoundError:
            continue
    raise RuntimeError("ffmpeg не найден")


def speaker_label(raw):
    if isinstance(raw, int):
        return f"SPEAKER_{raw:02d}"
    return str(raw)


def run_senko(media_file):
    """Senko → [{start, end, speaker}] в секундах."""
    try:
        import senko
    except ImportError:
        log("ОШИБКА: senko не установлен. Нужен Python 3.11 и пересборка venv — "
            "запусти setup.sh из папки скилла.")
        sys.exit(3)

    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "audio.wav")
        log("Конвертирую в WAV 16 кГц…")
        to_wav_16k(media_file, wav)
        log("Диаризация (senko)…")
        diarizer = senko.Diarizer(device="auto", warmup=False, quiet=True)
        result = diarizer.diarize(wav, generate_colors=False)

    raw = result.get("merged_segments") or result.get("segments") or []
    out = []
    for seg in raw:
        out.append({
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "speaker": speaker_label(seg.get("speaker", "SPEAKER_00")),
        })
    return out


def assign_speakers(segments, speaker_segments):
    """Каждому whisper-сегменту — спикер с максимальным пересечением."""
    for seg in segments:
        best, best_overlap = None, 0.0
        mid = (seg["start"] + seg["end"]) / 2
        best_dist, nearest = float("inf"), None
        for sp in speaker_segments:
            overlap = min(seg["end"], sp["end"]) - max(seg["start"], sp["start"])
            if overlap > best_overlap:
                best, best_overlap = sp["speaker"], overlap
            dist = abs((sp["start"] + sp["end"]) / 2 - mid)
            if dist < best_dist:
                best_dist, nearest = dist, sp["speaker"]
        seg["speaker"] = best or nearest or "SPEAKER_00"
    return segments


def group_turns(segments):
    """Подряд идущие сегменты одного спикера → реплики (turns)."""
    turns = []
    for seg in segments:
        if (turns
                and turns[-1]["speaker"] == seg["speaker"]
                and seg["start"] - turns[-1]["end"] <= TURN_MERGE_GAP_SEC):
            turns[-1]["end"] = seg["end"]
            turns[-1]["text"] += " " + seg["text"]
        else:
            turns.append({
                "speaker": seg["speaker"],
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
            })
    return turns


def write_outputs(run_dir, turns):
    with open(os.path.join(run_dir, "diarized.json"), "w", encoding="utf-8") as f:
        json.dump({"turns": turns}, f, ensure_ascii=False, indent=2)

    md_path = os.path.join(run_dir, "transcript.md")
    lines = []
    for t in turns:
        lines.append(f"**{t['speaker']}** `[{format_ts(t['start'])}]`")
        lines.append("")
        lines.append(t["text"])
        lines.append("")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return md_path


def report_speakers(turns):
    stats = {}
    for t in turns:
        s = stats.setdefault(t["speaker"], {"time": 0.0, "turns": 0, "first": None})
        s["time"] += t["end"] - t["start"]
        s["turns"] += 1
        if s["first"] is None:
            s["first"] = t["text"][:120]
    log(f"\nСпикеров: {len(stats)}")
    for name in sorted(stats):
        s = stats[name]
        log(f"  {name}: {format_ts(s['time'])} речи, {s['turns']} реплик")
        log(f"    первая реплика: «{s['first']}»")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("media_file", nargs="?", help="исходный аудио/видео файл")
    p.add_argument("run_dir", help="папка запуска transcribe.py")
    p.add_argument("--rename", default=None,
                   help='карта имён: "SPEAKER_00=Антон,SPEAKER_01=Мария"')
    args = p.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    diarized_path = os.path.join(run_dir, "diarized.json")

    if args.rename:
        if not os.path.exists(diarized_path):
            log(f"ОШИБКА: нет {diarized_path} — сначала прогони диаризацию.")
            sys.exit(1)
        mapping = dict(pair.split("=", 1) for pair in args.rename.split(","))
        with open(diarized_path, encoding="utf-8") as f:
            turns = json.load(f)["turns"]
        for t in turns:
            t["speaker"] = mapping.get(t["speaker"], t["speaker"])
        md_path = write_outputs(run_dir, turns)
        log(f"Имена применены: {mapping}")
        print(md_path)
        return

    if not args.media_file:
        p.error("укажи исходный файл (или --rename)")

    segments_path = os.path.join(run_dir, "segments.json")
    if not os.path.exists(segments_path):
        log(f"ОШИБКА: нет {segments_path} — сначала прогони transcribe.py.")
        sys.exit(1)

    with open(segments_path, encoding="utf-8") as f:
        segments = json.load(f)["segments"]
    if not segments:
        log("ОШИБКА: в segments.json нет сегментов.")
        sys.exit(1)

    speaker_segments = run_senko(os.path.abspath(args.media_file))
    if not speaker_segments:
        log("Senko не нашёл спикеров — оставляю транскрипт без разметки.")
        sys.exit(2)

    segments = assign_speakers(segments, speaker_segments)
    turns = group_turns(segments)
    md_path = write_outputs(run_dir, turns)
    report_speakers(turns)
    print(md_path)


if __name__ == "__main__":
    main()
