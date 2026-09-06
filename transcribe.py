#!/usr/bin/env python3
"""Локальная транскрипция аудио/видео: ffmpeg → Silero VAD → mlx-whisper.

Полностью офлайн, без токенов и внешних API. Apple Silicon (MLX).

Usage:
    transcribe.py <file> [--model turbo|large|<hf-repo>] [--language ru|en|auto]
                  [--out-dir DIR] [--no-vad] [--no-cleanup]
    transcribe.py --check

Пайплайн:
    1. Декодирование в 16 кГц моно (ffmpeg, через mlx-whisper).
    2. Silero VAD: речевые зоны; длинные паузы выбрасываются до модели
       (профилактика галлюцинаций + ускорение), короткие остаются —
       по ним Whisper ставит пунктуацию.
    3. mlx-whisper по зонам с анти-галлюцинационными параметрами.
    4. Детерминированная чистка: междометия, повторы, RU-галлюцинации
       («Субтитры сделал…»).

Результат — папка запуска: meta.json, segments.json, transcript.txt.
Путь к папке печатается в stdout, прогресс — в stderr.
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

MODELS = {
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "large": "mlx-community/whisper-large-v3-mlx",
}

AUDIO_FORMATS = {".mp3", ".wav", ".m4a", ".ogg", ".webm", ".flac", ".aac", ".opus"}
VIDEO_FORMATS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

SAMPLE_RATE = 16000

# Зоны речи, разделённые паузой короче этого порога, склеиваются в один
# регион — внутренние паузы уходят в модель и дают знаки препинания.
MERGE_GAP_SEC = 1.5

# Пауза между сегментами длиннее этого порога начинает новый абзац
# в transcript.md
PARA_GAP_SEC = 1.5

# Типовые галлюцинации Whisper на русском (обучен на субтитрах YouTube).
# Сегмент, содержащий такую строку, выбрасывается целиком.
RU_HALLUCINATION_BLACKLIST = [
    "субтитры сделал",
    "субтитры делал",
    "субтитры создавал",
    "dimatorzok",
    "редактор субтитров",
    "корректор а.егорова",
    "спасибо за просмотр",
    "продолжение следует",
    "подписывайтесь на канал",
]

# Междометия без смысловой нагрузки — удаляются вместе с прилегающей запятой.
INTERJECTIONS = r"(?:э-?э+|ээ+|мм+|м-м+|эм+|а-а+|ммм+)"

# Служебные слова, у которых подряд идущие повторы («я я хотел») схлопываются.
DUP_COLLAPSE_WORDS = {
    "ну", "вот", "это", "я", "ты", "мы", "вы", "он", "она", "они",
    "и", "в", "на", "не", "что", "как", "то", "же", "бы", "так", "у",
}

# Вводные-паразиты в начале сегмента: срезаются только вместе с запятой —
# запятая сигналит именно паразитную интонацию («Ну, я думаю…»).
LEADING_FILLERS = r"(?:Ну|Вот|Значит|Короче|В общем|Так вот)"


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def ensure_ffmpeg_on_path():
    for d in ("/opt/homebrew/bin", "/usr/local/bin"):
        if d not in os.environ.get("PATH", ""):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


def check_environment():
    """Проверка окружения: python, зависимости, ffmpeg, модели."""
    ok = True
    log(f"Python: {sys.version.split()[0]}")

    import shutil
    ffmpeg = shutil.which("ffmpeg")
    log(f"ffmpeg: {ffmpeg or 'НЕ НАЙДЕН'}")
    ok = ok and bool(ffmpeg)

    for mod, label in [("mlx_whisper", "mlx-whisper"), ("silero_vad", "silero-vad")]:
        try:
            __import__(mod)
            log(f"{label}: OK")
        except ImportError as e:
            log(f"{label}: НЕ УСТАНОВЛЕН ({e})")
            ok = False

    try:
        import senko  # noqa: F401
        log("senko (диаризация): OK")
    except ImportError:
        hint = ""
        if sys.version_info < (3, 10):
            hint = " (нужен Python 3.11 — запусти setup.sh из папки скилла)"
        log("senko (диаризация): не установлен — транскрипция работает, "
            f"диаризация нет{hint}")

    cache = os.path.expanduser("~/.cache/huggingface/hub")
    for alias, repo in MODELS.items():
        path = os.path.join(cache, "models--" + repo.replace("/", "--"))
        state = "скачана" if os.path.isdir(path) else "будет скачана при первом запуске"
        log(f"модель {alias} ({repo}): {state}")

    log("Окружение: " + ("OK" if ok else "ЕСТЬ ПРОБЛЕМЫ"))
    return ok


def load_audio(path):
    import numpy as np
    from mlx_whisper.audio import load_audio as _load
    # mlx-whisper отдаёт mx.array — приводим к numpy (16 кГц, моно, float32)
    return np.array(_load(path), copy=False)


def detect_speech_regions(audio):
    """Silero VAD → регионы речи [(start_sec, end_sec)], длинные паузы выброшены."""
    import torch
    from silero_vad import load_silero_vad, get_speech_timestamps

    model = load_silero_vad()
    ts = get_speech_timestamps(
        torch.from_numpy(audio),
        model,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=400,
        speech_pad_ms=300,
    )
    if not ts:
        return []

    regions = []
    cur_start, cur_end = ts[0]["start"], ts[0]["end"]
    for t in ts[1:]:
        if (t["start"] - cur_end) / SAMPLE_RATE <= MERGE_GAP_SEC:
            cur_end = t["end"]
        else:
            regions.append((cur_start, cur_end))
            cur_start, cur_end = t["start"], t["end"]
    regions.append((cur_start, cur_end))

    return [(s / SAMPLE_RATE, e / SAMPLE_RATE) for s, e in regions]


def transcribe_region(audio_slice, model_repo, language, precise=False):
    import mlx_whisper
    kwargs = {
        "path_or_hf_repo": model_repo,
        "condition_on_previous_text": False,
    }
    if precise:
        # Пословные таймштампы ~вдвое медленнее; сегментных хватает и для
        # диаризации, и для мемо. Включаем только по запросу.
        kwargs["word_timestamps"] = True
        kwargs["hallucination_silence_threshold"] = 2.0
    if language:
        kwargs["language"] = language
    return mlx_whisper.transcribe(audio_slice, **kwargs)


def is_blacklisted(text):
    low = text.lower()
    return any(pat in low for pat in RU_HALLUCINATION_BLACKLIST)


def clean_text(text):
    """Консервативная чистка. Возвращает (text, число_правок)."""
    edits = 0

    def sub(pattern, repl, s, flags=0):
        nonlocal edits
        new, n = re.subn(pattern, repl, s, flags=flags)
        edits += n
        return new

    # Междометия: «э-э», «мм» — вместе с прилегающей запятой
    text = sub(r"\s*\b" + INTERJECTIONS + r"\b[,…]?", "", text, flags=re.IGNORECASE)

    # Повторы служебных слов: «я я хотел» → «я хотел»
    def collapse(m):
        nonlocal edits
        if m.group(1).lower() in DUP_COLLAPSE_WORDS:
            edits += 1
            return m.group(1)
        return m.group(0)

    text = re.sub(r"\b(\w+)(?:[\s,]+\1\b)+", collapse, text, flags=re.IGNORECASE)

    # Вводный паразит с запятой в начале: «Ну, я думаю» → «Я думаю»
    def strip_leading(m):
        nonlocal edits
        edits += 1
        rest = m.group(1)
        return rest[:1].upper() + rest[1:]

    text = re.sub(r"^\s*" + LEADING_FILLERS + r",\s+(\S.*)", strip_leading, text)

    # Гигиена пробелов
    text = re.sub(r"\s+([,.!?…])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = re.sub(r"^[,.\s]+", "", text)

    return text, edits


def format_ts(sec):
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def format_paragraphs(segments):
    """Сегменты → markdown-абзацы по паузам, с таймкодом начала абзаца."""
    paras = []
    cur, cur_start, prev_end = [], None, None
    for seg in segments:
        if cur and seg["start"] - prev_end > PARA_GAP_SEC:
            paras.append((cur_start, " ".join(cur)))
            cur = []
        if not cur:
            cur_start = seg["start"]
        cur.append(seg["text"])
        prev_end = seg["end"]
    if cur:
        paras.append((cur_start, " ".join(cur)))
    return "\n\n".join(f"`[{format_ts(st)}]` {text}" for st, text in paras) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("file", nargs="?", help="аудио- или видеофайл")
    p.add_argument("--model", default="turbo",
                   help="turbo (дефолт) | large | HF-репозиторий")
    p.add_argument("--language", default="auto", help="ru | en | auto (дефолт)")
    p.add_argument("--out-dir", default=None,
                   help="папка результатов (дефолт: temp/transcribe/<имя>_<время>)")
    p.add_argument("--no-vad", action="store_true",
                   help="без VAD — сплошная транскрипция")
    p.add_argument("--no-cleanup", action="store_true",
                   help="без чистки междометий и повторов")
    p.add_argument("--precise", action="store_true",
                   help="пословные таймштампы + доп. защита от галлюцинаций "
                        "(~вдвое медленнее)")
    p.add_argument("--check", action="store_true",
                   help="проверить окружение и выйти")
    args = p.parse_args()

    ensure_ffmpeg_on_path()

    if args.check:
        sys.exit(0 if check_environment() else 1)

    if not args.file:
        p.error("укажи файл (или --check)")

    input_path = os.path.abspath(args.file)
    if not os.path.exists(input_path):
        log(f"ОШИБКА: файл не найден: {input_path}")
        sys.exit(1)

    ext = os.path.splitext(input_path)[1].lower()
    if ext not in AUDIO_FORMATS | VIDEO_FORMATS:
        log(f"ОШИБКА: неподдерживаемый формат: {ext}")
        sys.exit(1)

    model_repo = MODELS.get(args.model, args.model)
    language = None if args.language == "auto" else args.language

    if args.out_dir:
        out_dir = os.path.abspath(args.out_dir)
    else:
        stem = os.path.splitext(os.path.basename(input_path))[0]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Если скилл живёт в проекте с папкой temp/ (структура вида
        # <root>/.claude/skills/transcribe/) — складываем туда,
        # иначе в пользовательский кэш.
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
        if os.path.isdir(os.path.join(project_root, "temp")):
            base = os.path.join(project_root, "temp", "transcribe")
        else:
            base = os.path.expanduser("~/Library/Caches/claude-transcribe")
        out_dir = os.path.join(base, f"{stem}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    log(f"Файл: {os.path.basename(input_path)}")
    log(f"Модель: {model_repo}")

    audio = load_audio(input_path)
    duration = len(audio) / SAMPLE_RATE
    log(f"Длительность: {format_ts(duration)}")

    if args.no_vad:
        regions = [(0.0, duration)]
        speech_total = duration
    else:
        log("VAD: ищу речевые зоны…")
        regions = detect_speech_regions(audio)
        speech_total = sum(e - s for s, e in regions)
        if not regions:
            log("Речь не обнаружена — файл состоит из тишины/шума.")
            sys.exit(2)
        log(f"VAD: {len(regions)} зон, речи {format_ts(speech_total)} "
            f"из {format_ts(duration)} "
            f"(пропущено {100 * (1 - speech_total / duration):.0f}% тишины)")

    segments = []
    dropped = []
    cleanup_edits = 0
    detected_language = language

    for i, (start, end) in enumerate(regions, 1):
        log(f"[{i}/{len(regions)}] {format_ts(start)}–{format_ts(end)}…")
        s0, s1 = int(start * SAMPLE_RATE), int(end * SAMPLE_RATE)
        result = transcribe_region(audio[s0:s1], model_repo, detected_language,
                                   precise=args.precise)

        # Автоопределённый язык первой зоны фиксируем для остальных,
        # чтобы детекция не прыгала между зонами.
        if detected_language is None:
            detected_language = result.get("language")
            log(f"Язык: {detected_language}")

        for seg in result.get("segments", []):
            text = seg["text"].strip()
            if not text:
                continue
            if is_blacklisted(text):
                dropped.append({"start": round(start + seg["start"], 2),
                                "text": text})
                continue
            if not args.no_cleanup:
                text, n = clean_text(text)
                cleanup_edits += n
                if not text:
                    continue
            segments.append({
                "start": round(start + seg["start"], 2),
                "end": round(start + seg["end"], 2),
                "text": text,
            })

    elapsed = time.time() - t0
    full_text = " ".join(s["text"] for s in segments)

    meta = {
        "source": input_path,
        "created": datetime.now().isoformat(timespec="seconds"),
        "model": model_repo,
        "language": detected_language or "unknown",
        "duration_sec": round(duration, 1),
        "speech_sec": round(speech_total, 1),
        "regions": len(regions),
        "segments": len(segments),
        "dropped_hallucinations": dropped,
        "cleanup_edits": cleanup_edits,
        "elapsed_sec": round(elapsed, 1),
        "realtime_factor": round(duration / elapsed, 1) if elapsed else None,
        "vad": not args.no_vad,
        "cleanup": not args.no_cleanup,
        "precise": args.precise,
    }

    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "segments.json"), "w", encoding="utf-8") as f:
        json.dump({"segments": segments}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "transcript.txt"), "w", encoding="utf-8") as f:
        f.write(full_text + "\n")
    if segments:
        # transcript.md: абзацы по паузам; после диаризации diarize.py
        # перезапишет его версией с репликами по спикерам
        with open(os.path.join(out_dir, "transcript.md"), "w",
                  encoding="utf-8") as f:
            f.write(format_paragraphs(segments))

    log(f"\nГотово за {format_ts(elapsed)} "
        f"(×{meta['realtime_factor']} от реального времени)")
    log(f"Сегментов: {len(segments)}, чисток: {cleanup_edits}, "
        f"выброшено галлюцинаций: {len(dropped)}")
    print(out_dir)


if __name__ == "__main__":
    main()
