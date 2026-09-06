#!/bin/bash
# Доктор-установщик скилла transcribe. Идемпотентный: чинит только то,
# чего не хватает, повторный запуск безопасен.
#
# Что делает сам:
#   - проверяет ffmpeg (ставит через brew, если brew есть);
#   - скачивает установщик Python 3.11 с python.org и открывает его
#     (единственный ручной шаг — кликнуть установку, нужен пароль админа);
#   - создаёт/пересобирает .venv (рядом со скиллом либо переиспользует
#     существующий в корне проекта) на Python 3.11;
#   - доставляет пакеты (mlx-whisper, silero-vad, senko);
#   - прогоняет финальную проверку окружения.
#
# Python 3.11 обязателен: колесо senko (диаризация) для macOS собрано под
# cp311. Ставится с python.org в /Library/Frameworks — системный путь,
# одинаковый для всех профилей macOS.
#
# Переменные для тестов: TRANSCRIBE_PY (путь к питону),
# TRANSCRIBE_SETUP_NO_OPEN=1 (не открывать установщик).

set -euo pipefail

PY="${TRANSCRIBE_PY:-/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11}"
PKG_URL="https://www.python.org/ftp/python/3.11.9/python-3.11.9-macos11.pkg"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# venv: явный TRANSCRIBE_VENV → уже существующий (рядом со скиллом или в корне
# проекта при структуре <root>/.claude/skills/transcribe/) → создать рядом
# со скиллом.
VENV="${TRANSCRIBE_VENV:-}"
if [ -z "$VENV" ]; then
    for cand in "$SCRIPT_DIR/.venv" "$SCRIPT_DIR/../../../.venv"; do
        if [ -x "$cand/bin/python" ]; then
            VENV="$(cd "$cand" && pwd)"
            break
        fi
    done
fi
[ -z "$VENV" ] && VENV="$SCRIPT_DIR/.venv"

# Куда скачивать установщик Python: temp/ проекта, если есть, иначе кэш
DL_DIR="$SCRIPT_DIR/../../../temp"
[ -d "$DL_DIR" ] || DL_DIR="$HOME/Library/Caches/claude-transcribe"

step() { echo "==> $1"; }

# --- 1. ffmpeg ---------------------------------------------------------------

if command -v ffmpeg >/dev/null 2>&1 \
        || [ -x /opt/homebrew/bin/ffmpeg ] || [ -x /usr/local/bin/ffmpeg ]; then
    step "ffmpeg: есть"
elif command -v brew >/dev/null 2>&1; then
    step "ffmpeg: нет — ставлю через brew (может занять несколько минут)"
    brew install ffmpeg
else
    echo "ffmpeg не найден, а Homebrew нет. Установи вручную: https://brew.sh,"
    echo "затем: brew install ffmpeg — и запусти setup.sh снова."
    exit 1
fi

# --- 2. Python 3.11 ----------------------------------------------------------

if [ ! -x "$PY" ]; then
    step "Python 3.11 не найден — скачиваю официальный установщик python.org"
    PKG="$DL_DIR/$(basename "$PKG_URL")"
    mkdir -p "$DL_DIR"
    if [ ! -f "$PKG" ]; then
        curl -fL --progress-bar -o "$PKG" "$PKG_URL"
    fi
    echo
    echo "Установщик скачан: $PKG"
    if [ "${TRANSCRIBE_SETUP_NO_OPEN:-0}" != "1" ]; then
        open "$PKG"
        echo "Открыл установщик — пройди установку (нужен пароль администратора),"
    else
        echo "Открой его и пройди установку (нужен пароль администратора),"
    fi
    echo "затем запусти setup.sh ещё раз: он продолжит с этого места."
    echo "Текущий venv не тронут — скилл продолжает работать как раньше."
    exit 2
fi
step "Python 3.11: есть ($("$PY" -V 2>&1))"

# --- 3. venv -----------------------------------------------------------------

TARGET_VER="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
CURRENT_VER=""
if [ -x "$VENV/bin/python" ]; then
    CURRENT_VER="$("$VENV/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
fi

if [ "$CURRENT_VER" = "$TARGET_VER" ]; then
    step "venv ($VENV): уже на Python $CURRENT_VER — пересборка не нужна"
else
    if [ -n "$CURRENT_VER" ]; then
        step "venv ($VENV): Python $CURRENT_VER → пересобираю на $TARGET_VER"
    else
        step "venv ($VENV): нет — создаю на Python $TARGET_VER"
    fi
    rm -rf "$VENV"
    "$PY" -m venv "$VENV"
fi

# --- 4. Пакеты (идемпотентно: pip сам пропустит установленное) ---------------

step "Пакеты: mlx-whisper, silero-vad"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q mlx-whisper silero-vad

step "Пакет: senko (диаризация)"
if ! "$VENV/bin/pip" install -q senko; then
    echo "ВНИМАНИЕ: senko не установился (нужен Python 3.10–3.13, macOS 14+)."
    echo "Транскрипция будет работать, диаризация — нет."
fi

# --- 5. Финальная проверка ---------------------------------------------------

echo
"$VENV/bin/python" "$SCRIPT_DIR/transcribe.py" --check
echo
echo "Готово. Модель turbo (~1.6 ГБ) скачается при первой транскрипции,"
echo "если ещё не скачана."
