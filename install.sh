#!/bin/bash
# Установка скилла transcribe в ~/.claude/skills/ без плагин-системы.
# Standalone install into ~/.claude/skills/ (alternative to /plugin install).
#
#   curl -fsSL https://raw.githubusercontent.com/Rainary/claude-transcribe/main/install.sh | bash
#
# или из клона репозитория: bash install.sh

set -euo pipefail

DEST="$HOME/.claude/skills/transcribe"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || true)"

mkdir -p "$DEST/references"

if [ -n "$SRC" ] && [ -f "$SRC/SKILL.md" ]; then
    echo "==> Копирую скилл из $SRC в $DEST"
    cp "$SRC/SKILL.md" "$SRC/transcribe.py" "$SRC/diarize.py" \
       "$SRC/repair_segments.py" "$SRC/milchin.py" "$SRC/setup.sh" "$DEST/"
    cp "$SRC/references/memo-style.md" "$DEST/references/"
else
    echo "==> Скачиваю скилл с GitHub в $DEST"
    BASE="https://raw.githubusercontent.com/Rainary/claude-transcribe/main"
    for f in SKILL.md transcribe.py diarize.py repair_segments.py milchin.py setup.sh; do
        curl -fsSL -o "$DEST/$f" "$BASE/$f"
    done
    curl -fsSL -o "$DEST/references/memo-style.md" "$BASE/references/memo-style.md"
fi

chmod +x "$DEST/setup.sh"
echo "==> Файлы на месте. Запускаю доктора окружения (setup.sh)…"
echo
bash "$DEST/setup.sh"
