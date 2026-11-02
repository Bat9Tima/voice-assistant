# Voice Assistant

Локальный голосовой ассистент для Windows: Vosk (wake word) → GigaAM v3 / sherpa-onnx (STT)
→ быстрые команды или агент OpenCode → Chrome через CDP/Playwright.

- Точка входа: `python -m app.main`
- Диагностика: `python -m app.doctor --record 4 --wake --stt`
- Управление браузером для агента: `bin/browser.cmd <команда>` (см. `.opencode/agents/voice.md`)
- Не редактируй `%LOCALAPPDATA%\VoiceAssistant`, `logs/`, `state/` — это данные времени выполнения.
- Настройки: `config.json`.
