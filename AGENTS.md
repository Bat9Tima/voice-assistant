# Voice Assistant

Локальный голосовой ассистент для Windows: Vosk (wake word) → GigaAM v3 / sherpa-onnx (STT)
→ быстрые команды, облачная LLM или локальный планировщик → Chrome через CDP/Playwright.

- Точка входа: `python -m app.main`
- Диагностика: `python -m app.doctor --record 4 --wake --stt`
- Не редактируй `%LOCALAPPDATA%\VoiceAssistant`, `logs/`, `state/` — это данные времени выполнения.
- Настройки: `config.json`.
