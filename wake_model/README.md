# wake_model — своя модель openWakeWord «хей луна»

Этап 0 «Окружение» выполнен 6 ноября 2026. Полный план —
`docs/openwakeword_plan.md`. Vosk пока остаётся основным wake-движком;
эта папка не влияет на работу ассистента.

## Что готово

- Структура `wake_model/`: `data/`, `features/`, `checkpoints/`, `scripts/`,
  `config.yaml`.
- Отдельный учебный venv `wake_model/.venv` (Python 3.12.10), torch не
  попал в рабочий venv ассистента.
- Piper 2023.11.14-2 и четыре русских голоса `ru_RU`; после прослушивания
  годятся **irina** и **ruslan**, голоса **denis/dmitri** отбракованы —
  первое слово звучит как «кей/цей луна».
- Silero TTS v5_ru: пять голосов, SSML для темпа и паузы. По QC и
  прослушиванию годятся **aidar, baya, xenia**; **eugene/kseniya** дают
  «силна/килуна», а вариант `fast` слишком быстрый.
- Сгенерировано 100 клипов Piper и 20 клипов Silero, автоматическая
  проверка (длительность, пик, клиппинг, GigaAM) и страницы для
  прослушивания.

Итог по голосам для синтетики этапа 1: **5 голосов** — Silero aidar,
baya, xenia + Piper irina, ruslan. Этого мало для полностью
самостоятельной модели, поэтому этап 2 (реальные 3–5 голосов) остаётся
обязательным.

## Результаты QC

Piper, 100 клипов (`data/piper_qc/qc_report.txt`):

| Голос | Клипов | Точно «хей луна» | Есть «луна» |
|---|---:|---:|---:|
| irina | 25 | 11 | 18 |
| ruslan | 25 | 0 | 9 |
| denis | 25 | 1 | 5 |
| dmitri | 25 | 0 | 8 |

Первый прогон делался на всех четырёх голосах; в `config.yaml` оставлены
только irina/ruslan.

Silero, 20 клипов (`data/silero_probe/qc_report.txt`):

| Голос | Клипов | Точно «хей луна» | Есть «луна» |
|---|---:|---:|---:|
| aidar | 4 | 3 | 3 |
| baya | 4 | 3 | 3 |
| xenia | 4 | 3 | 3 |
| eugene | 4 | 0 | 1 |
| kseniya | 4 | 0 | 1 |

Первый прогон делался на всех пяти голосах и четырёх вариантах; в
`config.yaml` оставлены только прошедшие QC: aidar/baya/xenia и
plain/slow/break. По вариантам: `plain`, `slow`, `break` — по 3/5 точных,
`fast` — 0/5. GigaAM ошибается и на реальных записях, поэтому это
фильтр, а не приговор; окончательное решение — за прослушиванием.

Страницы для прослушивания (локально, не коммитятся):

- `data/piper_qc/listen_all.html` — Piper: реальные записи, одинаковые
  настройки, разброс параметров.
- `data/listen_tts.html` — Silero против Piper.

## Структура

```
wake_model/
├── config.yaml              фраза, пути, голоса, диапазоны синтеза
├── requirements-train.txt   версии учебного окружения
├── scripts/
│   ├── setup.ps1            venv + Piper + голоса
│   ├── synthesize.py        синтез Piper (irina/ruslan)
│   ├── synthesize_silero.py синтез Silero (5 голосов, SSML)
│   └── check_qc.py          длительность, пик, клиппинг, GigaAM
├── data/                    клипы и отчёты (gitignore)
├── features/                memmap фич (gitignore)
└── checkpoints/             чекпоинты обучения (gitignore)
```

Piper и голоса лежат в `%USERPROFILE%\Documents\VoiceAssistantTools`:
`piper.exe` падает на путях с кириллицей, а путь проекта содержит
«Проект по умолчанию».

## Команды

```powershell
# Восстановить окружение с нуля (скачает ~600 МБ)
powershell -ExecutionPolicy Bypass -File wake_model\scripts\setup.ps1

# Piper: 100 клипов и проверка (в config — только irina/ruslan)
.\.venv\Scripts\python.exe wake_model\scripts\synthesize.py --count 100 --clean
.\.venv\Scripts\python.exe wake_model\scripts\check_qc.py --dir wake_model\data\piper_qc

# Silero: 9 клипов текущими голосами и проверка
wake_model\.venv\Scripts\python.exe wake_model\scripts\synthesize_silero.py --count 9 --out wake_model\data\silero_probe_new
.\.venv\Scripts\python.exe wake_model\scripts\check_qc.py --dir wake_model\data\silero_probe_new
```

## Версии

- Python 3.12.10
- torch 2.15.0+cpu (CUDA нет, обучение планируется на Colab)
- openwakeword 0.6.0, torchinfo 1.8.0, torchmetrics 1.9.0
- numpy 2.5.4, scipy 1.18.1, onnxruntime 1.31.1
- Piper 2023.11.14-2, голоса `ru_RU` medium (22 050 Гц)
- Silero v5_ru (48 000 Гц, приводим к 16 000)

## Лицензии

Silero TTS — CC BY-NC-SA 4.0 (некоммерческое использование). Голоса
Piper: irina — RHVoice (лицензия не указана), ruslan — CC BY-NC-SA 4.0,
denis/dmitri — CC0. Для личного проекта годится; при распространении
модели лицензии синтеза нужно проверить отдельно.

## Дальше

Этап 1: 3–10 тыс. синтетических «хей луна» на Silero (aidar/baya/xenia)
с вариантами `plain`, `slow`, `break`; `fast` не использовать. Piper
irina/ruslan — небольшая добавка для разнообразия. Затем аугментации
(скорость, питч, шум, RIR) и реальные записи этапа 2.
