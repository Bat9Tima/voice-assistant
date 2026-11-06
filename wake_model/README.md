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
- Silero TTS v5_ru: пять голосов, SSML для темпа и паузы. Прослушивание:
  годятся **aidar, baya, eugene, xenia**; **kseniya** звучит как «кей» на
  всех темпах и отбракована. GigaAM ошибочно забраковала eugene — верим
  ушам, а не ASR.
- Supertonic 3: современный локальный ONNX-TTS на 31 язык, 10 голосов
  (F1–F5, M1–M5), русский поддерживается. Пробные 30 клипов: GigaAM
  слышит в основном слитное «хейлуна», точное совпадение 6/30; это
  нормальное слияние слов, а не ошибка произношения. По прослушиванию
  прошли **M1, M2, M5**.
- Silero v5_cis_base: 29 голосов `ru_*` (носители из СНГ, у части акцент).
  После автоматического QC и прослушивания в шортлисте 9 голосов:
  marat, nurgul, oksana, vika, aigul, alfia, roman, sibday, zhadyra.
  albina, eduard, alfia2, onaoy, ramilia, gamat отбракованы на слух.
- Кастомный Piper Puriso (дистилляция Supertonic M1 в ruslan): 6 клипов,
  GigaAM не слышит «хей луна» ни в одном (0/6, варианты «филна/кшилма»),
  поэтому голос отбракован. Официальных новых русских голосов Piper нет.
- Сгенерировано 100 клипов Piper, 20 клипов Silero, 30 клипов Supertonic
  и 58 клипов Silero CIS; автоматическая проверка (длительность, пик,
  клиппинг, GigaAM) и страницы для прослушивания.

Итог по голосам для синтетики этапа 1: **9 подтверждённых** — Silero
aidar, baya, eugene, xenia (все темпы) + Piper irina, ruslan (только
номинальный темп, `length_scale` ≈ 1.0) + Supertonic M1, M2, M5. Сверху —
9 голосов Silero CIS в статусе прослушивания. Даже с ними этап 2
(реальные 3–5 голосов) остаётся обязательным.

## Результаты QC

Piper, 100 клипов (`data/piper_qc/qc_report.txt`):

| Голос | Клипов | Точно «хей луна» | Есть «луна» |
|---|---:|---:|---:|
| irina | 25 | 11 | 18 |
| ruslan | 25 | 0 | 9 |
| denis | 25 | 1 | 5 |
| dmitri | 25 | 0 | 8 |

Первый прогон делался на всех четырёх голосах; в `config.yaml` оставлены
только irina/ruslan. Piper ломает первое слово при изменении темпа,
поэтому для него оставлен только `length_scale` ≈ 1.0.

Silero, 20 клипов (`data/silero_probe/qc_report.txt`):

| Голос | Клипов | Точно «хей луна» | Есть «луна» |
|---|---:|---:|---:|
| aidar | 4 | 3 | 3 |
| baya | 4 | 3 | 3 |
| xenia | 4 | 3 | 3 |
| eugene | 4 | 0 | 1 |
| kseniya | 4 | 0 | 1 |

Первый прогон делался на всех пяти голосах и четырёх вариантах. По
прослушиванию в `config.yaml` оставлены aidar/baya/eugene/xenia и все
четыре варианта; kseniya отбракована. GigaAM ошибается и на реальных
записях, поэтому это фильтр, а не приговор; окончательное решение — за
прослушиванием.

Supertonic 3, 30 клипов (`data/supertonic_probe/qc_report.txt`):

| Голос | Клипов | Точно «хей луна» | Есть «луна» |
|---|---:|---:|---:|
| F1 | 3 | 0 | 0 |
| F2 | 3 | 0 | 0 |
| F3 | 3 | 0 | 1 |
| F4 | 3 | 0 | 0 |
| F5 | 3 | 1 | 1 |
| M1 | 3 | 0 | 0 |
| M2 | 3 | 1 | 1 |
| M3 | 3 | 3 | 3 |
| M4 | 3 | 0 | 0 |
| M5 | 3 | 1 | 1 |

18 из 30 клипов GigaAM записала как «хейлуна» — слова слились, отдельной
ошибки «кей» нет. По вариантам точных совпадений: plain 4/10, slow 1/10,
fast 1/10; медленный и быстрый темп стоит сузить после прослушивания.
По прослушиванию выбраны **M1, M2, M5**.

Silero v5_cis_base, 58 клипов (`data/silero_cis_probe/qc_report.txt`):

| Группа | Голосов | Клипов | Точно «хей луна» |
|---|---:|---:|---:|
| 2 из 2 | 4 | 8 | 8 |
| 1 из 2 | 5 | 10 | 5 |
| отбракованы | 20 | 40 | 7 |

В шортлист попали голоса, где GigaAM уверенно слышит «хей луна» хотя бы
в одном темпе. Часть голосов произносит «фей/фи» вместо «хей» — такие
отброшены автоматически; albina, eduard, alfia2, onaoy, ramilia, gamat
отбракованы после прослушивания.

Piper Puriso, 6 клипов (`data/puriso_probe/qc_report.txt`): 0/6 точных,
1/6 с «луна»; GigaAM слышит «филна», «кшилма», «челуна». Голос
отбракован по ASR, до прослушивания не доходит.

Страницы для прослушивания (локально, не коммитятся):

- `data/piper_qc/listen_all.html` — Piper: реальные записи, одинаковые
  настройки, разброс параметров.
- `data/listen_tts.html` — Silero против Piper.
- `data/listen_supertonic.html` — Supertonic 3, 10 голосов, три темпа.
- `data/listen_silero_cis.html` — Silero v5_cis_base, 9 голосов
  в шортлисте.

## Структура

```
wake_model/
├── config.yaml              фраза, пути, голоса, диапазоны синтеза
├── requirements-train.txt   версии учебного окружения
├── scripts/
│   ├── setup.ps1            venv + Piper + голоса
│   ├── synthesize.py        синтез Piper (irina/ruslan)
│   ├── synthesize_silero.py синтез Silero (4 голоса, SSML)
│   ├── synthesize_supertonic.py синтез Supertonic (M1/M2/M5)
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

# Supertonic: 9 клипов, M1/M2/M5 × plain/slow/fast
wake_model\.venv\Scripts\python.exe wake_model\scripts\synthesize_supertonic.py --count 9 --out wake_model\data\supertonic_probe_new --clean
.\.venv\Scripts\python.exe wake_model\scripts\check_qc.py --dir wake_model\data\supertonic_probe_new
```

## Версии

- Python 3.12.10
- torch 2.15.0+cpu (CUDA нет, обучение планируется на Colab)
- openwakeword 0.6.0, torchinfo 1.8.0, torchmetrics 1.9.0
- numpy 2.5.4, scipy 1.18.1, onnxruntime 1.31.1
- Piper 2023.11.14-2, голоса `ru_RU` medium (22 050 Гц)
- Silero v5_ru (48 000 Гц, приводим к 16 000)
- Supertonic 3, ONNX (44 100 Гц, приводим к 16 000)

## Лицензии

Silero TTS — CC BY-NC-SA 4.0 (некоммерческое использование). Supertonic 3 —
OpenRAIL-M (использование с ограничениями, для личного проекта годится).
Голоса Piper: irina — RHVoice (лицензия не указана), ruslan — CC BY-NC-SA
4.0, denis/dmitri — CC0. Для личного проекта годится; при распространении
модели лицензии синтеза нужно проверить отдельно.

## Дальше

Этап 1: 3–10 тыс. синтетических «хей луна» на подтверждённых голосах
Silero (aidar/baya/eugene/xenia) с вариантами `plain`, `slow`, `fast`,
`break`, Supertonic (M1/M2/M5) и Piper irina/ruslan (только
`length_scale` ≈ 1.0). После прослушивания к ним могут добавиться голоса
Silero v5_cis_base. Затем аугментации (скорость, питч, шум, RIR) и
реальные записи этапа 2.
