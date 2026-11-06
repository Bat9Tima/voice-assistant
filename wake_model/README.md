# wake_model — своя модель openWakeWord «хей луна»

Этап 0 «Окружение» выполнен 6 ноября 2026. Полный план —
`docs/openwakeword_plan.md`. Vosk пока остаётся основным wake-движком;
эта папка не влияет на работу ассистента.

## Что готово

- Структура `wake_model/`: `data/`, `features/`, `checkpoints/`, `scripts/`,
  `config.yaml`.
- Отдельный учебный venv `wake_model/.venv` (Python 3.12.10), torch не
  попал в рабочий venv ассистента.
- Piper 2023.11.14-2 и четыре русских голоса `ru_RU`. После прослушивания
  годится только **irina**; **ruslan**, **denis**, **dmitri** отбракованы —
  первое слово звучит как «кей/цей» или не проходит контроль.
- Silero TTS v5_ru: пять голосов, SSML для темпа и паузы. Прослушивание:
  годятся **aidar, baya, eugene, xenia**; **kseniya** звучит как «кей» на
  всех темпах и отбракована. GigaAM ошибочно забраковала eugene — верим
  ушам, а не ASR.
- Supertonic 3: современный локальный ONNX-TTS на 31 язык, 10 голосов
  (F1–F5, M1–M5), русский поддерживается. Пробные 30 клипов: GigaAM
  слышит в основном слитное «хейлуна», точное совпадение 6/30; это
  нормальное слияние слов, а не ошибка произношения. По контрольному
  прослушиванию прошёл только **M5** (все темпы); M1, M2 и остальные
  голоса отбракованы.
- Silero v5_cis_base: 29 голосов `ru_*` (носители из СНГ, у части акцент).
  После автоматического QC и контрольного прослушивания подтверждены
  8 голосов: marat, nurgul, oksana, vika, aigul, alfia, roman, zhadyra.
  albina, eduard, alfia2, onaoy, ramilia, gamat, sibday отбракованы.
- Кастомный Piper Puriso (дистилляция Supertonic M1 в ruslan): 6 клипов,
  GigaAM не слышит «хей луна» ни в одном (0/6, варианты «филна/кшилма»),
  поэтому голос отбракован. Официальных новых русских голосов Piper нет.
- Сгенерировано 100 клипов Piper, 20 клипов Silero, 30 клипов Supertonic
  и 58 клипов Silero CIS; автоматическая проверка (длительность, пик,
  клиппинг, GigaAM) и страницы для прослушивания.

Итог по голосам для синтетики этапа 1: **14 подтверждённых** — Silero
v5_ru aidar, baya, eugene, xenia (все темпы) + Silero v5_cis_base marat,
nurgul, oksana, vika, aigul, alfia, roman, zhadyra (plain/slow) + Piper
irina (только номинальный темп, `length_scale` ≈ 1.0) + Supertonic M5
(все темпы). Непроверенных кандидатов не осталось. Этап 2 (реальные
3–5 голосов) по-прежнему обязателен.

## Результаты QC

Piper, 100 клипов (`data/piper_qc/qc_report.txt`):

| Голос | Клипов | Точно «хей луна» | Есть «луна» |
|---|---:|---:|---:|
| irina | 25 | 11 | 18 |
| ruslan | 25 | 0 | 9 |
| denis | 25 | 1 | 5 |
| dmitri | 25 | 0 | 8 |

Первый прогон делался на всех четырёх голосах; в `config.yaml` оставлена
только irina. Ruslan прошёл первый QC, но не прошёл контрольное
прослушивание. Piper ломает первое слово при изменении темпа, поэтому
для него оставлен только `length_scale` ≈ 1.0.

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
fast 1/10. По контрольному прослушиванию оставлен только M5 (все темпы);
M1, M2 и остальные голоса отбракованы.

Silero v5_cis_base, 58 клипов (`data/silero_cis_probe/qc_report.txt`):

| Группа | Голосов | Клипов | Точно «хей луна» |
|---|---:|---:|---:|
| 2 из 2 | 4 | 8 | 8 |
| 1 из 2 | 4 | 8 | 4 |
| отбракованы | 21 | 42 | 8 |

В шортлист попали голоса, где GigaAM уверенно слышит «хей луна» хотя бы
в одном темпе. Часть голосов произносит «фей/фи» вместо «хей» — такие
отброшены автоматически; albina, eduard, alfia2, onaoy, ramilia, gamat,
sibday отбракованы после прослушивания.

Piper Puriso, 6 клипов (`data/puriso_probe/qc_report.txt`): 0/6 точных,
1/6 с «луна»; GigaAM слышит «филна», «кшилма», «челуна». Голос
отбракован по ASR, до прослушивания не доходит.

Страницы для прослушивания (локально, не коммитятся):

- `data/piper_qc/listen_all.html` — Piper: реальные записи, одинаковые
  настройки, разброс параметров.
- `data/listen_tts.html` — Silero против Piper.
- `data/listen_supertonic.html` — Supertonic 3, все 10 голосов для истории.
- `data/listen_silero_cis.html` — Silero v5_cis_base, 8 голосов
  в шортлисте.
- `data/listen_control.html` — контрольный список: 14 подтверждённых
  голосов, кандидатов не осталось.

## Структура

```
wake_model/
├── config.yaml              фраза, пути, голоса, диапазоны синтеза
├── requirements-train.txt   версии учебного окружения
├── scripts/
│   ├── setup.ps1            venv + Piper + голоса
│   ├── synthesize.py        синтез Piper (irina)
│   ├── synthesize_silero.py синтез Silero v5_ru и v5_cis_base
│   ├── synthesize_supertonic.py синтез Supertonic (M5)
│   ├── check_qc.py          длительность, пик, клиппинг, GigaAM
│   ├── collect_positives.py сборка manifest после QC
│   └── make_listen_positives.py страница выборочного прослушивания
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

# Piper: 100 клипов и проверка (в config — только irina)
.\.venv\Scripts\python.exe wake_model\scripts\synthesize.py --count 100 --clean
.\.venv\Scripts\python.exe wake_model\scripts\check_qc.py --dir wake_model\data\piper_qc

# Silero v5_ru: 16 клипов, 4 голоса × plain/slow/fast/break
wake_model\.venv\Scripts\python.exe wake_model\scripts\synthesize_silero.py --count 16 --out wake_model\data\silero_probe_new --clean
.\.venv\Scripts\python.exe wake_model\scripts\check_qc.py --dir wake_model\data\silero_probe_new

# Silero v5_cis_base: 16 клипов, 8 голосов × plain/slow
wake_model\.venv\Scripts\python.exe wake_model\scripts\synthesize_silero.py --profile cis --count 16 --out wake_model\data\silero_cis_smoke --clean
.\.venv\Scripts\python.exe wake_model\scripts\check_qc.py --dir wake_model\data\silero_cis_smoke

# Supertonic: 3 клипа, M5 × plain/slow/fast
wake_model\.venv\Scripts\python.exe wake_model\scripts\synthesize_supertonic.py --count 3 --out wake_model\data\supertonic_probe_new --clean
.\.venv\Scripts\python.exe wake_model\scripts\check_qc.py --dir wake_model\data\supertonic_probe_new

# Этап 1: около 3040 синтетических позитивов на 14 голосах
wake_model\.venv\Scripts\python.exe wake_model\scripts\synthesize_silero.py --count 1280 --out wake_model\data\positives_tts\silero_ru --clean
wake_model\.venv\Scripts\python.exe wake_model\scripts\synthesize_silero.py --profile cis --count 1280 --out wake_model\data\positives_tts\silero_cis --clean
wake_model\.venv\Scripts\python.exe wake_model\scripts\synthesize_supertonic.py --count 240 --out wake_model\data\positives_tts\supertonic --clean
.\.venv\Scripts\python.exe wake_model\scripts\synthesize.py --count 240 --out wake_model\data\positives_tts\piper --clean --no-html

# QC каждого каталога, затем единый manifest
.\.venv\Scripts\python.exe wake_model\scripts\check_qc.py --dir wake_model\data\positives_tts\silero_ru
.\.venv\Scripts\python.exe wake_model\scripts\check_qc.py --dir wake_model\data\positives_tts\silero_cis
.\.venv\Scripts\python.exe wake_model\scripts\check_qc.py --dir wake_model\data\positives_tts\supertonic
.\.venv\Scripts\python.exe wake_model\scripts\check_qc.py --dir wake_model\data\positives_tts\piper
.\.venv\Scripts\python.exe wake_model\scripts\collect_positives.py
.\.venv\Scripts\python.exe wake_model\scripts\make_listen_positives.py
```

## Версии

- Python 3.12.10
- torch 2.15.0+cpu (CUDA нет, обучение планируется на Colab)
- openwakeword 0.6.0, torchinfo 1.8.0, torchmetrics 1.9.0
- numpy 2.5.4, scipy 1.18.1, onnxruntime 1.31.1
- Piper 2023.11.14-2, голоса `ru_RU` medium (22 050 Гц)
- Silero v5_ru и v5_cis_base (48 000 Гц, приводим к 16 000)
- Supertonic 3, ONNX (44 100 Гц, приводим к 16 000)

## Лицензии

Silero TTS — CC BY-NC-SA 4.0 (некоммерческое использование). Supertonic 3 —
OpenRAIL-M (использование с ограничениями, для личного проекта годится).
Голоса Piper: irina — RHVoice (лицензия не указана), ruslan — CC BY-NC-SA
4.0, denis/dmitri — CC0. Для личного проекта годится; при распространении
модели лицензии синтеза нужно проверить отдельно.

## Дальше

Этап 1: 3–10 тыс. синтетических «хей луна» на 14 подтверждённых голосах:
Silero v5_ru (aidar/baya/eugene/xenia; `plain`, `slow`, `fast`, `break`),
Silero v5_cis_base (8 голосов; `plain`, `slow`), Supertonic M5 (все
темпы) и Piper irina (только `length_scale` ≈ 1.0). Затем аугментации
(скорость, питч, шум, RIR) и реальные записи этапа 2.
