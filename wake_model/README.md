# wake_model — своя модель openWakeWord «хей луна»

Этапы 0–1.5 (окружение, синтетические позитивы и добор уникальных волн)
выполнены 6–7 ноября 2026. Полный план — `docs/openwakeword_plan.md`.
Vosk пока остаётся основным wake-движком; эта папка не влияет на работу
ассистента. Инструмент этапа 2 (реальные записи) готов, сами сессии ещё
не начаты.

## Что готово

- Структура `wake_model/`: `data/`, `features/`, `checkpoints/`, `scripts/`,
  `config.yaml`.
- Отдельный учебный venv `wake_model/.venv` (Python 3.12.10), torch не
  попал в рабочий venv ассистента.
- Piper 2023.11.14-2 и четыре русских голоса `ru_RU` проверены на этапе 0.
  **irina** после повторного прослушивания исключена полностью;
  **ruslan**, **denis**, **dmitri** отбракованы раньше. В датасете Piper
  нет; скрипт и архив проб оставлены только для истории.
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
  После автоматического QC и прослушивания подтверждены 15 голосов:
  marat, nurgul, oksana, vika, aigul, alfia, roman, zhadyra, alexandr,
  ekaterina, kermen, saida, zara, zhazira, zinaida. Отбракованы albina,
  eduard, alfia2, onaoy, ramilia, gamat, sibday, bogdan, dmitriy, igor,
  karina, kejilgan, safarhuja, miyau.
- Кастомный Piper Puriso (дистилляция Supertonic M1 в ruslan) проверен
  и отбракован: 0/6 точных, GigaAM слышит «филна/кшилма».
- На этапе 0 сгенерировано 100 клипов Piper (архив), 20 клипов Silero,
  30 клипов Supertonic и 58 клипов Silero CIS; автоматическая проверка
  (длительность, пик, клиппинг, GigaAM) и страницы для прослушивания.
- `scripts/record_positives.py` — рекордер этапа 2: сессия «диктор ×
  микрофон», калибровка фона, пилот уровня, матрица 24 ячеек, плейбек и
  приём/пересдача, журнал с `--resume`, финализация в 16 кГц и QC.
  Спецификация — `docs/stage2_recording_spec.md`.

## Этап 1: синтетические позитивы

Сгенерировано **3600** клипов «хей луна», после QC в manifest оставлено
**3078** (86 %), из них точное совпадение по ASR — 2213 (72 %):

| Источник | Голосов | Оставлено | Точно «хей луна» |
|---|---:|---:|---:|
| Silero v5_ru | 4 | 1406 | 953 |
| Silero v5_cis_base | 8 | 1434 | 1166 |
| Supertonic 3 | 1 | 238 | 94 |

По голосам: aidar 342, baya 359, eugene 360, xenia 345, marat/nurgul/
oksana/alfia по 200, vika 183, aigul 182, roman 180, M5 238, zhadyra 89.
Клипов zhadyra меньше: её `plain` часто проглатывает «лу», поэтому добор
шёл только в `slow`.

Silero детерминирован: 3078 клипов этапа 1 — это 794 уникальных волны,
остальное — повторы SSML-настроек. Повторы не удаляются: на этапе 4 они получат
разные аугментации (темп, питч, шум, RIR). При разбиении train/val/test
учитывать хеш исходной волны, чтобы копии не попали в разные выборки.

QC-отчёт: `data/positives_tts/*/qc_report.txt`, manifest:
`data/positives_tts/manifest.csv` (локально, gitignore). Ручные исключения
после прослушивания — `data/positives_tts/exclude.txt` (по строке
`source/file`); `collect_positives.py` помечает их `kept=0`.

Итог по голосам для синтетики: **20 подтверждённых** — Silero
v5_ru aidar, baya, eugene, xenia (plain/slow/break; `fast` отбракован
после массового QC) + Silero v5_cis_base marat,
nurgul, oksana, vika, aigul, alfia, roman, zhadyra, alexandr, ekaterina,
kermen, saida, zara, zhazira, zinaida (plain/slow) + Supertonic M5
(все темпы). Piper исключён полностью. CIS-кандидаты прослушаны:
7 из 14 прошли, 7 отбракованы — bogdan, dmitriy, igor, karina,
kejilgan, safarhuja, miyau. Этап 2 (реальные 3–5 голосов) по-прежнему
обязателен.

### Добор уникальных волн (этап 1.5)

Silero квантует темп до 5 %, питч — до 1 %, поэтому случайный jitter
быстро упирается в байт-копии. Режим `--grid` перебирает
детерминированную сетку «темп × питч × пауза `break_ms`»: пауза даёт
новую волну почти на каждое значение. Скрипт пропускает волны, уже
существующие в `data/positives_tts`, и добирает `grid.target` на голос.

```powershell
wake_model\.venv\Scripts\python.exe wake_model\scripts\synthesize_silero.py --grid --profile silero --clean
wake_model\.venv\Scripts\python.exe wake_model\scripts\synthesize_silero.py --grid --profile cis --clean
```

Итог этапа 1.5: сгенерировано 2030 волн, после QC оставлено 1793.
Из них 1715 — новые уникальные волны; 78 оказались копиями этапа 1
(старый дедуп сравнивал хеш файла с PCM-хешем, исправлено). Вместе
с этапом 1 — **4871 клип, 3880 точных «хей луна», 2509 уникальных
волн, 20 голосов**. Дубли сохранены для аугментации этапа 4.

| Партия | Оставлено | Точно «хей луна» |
|---|---:|---:|
| Silero v5_ru, сетка | 364 | 265 |
| Silero CIS, 8 голосов | 755 | 746 |
| Silero CIS, 7 новых голосов | 674 | 656 |

У roman (66 уникальных) и zara (62) меньше: сетка часто даёт «фей/сей
луна», такие клипы отсеивает ASR-фильтр.

## Результаты QC

Piper (этап 0, исторический архив `data/stage0/piper_qc/`): проверены
четыре голоса. Irina прошла первый автоматический QC, но после повторного
прослушивания исключена полностью; ruslan, denis и dmitri отбракованы
раньше. В датасете Piper нет, его клипы удалены.

Silero, 20 клипов (`data/stage0/silero_probe/qc_report.txt`):

| Голос | Клипов | Точно «хей луна» | Есть «луна» |
|---|---:|---:|---:|
| aidar | 4 | 3 | 3 |
| baya | 4 | 3 | 3 |
| xenia | 4 | 3 | 3 |
| eugene | 4 | 0 | 1 |
| kseniya | 4 | 0 | 1 |

Первый прогон делался на всех пяти голосах и четырёх вариантах. По
прослушиванию в `config.yaml` оставлены aidar/baya/eugene/xenia;
kseniya отбракована. Позже массовый QC забраковал вариант `fast` —
на нём часто проглатывается «лу». GigaAM ошибается и на реальных
записях, поэтому это фильтр, а не приговор; окончательное решение — за
прослушиванием.

Supertonic 3, 30 клипов (`data/stage0/supertonic_probe/qc_report.txt`):

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

Silero v5_cis_base, 58 клипов (`data/stage0/silero_cis_probe/qc_report.txt`):

| Группа | Голосов | Клипов | Точно «хей луна» |
|---|---:|---:|---:|
| 2 из 2 | 4 | 8 | 8 |
| 1 из 2 | 4 | 8 | 4 |
| отбракованы | 21 | 42 | 8 |

В шортлист попали голоса, где GigaAM уверенно слышит «хей луна» хотя бы
в одном темпе. Часть голосов произносит «фей/фи» вместо «хей» — такие
отброшены автоматически; albina, eduard, alfia2, onaoy, ramilia, gamat,
sibday отбракованы после прослушивания.

Piper Puriso (этап 0): 0/6 точных, GigaAM слышит «филна/кшилма», голос
отбракован.

Страницы для прослушивания (локально, не коммитятся; этап 0 — в `data/stage0/`):

- `data/stage0/piper_qc/` — исторический архив проб Piper; в датасете
  Piper нет.
- `data/stage0/listen_supertonic.html` — Supertonic 3, все 10 голосов
  для истории.
- `data/stage0/listen_silero_cis.html` — Silero v5_cis_base, 8 голосов
  в шортлисте.
- `data/stage0/listen_control.html` — исторический контрольный список
  14 голосов этапа 1.
- `data/stage0/listen_cis_candidates.html` — 14 CIS-голосов, отбракованных
  только по ASR; 7 прошли, 7 отбракованы (bogdan, dmitriy, igor, karina,
  kejilgan, safarhuja, miyau).
- `data/positives_tts/listen_samples.html` — выборочный контроль этапа 1:
  по два клипа на голос.
- `data/positives_tts/listen_grid.html` — этап 1.5: по два клипа на голос
  из сетки (темп/питч/пауза).

## Структура

```
wake_model/
├── config.yaml              фраза, пути, голоса, диапазоны синтеза
├── requirements-train.txt   версии учебного окружения
├── scripts/
│   ├── setup.ps1            venv + инструменты TTS
│   ├── synthesize_piper.py  синтез Piper (не используется)
│   ├── synthesize_silero.py синтез Silero v5_ru и v5_cis_base
│   ├── synthesize_supertonic.py синтез Supertonic (M5)
│   ├── check_qc.py          длительность, пик, клиппинг, GigaAM
│   ├── collect_positives.py сборка manifest после QC
│   ├── make_listen_positives.py страница выборочного прослушивания
│   └── record_positives.py  запись реальных позитивов этапа 2
├── data/                    клипы и отчёты (gitignore)
│   ├── stage0/              архив этапа 0: пробы QC и страницы прослушивания
│   ├── positives_tts/       синтетика этапа 1 + manifest.csv
│   ├── positives_real/      реальные записи этапа 2 + manifest.csv
│   └── negative/, rir/      заготовки этапов 3–4
├── features/                memmap фич (gitignore)
└── checkpoints/             чекпоинты обучения (gitignore)
```

Инструменты Piper (архив) лежат в `%USERPROFILE%\Documents\VoiceAssistantTools`:
`piper.exe` падает на путях с кириллицей, а путь проекта содержит
«Проект по умолчанию». В датасете Piper не используется.

## Команды

```powershell
# Восстановить окружение с нуля (скачает ~600 МБ)
powershell -ExecutionPolicy Bypass -File wake_model\scripts\setup.ps1

# Silero v5_ru: 12 клипов, 4 голоса × plain/slow/break
wake_model\.venv\Scripts\python.exe wake_model\scripts\synthesize_silero.py --count 12 --out wake_model\data\stage0\silero_probe_new --clean
.\.venv\Scripts\python.exe wake_model\scripts\check_qc.py --dir wake_model\data\stage0\silero_probe_new

# Silero v5_cis_base: 16 клипов, 8 голосов × plain/slow
wake_model\.venv\Scripts\python.exe wake_model\scripts\synthesize_silero.py --profile cis --count 16 --out wake_model\data\stage0\silero_cis_smoke --clean
.\.venv\Scripts\python.exe wake_model\scripts\check_qc.py --dir wake_model\data\stage0\silero_cis_smoke

# Supertonic: 3 клипа, M5 × plain/slow/fast
wake_model\.venv\Scripts\python.exe wake_model\scripts\synthesize_supertonic.py --count 3 --out wake_model\data\stage0\supertonic_probe_new --clean
.\.venv\Scripts\python.exe wake_model\scripts\check_qc.py --dir wake_model\data\stage0\supertonic_probe_new

# Этап 1: 3600 синтетических позитивов на 13 голосах.
# База — 2800 клипов:
wake_model\.venv\Scripts\python.exe wake_model\scripts\synthesize_silero.py --count 1280 --out wake_model\data\positives_tts\silero_ru --clean
wake_model\.venv\Scripts\python.exe wake_model\scripts\synthesize_silero.py --profile cis --count 1280 --out wake_model\data\positives_tts\silero_cis --clean
wake_model\.venv\Scripts\python.exe wake_model\scripts\synthesize_supertonic.py --count 240 --out wake_model\data\positives_tts\supertonic --clean

# Добор — 800 клипов (Silero 480, CIS slow 320):
wake_model\.venv\Scripts\python.exe wake_model\scripts\synthesize_silero.py --count 480 --seed 1 --out wake_model\data\positives_tts\silero_ru_extra --clean
wake_model\.venv\Scripts\python.exe wake_model\scripts\synthesize_silero.py --profile cis --variant slow --count 320 --seed 1 --out wake_model\data\positives_tts\silero_cis_extra --clean

# QC каждого каталога, затем единый manifest
.\.venv\Scripts\python.exe wake_model\scripts\check_qc.py --dir wake_model\data\positives_tts\silero_ru
.\.venv\Scripts\python.exe wake_model\scripts\check_qc.py --dir wake_model\data\positives_tts\silero_cis
.\.venv\Scripts\python.exe wake_model\scripts\check_qc.py --dir wake_model\data\positives_tts\supertonic
.\.venv\Scripts\python.exe wake_model\scripts\check_qc.py --dir wake_model\data\positives_tts\silero_ru_extra
.\.venv\Scripts\python.exe wake_model\scripts\check_qc.py --dir wake_model\data\positives_tts\silero_cis_extra
.\.venv\Scripts\python.exe wake_model\scripts\collect_positives.py
.\.venv\Scripts\python.exe wake_model\scripts\make_listen_positives.py

# Этап 2: реальные записи (основной venv: sounddevice + GigaAM-QC)
.\.venv\Scripts\python.exe wake_model\scripts\record_positives.py --list-devices
.\.venv\Scripts\python.exe wake_model\scripts\record_positives.py --setup --speaker anna --mic laptop
.\.venv\Scripts\python.exe wake_model\scripts\record_positives.py --setup --speaker anna --mic studio --device "USB"
.\.venv\Scripts\python.exe wake_model\scripts\record_positives.py --resume
.\.venv\Scripts\python.exe wake_model\scripts\record_positives.py --finalize
.\.venv\Scripts\python.exe wake_model\scripts\record_positives.py --review
```

## Версии

- Python 3.12.10
- torch 2.15.0+cpu (CUDA нет, обучение планируется на Colab)
- openwakeword 0.6.0, torchinfo 1.8.0, torchmetrics 1.9.0
- numpy 2.5.4, scipy 1.18.1, onnxruntime 1.31.1
- Silero v5_ru и v5_cis_base (48 000 Гц, приводим к 16 000)
- Supertonic 3, ONNX (44 100 Гц, приводим к 16 000)

## Лицензии

Silero TTS — CC BY-NC-SA 4.0 (некоммерческое использование). Supertonic 3 —
OpenRAIL-M (использование с ограничениями, для личного проекта годится).
При распространении модели лицензии синтеза нужно проверить отдельно.

## Дальше

Этапы 1 и 1.5 выполнены: 4871 синтетический «хей луна» (2509 уникальных
волн) на 20 голосах. Дальше:

- этап 2 — пилот 20 дублей на обоих микрофонах, затем 3–5 дикторов по
  60–100 дублей на микрофон (инструмент готов, нужны сессии записи);
- этап 3 — негативы: 20–50 ч ТВ/музыки/речи/улицы и публичные корпуса;
- этап 4 — аугментации: скорость, питч, шум, RIR;
- затем датасет, обучение на Colab и сравнение с Vosk.
