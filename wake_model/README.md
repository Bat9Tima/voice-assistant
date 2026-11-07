# wake_model — своя модель openWakeWord «хей луна»

Этап 0 «Окружение» и этап 1 «Синтетические позитивы» выполнены
6–7 ноября 2026. Полный план — `docs/openwakeword_plan.md`. Vosk пока
остаётся основным wake-движком; эта папка не влияет на работу ассистента.
Этап 2 (реальные записи) не начат.

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

## Этап 1: синтетические позитивы

Сгенерировано **4320** клипов «хей луна», после QC в manifest оставлено
**3351** (78 %), из них точное совпадение по ASR — 2452 (73 %):

| Источник | Голосов | Оставлено | Точно «хей луна» |
|---|---:|---:|---:|
| Silero v5_ru | 4 | 1406 | 953 |
| Silero v5_cis_base | 8 | 1434 | 1166 |
| Supertonic 3 | 1 | 238 | 94 |
| Piper | 1 | 273 | 239 |

По голосам: aidar 342, baya 359, eugene 360, xenia 345, marat/nurgul/
oksana/alfia по 200, vika 183, aigul 182, roman 180, M5 238, irina 273,
zhadyra 89. Клипов zhadyra меньше: её `plain` часто проглатывает «лу»,
поэтому добор шёл только в `slow`.

Silero детерминирован: 3351 клип — это 1067 уникальных волн, остальное
повторы SSML-настроек. Повторы не удаляются: на этапе 4 они получат
разные аугментации (темп, питч, шум, RIR). При разбиении train/val/test
учитывать хеш исходной волны, чтобы копии не попали в разные выборки.

QC-отчёт: `data/positives_tts/*/qc_report.txt`, manifest:
`data/positives_tts/manifest.csv` (локально, gitignore). Ручные исключения
после прослушивания — `data/positives_tts/exclude.txt` (по строке
`source/file`); `collect_positives.py` помечает их `kept=0`.

Итог по голосам для синтетики: **22 подтверждённых** — Silero
v5_ru aidar, baya, eugene, xenia (plain/slow/break; `fast` отбракован
после массового QC) + Silero v5_cis_base marat,
nurgul, oksana, vika, aigul, alfia, roman, zhadyra, alexandr, ekaterina,
kejilgan, kermen, saida, zara, zhazira, zinaida (plain/slow) + Piper
irina (только номинальный темп, `length_scale` ≈ 1.0; прослушивание
пройдено, 2 клипа исключены) + Supertonic M5 (все темпы). CIS-кандидаты
прослушаны: 8 из 14 прошли, 6 отбракованы — bogdan, dmitriy, igor,
karina, safarhuja, miyau. Этап 2 (реальные 3–5 голосов) по-прежнему
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

## Результаты QC

Piper, 100 клипов (`data/stage0/piper_qc/qc_report.txt`):

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

Piper Puriso, 6 клипов (`data/stage0/puriso_probe/qc_report.txt`): 0/6 точных,
1/6 с «луна»; GigaAM слышит «филна», «кшилма», «челуна». Голос
отбракован по ASR, до прослушивания не доходит.

Страницы для прослушивания (локально, не коммитятся; этап 0 — в `data/stage0/`):

- `data/stage0/piper_qc/listen_all.html` — Piper: реальные записи,
  одинаковые настройки, разброс параметров.
- `data/stage0/listen_tts.html` — Silero против Piper.
- `data/stage0/listen_supertonic.html` — Supertonic 3, все 10 голосов
  для истории.
- `data/stage0/listen_silero_cis.html` — Silero v5_cis_base, 8 голосов
  в шортлисте.
- `data/stage0/listen_control.html` — контрольный список: 14
  голосов этапа 1.
- `data/stage0/listen_cis_candidates.html` — 14 CIS-голосов, отбракованных
  только по ASR; 8 прошли, 6 отбракованы (bogdan, dmitriy, igor, karina,
  safarhuja, miyau).
- `data/positives_tts/listen_samples.html` — выборочный контроль этапа 1:
  по два клипа на голос.
- `data/positives_tts/listen_piper_stage1.html` — Piper irina, 24
  случайных клипа этапа 1.

## Структура

```
wake_model/
├── config.yaml              фраза, пути, голоса, диапазоны синтеза
├── requirements-train.txt   версии учебного окружения
├── scripts/
│   ├── setup.ps1            venv + Piper + голоса
│   ├── synthesize_piper.py  синтез Piper (irina)
│   ├── synthesize_silero.py синтез Silero v5_ru и v5_cis_base
│   ├── synthesize_supertonic.py синтез Supertonic (M5)
│   ├── check_qc.py          длительность, пик, клиппинг, GigaAM
│   ├── collect_positives.py сборка manifest после QC
│   └── make_listen_positives.py страница выборочного прослушивания
├── data/                    клипы и отчёты (gitignore)
│   ├── stage0/              архив этапа 0: пробы QC и страницы прослушивания
│   ├── positives_tts/       синтетика этапа 1 + manifest.csv
│   ├── positive/, negative/, rir/ — заготовки этапов 2–4
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

# Piper: 100 клипов и проверка (в config — только irina; по умолчанию
# пишет в data/stage0/piper_qc)
.\.venv\Scripts\python.exe wake_model\scripts\synthesize_piper.py --count 100 --clean
.\.venv\Scripts\python.exe wake_model\scripts\check_qc.py --dir wake_model\data\stage0\piper_qc

# Silero v5_ru: 12 клипов, 4 голоса × plain/slow/break
wake_model\.venv\Scripts\python.exe wake_model\scripts\synthesize_silero.py --count 12 --out wake_model\data\stage0\silero_probe_new --clean
.\.venv\Scripts\python.exe wake_model\scripts\check_qc.py --dir wake_model\data\stage0\silero_probe_new

# Silero v5_cis_base: 16 клипов, 8 голосов × plain/slow
wake_model\.venv\Scripts\python.exe wake_model\scripts\synthesize_silero.py --profile cis --count 16 --out wake_model\data\stage0\silero_cis_smoke --clean
.\.venv\Scripts\python.exe wake_model\scripts\check_qc.py --dir wake_model\data\stage0\silero_cis_smoke

# Supertonic: 3 клипа, M5 × plain/slow/fast
wake_model\.venv\Scripts\python.exe wake_model\scripts\synthesize_supertonic.py --count 3 --out wake_model\data\stage0\supertonic_probe_new --clean
.\.venv\Scripts\python.exe wake_model\scripts\check_qc.py --dir wake_model\data\stage0\supertonic_probe_new

# Этап 1: 4320 синтетических позитивов на 14 голосах.
# База — 3040 клипов:
wake_model\.venv\Scripts\python.exe wake_model\scripts\synthesize_silero.py --count 1280 --out wake_model\data\positives_tts\silero_ru --clean
wake_model\.venv\Scripts\python.exe wake_model\scripts\synthesize_silero.py --profile cis --count 1280 --out wake_model\data\positives_tts\silero_cis --clean
wake_model\.venv\Scripts\python.exe wake_model\scripts\synthesize_supertonic.py --count 240 --out wake_model\data\positives_tts\supertonic --clean
.\.venv\Scripts\python.exe wake_model\scripts\synthesize_piper.py --count 240 --out wake_model\data\positives_tts\piper --clean --no-html

# Добор — 1280 клипов (Silero 480, CIS slow 320, Piper 480):
wake_model\.venv\Scripts\python.exe wake_model\scripts\synthesize_silero.py --count 480 --seed 1 --out wake_model\data\positives_tts\silero_ru_extra --clean
wake_model\.venv\Scripts\python.exe wake_model\scripts\synthesize_silero.py --profile cis --variant slow --count 320 --seed 1 --out wake_model\data\positives_tts\silero_cis_extra --clean
.\.venv\Scripts\python.exe wake_model\scripts\synthesize_piper.py --count 480 --seed 1 --out wake_model\data\positives_tts\piper_extra --clean --no-html

# QC каждого каталога, затем единый manifest
.\.venv\Scripts\python.exe wake_model\scripts\check_qc.py --dir wake_model\data\positives_tts\silero_ru
.\.venv\Scripts\python.exe wake_model\scripts\check_qc.py --dir wake_model\data\positives_tts\silero_cis
.\.venv\Scripts\python.exe wake_model\scripts\check_qc.py --dir wake_model\data\positives_tts\supertonic
.\.venv\Scripts\python.exe wake_model\scripts\check_qc.py --dir wake_model\data\positives_tts\piper
.\.venv\Scripts\python.exe wake_model\scripts\check_qc.py --dir wake_model\data\positives_tts\silero_ru_extra
.\.venv\Scripts\python.exe wake_model\scripts\check_qc.py --dir wake_model\data\positives_tts\silero_cis_extra
.\.venv\Scripts\python.exe wake_model\scripts\check_qc.py --dir wake_model\data\positives_tts\piper_extra
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

Этап 1 выполнен: 3351 чистый синтетический «хей луна» на 14 голосах
(см. выше). Дальше:

- этап 2 — 300–500 реальных позитивов от 3–5 человек (нужно разрешение
  пользователя и сессии записи);
- этап 3 — негативы: 20–50 ч ТВ/музыки/речи/улицы и публичные корпуса;
- этап 4 — аугментации: скорость, питч, шум, RIR;
- затем датасет, обучение на Colab и сравнение с Vosk.
