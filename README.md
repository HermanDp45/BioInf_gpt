# BioInf_gpt - GPT для генерации белковых последовательностей

Адаптация nanoGPT для обучения и генерации белковых последовательностей на основе датасета

---

## Оглавление
- [Описание проекта](#описание-проекта)
- [Установка](#установка)
- [Быстрый старт](#быстрый-старт)
- [Подробности описания](#подробная-инструкция)
- [Конфигурация](#конфигурация)
- [Примеры использования](#примеры-использования)
- [Архитектура](#архитектура)

---

## Описание проекта

Проект реализует GPT-модель для генерации белковых последовательностей

**Основные возможности:**
- **Два типа последовательностей**: `sequence` (выровненные - с "-") и `init_seq` 
- **Условная генерация**: добавление префиксов `<class>` и `<type>` с настраиваемой вероятностью
- **Стриминг данных**: работа с датасетом без полной загрузки
- **Кастомный вокобуляр**: 20 аминокислот + специальные токены - на классы, типы и обозначения начала, конца последовательности, неизвестный символ, токен для выравнивания данных в батче (class, type, eos, sos, unk, pad)

---

## Установка

### Требования
- Python 3.8+

### Установка зависимостей

```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

**Зависимости:**
- `torch` - PyTorch для обучения модели
- `numpy` - работа с массивами
- `transformers` - загрузка GPT-2 checkpoints
- `datasets` - Hugging Face Datasets для загрузки датасета
- `tiktoken` - токенизация (не используется для белков, но была определена для nanoGPT)
- `wandb` - логирование
- `tqdm` - прогресс-бары

### Клонирование репозитория

```bash
git clone https://github.com/HermanDp45/BioInf_gpt.git
cd BioInf_gpt
```

---

## Быстрый старт

### Шаг 1: Подготовка метаданных

```bash
python data/protein/prepare.py
```

**Что происходит:**
- Создаётся словарь: 20 аминокислот + специальные токены (на все class, на все type, pad, eos, unk, sos, "-")
- Сохраняется в `data/protein/meta.pkl`


### Шаг 2: Обучение модели

```bash
python train.py config/train_protein.py \
    --out_dir=out-protein-test \
    --device=cpu \
    --compile=False \
    --batch_size=4 \
    --block_size=64 \
    --n_layer=2 \
    --n_head=2 \
    --n_embd=64 \
    --max_iters=100 \
    --eval_interval=20 \
    --log_interval=5 \
    --data_type=init_seq
```

**Для GPU (полноценное обучение):**

```bash
python train.py config/train_protein.py \
    --out_dir=out-protein-test \
    --device=cuda \
    --batch_size=32 \
    --block_size=155 \
    --n_layer=6 \
    --n_head=8 \
    --n_embd=256 \
    --max_iters=10000 \
    --eval_interval=500 \
    --data_type=init_seq
```

тут используется конфиг + переопределение части его значений, можно использовать только конфиг или только CLI

**Параметры обучения:**
- `--data_type`: `sequence` (с "-") или `init_seq` (без "-")
- `--class_prob`: вероятность добавления class токена (0.0-1.0, по умолчанию 0.5)
- `--type_prob`: вероятность добавления type токена (0.0-1.0, по умолчанию 0.3)
- `--out_dir`: папка для сохранения чекпоинтов
- `--device=cuda`: устройство используемое для обучения
- `--batch_size=32`: размер батча в эпохе
- `--block_size=155`: размер каждой последовательности в батче
- `--n_layer=6`: количество слоев
- `--n_head=8`: колво параллельных голов внимания
- `-n_embd=256`: размерность эмбелингов
- `--max_iters=10000`: максимальное колво эпох
- `--eval_interval=500`: каждые 500 будет оценка лоса на валидационных данных и сохранение чекпоинта
- `--data_type=init_seq`: определение типа последовательности4

### Шаг 3: Генерация последовательностей

**Базовая генерация (без префиксов):**

```bash
python sample.py \
    --out_dir=out-protein-test \
    --device=cpu \
    --num_samples=5 \
    --max_new_tokens=200 \
    --max_protein_length=150 \
    --temperature=0.8
```

**С class и type и start:**

```bash
python sample.py \
    --out_dir=out-protein-test \
    --device=cpu \
    --class_label="human" \
    --type_label="Heavy" \
    --num_samples=5 \
    --max_new_tokens=200 \
    --max_protein_length=200 \
    --start="QV"
```

**Доступные параметры генерации:**
- `--class_label`: класс последовательности (например, "human", "mouse_C57BL_6", "rabbit")
- `--type_label`: тип цепи ("Heavy" или "Light")
- `--start`: часть последовательности на старте (строка аминокислот)
- `--temperature`: температура сэмплирования 
- `--top_k`: top-k фильтрация
- `--max_protein_length`: максимальная длина генерируемой последовательности

---

## Подробости описания

### 1. Подготовка данных (`prepare.py`)

data/protein/prepare.py создаёт только meta.pkl
Включает:
 - Словарь (stoi, itos)
 - vocab_size
 - Информацию о классах и типах
 - Размер датасета


**Важные классы и типы:**

**17 классов последовательностей:**
- `Camel`, `HIS_mouse`, `human`, `mouse_BALB_c`, `mouse_Balb_c`, `mouse_C57BL_6`, `mouse_C57BL_6J`, `mouse_Igh_wt`, `mouse_Ighe_e`, `mouse_Ighg_g`, `mouse_RAG2_GFP_129Sve`, `mouse_Swiss_Webster`, `mouse_outbred`,`mouse_outbred_C57BL_6`, `rabbit`, `rat`, `rhesus`

**2 типа цепей:**
- `Heavy`
- `Light`

**Vocabulary:**
- 20 аминокислот: `ARNDCQEGHILKMFPSTWYV`
- Базовые токены: `<pad>`, `<eos>`, `<unk>`, `<sos>`
- Class токены: `<cls_human>`, `<cls_mouse_C57BL_6>`, и т.д.
- Type токены: `<type_Heavy>`, `<type_Light>`
- Gap символ: `-` (только для `sequence`, удаляется для `init_seq`)

**Итоговый vocab_size:** 43 токена
**После очистки для init_seq:** 42 токена (без "-")

### 2. Обучение модели (`train.py`)

**Архитектура:**
```python
class FlexibleProteinDataset(IterableDataset):
    # Streaming загрузка с Hugging Face Datasets
    # Динамическое добавление префиксов с заданной вероятностью
```

**Процесс обучения:**

1. **Загрузка meta.pkl:**
   ```python
   # Автоматическая очистка словаря для init_seq
   if data_type == 'init_seq':
       # Удаляется "-" из stoi/itos
       # Пересохраняется meta.pkl
   ```

2. **Создание датасета:**
   ```python
   # Streaming загрузка OAS95-aligned-cleaned
   train_dataset = FlexibleProteinDataset(
       split_name="train", 
       config=config, 
       meta=meta
   )
   ```

3. **Обработка примера:**

Для каждой последовательности:

1. Выбор data_type (sequence или init_seq)
2. Добавление класс токена (с вероятностью class_prob)
3. Добавление type токена (с вероятностью type_prob)
4. Формат: <sos> [<cls_X>] [<type_Y>] AMINO_ACIDS <eos>
5. Padding до block_size

4. Training loop:
   ```python
   # Standard GPT training:
   # - AdamW optimizer
   # - Cosine learning rate schedule
   # - Gradient accumulation
   # - Checkpointing best model
   ```

**Ключевые параметры модели:**
- `block_size`: длина контекста (155 для полных последовательностей)
- `n_layer`: количество Transformer слоёв (6 для среднего размера)
- `n_head`: количество attention heads (8)
- `n_embd`: размер embeddings (256)
- `dropout`: dropout rate (0.2 для регуляризации)

### 3. Генерация (`sample.py`)

**Процесс генерации:**

1. **Загрузка модели:**

* Загрузка checkpoint из out_dir
* Загрузка meta.pkl для decode


2. **Формирование префикса:**

* start_tokens = [<sos>]
* if class_label: добавить <cls_X>
* if type_label: добавить <type_Y>
* Закодировать начало - добавить частичную последовательность

3. **Генерация:**

* model.generate() - autoregressive sampling
* temperature для контроля разнообразия
* top_k для фильтрации маловероятных токенов

4. **Постобработка:**

* Обрезка по <eos>
* Удаление всех спецтоксенов после префикса
* Оставить только аминокислоты (и "-" для sequence)
* Обрезка до max_protein_length

**Примеры затравок:**

```bash
# Пустая затравка (модель сама решает всё)
python sample.py --out_dir=out-protein --start=""

# Только class
python sample.py --out_dir=out-protein --class_label="human"

# Только type
python sample.py --out_dir=out-protein --type_label="Heavy"

# Class + type
python sample.py --out_dir=out-protein --class_label="mouse_C57BL_6" --type_label="Light"

# С начальной последовательностью
python sample.py --out_dir=out-protein --start="EVQLV" --class_label="human"
```

---

## Конфигурация

### Файл конфигурации (`config/train_protein.py`)

```python
out_dir = 'out-protein'
eval_interval = 50        # Каждые N итераций - оценка на validation
log_interval = 10         # Каждые N итераций - вывод логов
max_iters = 1000          # Общее количество итераций

# Dataset
dataset = 'protein'

# Model size
batch_size = 32           # Размер батча
block_size = 155          # Длина контекста (максимальная длина последовательности)

n_layer = 6               # Количество Transformer слоёв
n_head = 8                # Количество attention heads
n_embd = 256              # Размерность embeddings
dropout = 0.2             # Dropout rate

# Training
learning_rate = 6e-4      # Learning rate
device = 'cpu'            # 'cpu', 'cuda', 'mps' (для Apple Silicon)
compile = False           # PyTorch 2.0 compile (True для GPU)

# Protein-specific
class_prob = 0.5          # Вероятность добавления class токена [0.0-1.0]
type_prob = 0.3           # Вероятность добавления type токена [0.0-1.0]
data_type = 'init_seq'    # 'sequence' (с "-") или 'init_seq' (без "-")
```

## Примеры использования

### Пример 1: Быстрый тест на CPU

```bash
# 1. Создание meta.pkl
python data/protein/prepare.py

# 2. Обучение маленькой модели (2-3 минуты)
python train.py config/train_protein.py \
    --out_dir=out-quick-test \
    --device=cpu \
    --compile=False \
    --batch_size=2 \
    --block_size=64 \
    --n_layer=2 \
    --n_head=2 \
    --n_embd=32 \
    --max_iters=50 \
    --eval_interval=10 \
    --data_type=init_seq

# 3. Генерация
python sample.py \
    --out_dir=out-quick-test \
    --device=cpu \
    --num_samples=3 \
    --max_protein_length=50 \
    --class_label="human" \
    --type_label="Heavy"
```

### Пример 2: Обучение на GPU

```bash
# В Kaggle Notebook или Google Colab
!git clone https://github.com/HermanDp45/BioInf_gpt.git
%cd BioInf_gpt
!pip install torch datasets transformers tqdm

# Подготовка
!python data/protein/prepare.py

# Обучение (настройте под доступную память)
!python train.py config/train_protein.py \
    --out_dir=out-protein-gpu \
    --device=cuda \
    --batch_size=16 \
    --block_size=128 \
    --n_layer=4 \
    --n_head=4 \
    --n_embd=128 \
    --max_iters=2000 \
    --eval_interval=100 \
    --data_type=init_seq

# Генерация
!python sample.py \
    --out_dir=out-protein-gpu \
    --device=cuda \
    --num_samples=10 \
    --class_label="human" \
    --type_label="Heavy"
```

### Пример 3: Сравнение sequence vs init_seq

```bash
# Обучение на sequence (с "-")
python train.py config/train_protein.py \
    --out_dir=out-sequence \
    --data_type=sequence \
    --max_iters=500

# Обучение на init_seq (без "-")
python train.py config/train_protein.py \
    --out_dir=out-init-seq \
    --data_type=init_seq \
    --max_iters=500

# Генерация из обеих моделей
python sample.py --out_dir=out-sequence --num_samples=5
python sample.py --out_dir=out-init-seq --num_samples=5
```

### Пример 4: Эксперименты с вероятностями префиксов

```bash
# Без префиксов (unconditional generation)
python train.py config/train_protein.py \
    --out_dir=out-no-prefix \
    --class_prob=0.0 \
    --type_prob=0.0 \
    --max_iters=500

# Всегда с префиксами (conditional generation)
python train.py config/train_protein.py \
    --out_dir=out-full-prefix \
    --class_prob=1.0 \
    --type_prob=1.0 \
    --max_iters=500

# Смешанный режим (по умолчанию)
python train.py config/train_protein.py \
    --out_dir=out-mixed-prefix \
    --class_prob=0.5 \
    --type_prob=0.3 \
    --max_iters=500
```

---

## Архитектура

### Структура проекта

```
BioInf_gpt/
├── data/
│   └── protein/
│       ├── prepare.py          # Создание meta.pkl
│       └── meta.pkl            # Словарь и метаданные
├── config/
│   └── train_protein.py        # Конфигурация обучения
├── model.py                    # GPT архитектура
├── train.py                    # Training loop
├── sample.py                   # Генерация
├── configurator.py             # Парсинг аргументов
└── README_PROTEIN.md           # Эта документация
```
---


## Дальнейшие улучшения

### Предложения по развитию проекта:
1. **Оценка качества:**
   - Добавить метрики: perplexity, BLOSUM similarity с реальными последовательностями
   - Проверка на валидность структуры (с помощью AlphaFold)

2. **Архитектура:**
   - Попробовать диффузионную модель
   - Экспериментировать с размером модели

3. **Данные:**
   - Фильтрация последовательностей по качеству
   - Балансировка классов
   - Аугментация (например random masking)

---
