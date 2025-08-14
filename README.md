# Bees Spore Counter

Автоматический подсчёт спор на микроскопических изображениях камеры Горяева.

## Структура проекта

- `datasetN/` — изображения и метаданные
- `resultsN/` — результаты анализа (Markdown)
- `bees/` — основной код
- `config.yaml` - настройки проекта и параметров распознавания

## Запуск

1. Установите зависимости:
   ```
   pip install -r requirements.txt
   ```
2. Подготовьте конфигурацию `config.yaml` (пример ниже) и запустите CLI:
   ```
   python -m bees.main -c config.yaml
   ```

### Пример config.yaml
```yaml
# Путь к папке с изображениями.Файлы должны иметь формат <имя>_1.jpg, <имя>_2.jpg, <имя>_3.jpg
data_dir: dataset2
# Папка для результатов (.md, debug изображения, отчёт Excel)
results_dir: results2

# Параметры обработки (влияние на детекцию см. bees/image_proc.py)
# Минимальная/максимальная площадь эллиптического контура споры (пиксели)
min_spore_area: 25
max_spore_area: 500
# Пороговые значения алгоритма Canny для выделения границ
canny_threshold1: 40
canny_threshold2: 125
# Минимальная длина контура (число точек) для эллипса
min_spore_contour_length: 5
# Порог по интенсивности: насколько средняя яркость внутри эллипса может отличаться от фона
intensity_threshold: 50
```

Дополнительно:
- Можно переопределить параметры командной строкой: `-d` для data_dir, `-o` для results_dir.
- Для экспорта CVAT 1.1 с эллипсами добавьте флаг `--export-cvat-zip`.

Пример:
```
python -m bees.main -c config.yaml -d path\to\data -o path\to\results --export-cvat-zip
```