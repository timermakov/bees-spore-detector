# Конфигурация проекта Bees

DATA_DIR = 'data'
RESULTS_DIR = 'res'
SCALE_MICRONS = 20  # масштаб на изображении (микрометры)

# Параметры обработки изображений (можно расширять)
DEFAULT_THRESHOLD = 127
MIN_SPORE_AREA = 25  # минимальная площадь споры (пиксели)
MAX_SPORE_AREA = 150 # максимальная площадь споры (пиксели) 
USE_ML_DETECTION = False  # использовать ML для детекции спор
ML_MODEL_PATH = 'model.pt'  # путь к обученной модели (YOLOv5/YOLOv8 .pt) 