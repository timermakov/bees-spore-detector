import numpy as np
from PIL import Image

def preprocess_image(image: Image.Image):
    # TODO: преобразование в ч/б, фильтрация, повышение контраста
    return np.array(image)

def detect_spores(image_array: np.ndarray):
    # TODO: выделение спор, возвращает список контуров или маску
    return [] 