import numpy as np
from PIL import Image
import cv2

def preprocess_image(image: Image.Image, debug_path=None):
    # Переводим в оттенки серого
    gray = image.convert('L')
    arr = np.array(gray)
    # Гистограммное выравнивание для повышения контраста
    arr = cv2.equalizeHist(arr)
    if debug_path is not None:
        save_debug_image(arr, [], debug_path, is_mask=True)
    return arr

def detect_spores(image_array: np.ndarray, min_area=10, max_area=500, debug_path=None):
    # Бинаризация по Оцу
    _, thresh = cv2.threshold(image_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Морфологическая обработка для удаления шума
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    if debug_path is not None:
        save_debug_image(clean, [], debug_path, is_mask=True)
    # Поиск контуров
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Фильтрация по площади
    spores = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
    return spores

def save_debug_image(image, spores, out_path, is_mask=False):
    """
    Сохраняет изображение с контурами (если есть). Если is_mask=True, просто сохраняет ч/б маску.
    image: np.ndarray (ч/б) или PIL.Image (цвет)
    spores: список контуров
    """
    if is_mask:
        if isinstance(image, Image.Image):
            arr = np.array(image)
        else:
            arr = image
        cv2.imwrite(out_path, arr)
        return
    if isinstance(image, Image.Image):
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        # если уже np.ndarray (например, после обработки)
        if len(image.shape) == 2:
            img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = image
    cv2.drawContours(img_bgr, spores, -1, (0,0,255), 2)
    cv2.imwrite(out_path, img_bgr) 