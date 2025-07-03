import numpy as np
from PIL import Image
import cv2

def preprocess_image(image: Image.Image, debug_path=None):
    # Переводим в оттенки серого
    gray = image.convert('L')
    arr = np.array(gray)
    # Сглаживание для уменьшения шума
    blurred = cv2.GaussianBlur(arr, (5, 5), 2)
    if debug_path is not None:
        save_debug_image(blurred, [], debug_path + '_blur', is_mask=True)
    # Локальное повышение контраста (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    arr = clahe.apply(blurred)
    if debug_path is not None:
        save_debug_image(arr, [], debug_path, is_mask=True)
    return arr

def detect_spores(image_array: np.ndarray, min_area=10, max_area=500, debug_path=None):
    # Адаптивная бинаризация
    bin_img = cv2.adaptiveThreshold(
        image_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)
    # Инвертируем, чтобы споры были белыми
    bin_img = 255 - bin_img
    if debug_path is not None:
        save_debug_image(bin_img, [], debug_path + '_adaptive', is_mask=True)
    # Морфологическая обработка
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    clean = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=2)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=1)
    if debug_path is not None:
        save_debug_image(clean, [], debug_path + '_morph', is_mask=True)
    # Поиск контуров
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Фильтрация по площади и форме
    spores = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area < area < max_area):
            continue
        if len(cnt) < 5:
            continue
        ellipse = cv2.fitEllipse(cnt)
        (x, y), (MA, ma), angle = ellipse
        ratio = min(MA, ma) / max(MA, ma)
        if ratio < 0.4:  # слишком вытянутые или слишком круглые
            continue
        # Эксцентриситет
        ecc = np.sqrt(1 - (min(MA, ma) / max(MA, ma))**2)
        if not (0.3 < ecc < 0.95):
            continue
        spores.append(cnt)
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
        cv2.imwrite(out_path + '.jpg', arr)
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
    cv2.imwrite(out_path + '.jpg', img_bgr) 