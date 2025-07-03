import numpy as np
from PIL import Image
import cv2
from bees import config

def preprocess_image(image: Image.Image, debug_path=None):
    gray = image.convert('L')
    arr = np.array(gray)
    blurred = cv2.GaussianBlur(arr, (5, 5), 2)
    if debug_path is not None:
        save_debug_image(blurred, [], debug_path + '_blur', is_mask=True)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    arr = clahe.apply(blurred)
    if debug_path is not None:
        save_debug_image(arr, [], debug_path + '_clahe', is_mask=True)
    return arr

def detect_spores(image_array: np.ndarray, min_area=10, max_area=500, debug_path=None):
    # 1. Детектор границ (Canny)
    edges = cv2.Canny(image_array, 40, 120)
    if debug_path is not None:
        save_debug_image(edges, [], debug_path + '_edges', is_mask=True)
    # 2. Поиск замкнутых контуров
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    spores = []
    for cnt in contours:
        if len(cnt) < 5:
            continue
        area = cv2.contourArea(cnt)
        if not (min_area < area < max_area):
            continue
        ellipse = cv2.fitEllipse(cnt)
        (x, y), (MA, ma), angle = ellipse
        ratio = min(MA, ma) / max(MA, ma)
        #if ratio < 0.4 or ratio > 0.95:
        #    continue
        ecc = np.sqrt(1 - (min(MA, ma) / max(MA, ma))**2)
        #if not (0.3 < ecc < 0.95):
        #    continue
        # Проверка на "полость": средняя яркость внутри эллипса близка к фону
        mask = np.zeros(image_array.shape, dtype=np.uint8)
        cv2.ellipse(mask, (int(x), int(y)), (int(MA/2), int(ma/2)), angle, 0, 360, 255, -1)
        mean_inside = cv2.mean(image_array, mask=mask)[0]
        mean_total = np.mean(image_array)
        if abs(mean_inside - mean_total) > 50:  # если внутри эллипса сильно отличается от фона, пропускаем
            continue
        spores.append(cnt)
    # Debug: рисуем эллипсы
    if debug_path is not None:
        debug_img = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
        for cnt in spores:
            ellipse = cv2.fitEllipse(cnt)
            cv2.ellipse(debug_img, ellipse, (0,0,255), 2)
        cv2.imwrite(debug_path + '_ellipses.jpg', debug_img)
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