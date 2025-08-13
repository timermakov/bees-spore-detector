import numpy as np
from PIL import Image
import cv2

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

def detect_spores(image_array: np.ndarray,
                  min_area: int,
                  max_area: int,
                  canny_threshold1: int,
                  canny_threshold2: int,
                  min_spore_contour_length: int,
                  intensity_threshold: int,
                  debug_path=None):
    # 1. Детектор границ (Canny)
    edges = cv2.Canny(image_array, canny_threshold1, canny_threshold2)
    if debug_path is not None:
        save_debug_image(edges, [], debug_path + '_edges', is_mask=True)
    # 1.1. Лёгкая морфологическая очистка шума
    #kernel = np.ones((3, 3), np.uint8)
    #edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    #if debug_path is not None:
    #    save_debug_image(edges, [], debug_path + '_edges_morph', is_mask=True)
    # 1.2. Попытка убрать длинные линии (волоски/границы)
    #lines = cv2.HoughLinesP(edges, 1, np.pi/180, 120, minLineLength=60, maxLineGap=8)
    #if lines is not None:
    #    for line in lines:
    #        x1, y1, x2, y2 = line[0]
    #        cv2.line(edges, (x1, y1), (x2, y2), 0, 2)
    #if debug_path is not None:
    #    save_debug_image(edges, [], debug_path + '_edges_nolines', is_mask=True)
    
    # 2. Поиск замкнутых контуров
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    spores = []
    for cnt in contours:
        # 2.1. Ограничение длины контура споры (по количеству точек)
        if len(cnt) < min_spore_contour_length:
            continue
        
        # 2.2. Ограничение размеров (по площади) споры
        area = cv2.contourArea(cnt)
        if not (min_area < area < max_area):
            continue
        
        # 2.3. Определение эллипса (нужно настроить)
        if len(cnt) < 5:
            continue
        try:
            ellipse = cv2.fitEllipse(cnt)
        except Exception:
            # Невалидный контур для эллипса
            continue
        
        (x, y), (MA, ma), angle = ellipse
        ratio = min(MA, ma) / max(MA, ma)
        # Фильтр по отношению осей и эксцентриситету (под споры)
        if ratio < 0.5 or ratio > 0.9:
            continue
        ecc = np.sqrt(1 - (min(MA, ma) / max(MA, ma))**2)
        if not (0.5 < ecc < 0.92):
            continue
        
        # 2.4. Проверка на "полость": средняя яркость внутри эллипса близка к фону
        mask = np.zeros(image_array.shape, dtype=np.uint8)
        cv2.ellipse(mask, (int(x), int(y)), (int(MA/2), int(ma/2)), angle, 0, 360, 255, -1)
        mean_inside = cv2.mean(image_array, mask=mask)[0]
        mean_total = np.mean(image_array)
        if abs(mean_inside - mean_total) > intensity_threshold:  # если внутри эллипса сильно отличается от фона, пропускаем
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