import numpy as np
from PIL import Image
import cv2

def preprocess_image(image: Image.Image):
    # Переводим в оттенки серого
    gray = image.convert('L')
    arr = np.array(gray)
    # Гистограммное выравнивание для повышения контраста
    arr = cv2.equalizeHist(arr)
    return arr

def detect_spores(image_array: np.ndarray, min_area=10, max_area=500):
    # Бинаризация по Оцу
    _, thresh = cv2.threshold(image_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Морфологическая обработка для удаления шума
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # Поиск контуров
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Фильтрация по площади
    spores = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
    return spores

def save_debug_image(orig_image: Image.Image, spores, out_path):
    # Рисуем контуры на копии исходного изображения
    img_bgr = cv2.cvtColor(np.array(orig_image), cv2.COLOR_RGB2BGR)
    cv2.drawContours(img_bgr, spores, -1, (0,0,255), 2)
    cv2.imwrite(out_path, img_bgr) 