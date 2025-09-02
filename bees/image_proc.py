import numpy as np
from PIL import Image
import cv2

def get_analysis_square_coords(image_shape, square_size):
    """
    Calculate the coordinates for the analysis square centered in the image.
    
    Args:
        image_shape: tuple of (height, width) or (height, width, channels)
        square_size: size of the square in pixels
    
    Returns:
        tuple: (x1, y1, x2, y2) coordinates of the square
    """
    height, width = image_shape[:2]
    
    # Calculate center of the image
    center_x = width // 2
    center_y = height // 2
    
    # Calculate square coordinates
    half_size = square_size // 2
    x1 = center_x - half_size
    y1 = center_y - half_size
    x2 = center_x + half_size
    y2 = center_y + half_size
    
    return x1, y1, x2, y2

def is_spore_in_analysis_square(ellipse_center, square_coords):
    """
    Check if a spore's center is within the analysis square (including touching the lines).
    
    Args:
        ellipse_center: tuple of (x, y) coordinates of the spore center
        square_coords: tuple of (x1, y1, x2, y2) coordinates of the square
    
    Returns:
        bool: True if spore is within or touching the square
    """
    x, y = ellipse_center
    x1, y1, x2, y2 = square_coords
    
    # Check if center is within or on the boundary of the square
    return x1 <= x <= x2 and y1 <= y <= y2

def preprocess_image(image: Image.Image, debug_path=None, analysis_square_size=None, analysis_square_line_width=None):
    gray = image.convert('L')
    arr = np.array(gray)
    blurred = cv2.GaussianBlur(arr, (5, 5), 2)
    if debug_path is not None:
        save_debug_image(blurred, [], debug_path + '_blur', is_mask=True, 
                        analysis_square_size=analysis_square_size, 
                        analysis_square_line_width=analysis_square_line_width)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    arr = clahe.apply(blurred)
    if debug_path is not None:
        save_debug_image(arr, [], debug_path + '_clahe', is_mask=True,
                        analysis_square_size=analysis_square_size, 
                        analysis_square_line_width=analysis_square_line_width)
    return arr

def detect_spores(image_array: np.ndarray,
                  min_contour_area: int,
                  max_contour_area: int,
                  min_ellipse_area: int,
                  max_ellipse_area: int,
                  canny_threshold1: int,
                  canny_threshold2: int,
                  min_spore_contour_length: int,
                  intensity_threshold: int,
                  analysis_square_size: int,
                  analysis_square_line_width: int,
                  debug_path=None):
    # 0. Предварительное подавление шума с сохранением граней
    # Небольшой bilateral уменьшает текстуру, оставляя контуры спор
    denoised = cv2.bilateralFilter(image_array, d=5, sigmaColor=20, sigmaSpace=5)

    # 1. Детектор границ (Canny) и объединение двух уровней для повышения полноты
    edges_strong = cv2.Canny(denoised, canny_threshold1, canny_threshold2)
    edges_soft = cv2.Canny(denoised, max(0, int(canny_threshold1 * 0.8)), max(0, int(canny_threshold2 * 0.8)))
    edges = cv2.bitwise_or(edges_strong, edges_soft)
    if debug_path is not None:
        save_debug_image(edges, [], debug_path + '_edges', is_mask=True,
                        analysis_square_size=analysis_square_size, 
                        analysis_square_line_width=analysis_square_line_width)
    
    # 1.1. Лёгкая морфологическая очистка шума
    kernel = np.ones((2, 2), np.uint8)  # gentler kernel
    edges_morph = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    # Всегда сохраняем результат морфологии отдельным файлом
    if debug_path is not None:
        save_debug_image(edges_morph, [], debug_path + '_edges_morph', is_mask=True,
                        analysis_square_size=analysis_square_size, 
                        analysis_square_line_width=analysis_square_line_width)
    # Если морфология слишком агрессивна — используем исходные края дальше
    use_fallback = False
    sum_edges = float(np.count_nonzero(edges)) + 1e-9
    sum_morph = float(np.count_nonzero(edges_morph))
    if sum_morph < 0.1 * sum_edges:
        use_fallback = True
        if debug_path is not None:
            print("DEBUG: Morph too aggressive → fallback to raw edges")
    edges_working = edges if use_fallback else edges_morph
    
    # 1.2. Попытка убрать длинные линии (волоски/границы)
    lines = cv2.HoughLinesP(edges_working, 1, np.pi/180, 150, minLineLength=80, maxLineGap=5)  # conservative
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(edges_working, (x1, y1), (x2, y2), 0, 2)
    if debug_path is not None:
        save_debug_image(edges_working, [], debug_path + '_edges_nolines', is_mask=True,
                        analysis_square_size=analysis_square_size, 
                        analysis_square_line_width=analysis_square_line_width)

    # 1.3. Лёгкое замыкание разорванных контуров (closing)
    kernel_close = np.ones((2, 2), np.uint8)
    edges_closed = cv2.morphologyEx(edges_working, cv2.MORPH_CLOSE, kernel_close)
    if debug_path is not None:
        save_debug_image(edges_closed, [], debug_path + '_edges_close', is_mask=True,
                        analysis_square_size=analysis_square_size, 
                        analysis_square_line_width=analysis_square_line_width)
    # Если closing раздувает шум более чем на 50%, оставим предыдущее
    if float(np.count_nonzero(edges_closed)) > 1.5 * float(np.count_nonzero(edges_working)):
        edges_final = edges_working
    else:
        edges_final = edges_closed
    
    # 2. Поиск замкнутых контуров
    # Важно: используем RETR_LIST, чтобы не терять внутренние/разорванные контуры спор
    contours, _ = cv2.findContours(edges_final, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print(f"DEBUG {debug_path}: Found {len(contours)} total contours")
    spores_inside = []
    spores_outside = []
    accepted_centers = []  # для дедубликации близких эллипсов (для всех принятых спор)
    min_center_dist_px = 6.0
    contour_count = 0
    # Рассчитываем координаты квадрата один раз
    square_coords = get_analysis_square_coords(image_array.shape, analysis_square_size)
    for cnt in contours:
        contour_count += 1
        # 2.1. Ограничение длины контура споры (по количеству точек)
        if len(cnt) < min_spore_contour_length:
            continue
        
        # 2.2. Ограничение размеров (по площади) контура
        area = cv2.contourArea(cnt)
        if not (min_contour_area < area < max_contour_area):
            continue
        
        # 2.3. Определение эллипса и геометрические фильтры
        if len(cnt) < 5:  # fitEllipse requires at least 5 points
            continue
        try:
            ellipse = cv2.fitEllipse(cnt)
        except Exception:
            # Невалидный контур для эллипса
            continue

        # Параметры эллипса
        (x, y), (MA, ma), angle = ellipse
        
        # 2.4. Ограничение размеров (по площади) эллипса
        ellipse_area = np.pi * (MA / 2.0) * (ma / 2.0)
        if not (min_ellipse_area < ellipse_area < max_ellipse_area):
            continue 

        # 2.5. Фильтр по отношению осей и эксцентриситету (расширено для повышения полноты)
        ratio = min(MA, ma) / max(MA, ma)
        if ratio < 0.45 or ratio > 0.92:
            continue
        ecc = np.sqrt(1 - (min(MA, ma) / max(MA, ma))**2)
        if not (0.45 < ecc < 0.95):
            continue
        
        # 2.6. Проверка на "полость": средняя яркость внутри эллипса близка к фону
        mask = np.zeros(image_array.shape, dtype=np.uint8)
        cv2.ellipse(mask, (int(x), int(y)), (int(MA/2), int(ma/2)), angle, 0, 360, 255, -1)
        mean_inside = cv2.mean(image_array, mask=mask)[0]
        mean_total = np.mean(image_array)
        diff_intensity = abs(mean_inside - mean_total)
        # Проверку можно отключить, установив intensity_threshold < 0 в конфиге
        if intensity_threshold is not None and intensity_threshold >= 0:
            if diff_intensity > intensity_threshold:
                continue

        # 2.7. Подтверждение по поддержке границ
        edges_for_support = cv2.dilate(edges_strong, np.ones((3, 3), np.uint8), iterations=1)
        perim_mask = np.zeros_like(edges_for_support)
        cv2.ellipse(perim_mask, (int(x), int(y)), (int(MA/2), int(ma/2)), angle, 0, 360, 255, 1)
        support_pixels = cv2.countNonZero(cv2.bitwise_and(edges_for_support, perim_mask))
        perim_pixels = cv2.countNonZero(perim_mask)
        support_ratio = float(support_pixels) / float(perim_pixels + 1e-9)
        if support_ratio < 0.25:
            continue
        
        # 2.8. Дедубликация по центрам среди всех принятых спор
        is_duplicate = False
        for (cx0, cy0) in accepted_centers:
            if (x - cx0) ** 2 + (y - cy0) ** 2 < (min_center_dist_px ** 2):
                is_duplicate = True
                break
        if is_duplicate:
            continue

        # 2.9. Классификация по зоне анализа: внутри/снаружи
        if is_spore_in_analysis_square((x, y), square_coords):
            spores_inside.append(cnt)
        else:
            spores_outside.append(cnt)
        accepted_centers.append((x, y))
    
    print(f"DEBUG {debug_path}: Processed {contour_count} contours, kept {len(spores_inside) + len(spores_outside)} spores (inside={len(spores_inside)}, outside={len(spores_outside)})")
    # Debug: рисуем эллипсы и зелёный квадрат анализа
    if debug_path is not None:
        debug_img = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
        
        # Draw the analysis square
        x1, y1, x2, y2 = square_coords
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), analysis_square_line_width)
        
        # Draw detected spores: inside (red), outside (blue)
        for cnt in spores_inside:
            ellipse = cv2.fitEllipse(cnt)
            cv2.ellipse(debug_img, ellipse, (0,0,255), 2)
        for cnt in spores_outside:
            ellipse = cv2.fitEllipse(cnt)
            cv2.ellipse(debug_img, ellipse, (255,0,0), 2)
        
        cv2.imwrite(debug_path + '_ellipses.jpg', debug_img)
    return spores_inside, spores_outside

def save_debug_image(image, spores, out_path, is_mask=False, analysis_square_size=None, analysis_square_line_width=None, spores_outside=None):
    """
    Сохраняет изображение с контурами (если есть). Если is_mask=True, просто сохраняет ч/б маску.
    image: np.ndarray (ч/б) или PIL.Image (цвет)
    spores: список контуров
    analysis_square_size: размер квадрата анализа (если None, квадрат не рисуется)
    analysis_square_line_width: толщина линии квадрата
    spores_outside: список контуров спор за пределами квадрата (для отрисовки синим)
    """
    if is_mask:
        if isinstance(image, Image.Image):
            arr = np.array(image)
        else:
            arr = image
        # Если нужно рисовать квадрат, переключимся на BGR и нарисуем
        if analysis_square_size is not None and analysis_square_line_width is not None:
            if len(arr.shape) == 2:
                img_bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            else:
                img_bgr = arr.copy()
            square_coords = get_analysis_square_coords(img_bgr.shape, analysis_square_size)
            x1, y1, x2, y2 = square_coords
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), analysis_square_line_width)
            cv2.imwrite(out_path + '.jpg', img_bgr)
        else:
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
    
    # Draw the analysis square if parameters are provided
    if analysis_square_size is not None and analysis_square_line_width is not None:
        square_coords = get_analysis_square_coords(img_bgr.shape, analysis_square_size)
        x1, y1, x2, y2 = square_coords
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), analysis_square_line_width)
    
    # Draw inside spores (red)
    cv2.drawContours(img_bgr, spores, -1, (0,0,255), 2)
    # Draw outside spores (blue)
    if spores_outside:
        cv2.drawContours(img_bgr, spores_outside, -1, (255,0,0), 2)
    cv2.imwrite(out_path + '.jpg', img_bgr) 