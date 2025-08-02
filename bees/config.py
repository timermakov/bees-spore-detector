# Конфигурация проекта Bees

DATA_DIR = 'data'
RESULTS_DIR = 'res_ellipse'
SCALE_MICRONS = 20  # масштаб на изображении (микрометры)

# Параметры обработки изображений (можно расширять)
MIN_SPORE_AREA = 25  				# минимальная площадь споры (пиксели)
MAX_SPORE_AREA = 500 				# максимальная площадь споры (пиксели) 
CANNY_THRESHOLD1 = 40 				# порог 1 для Canny
CANNY_THRESHOLD2 = 125 				# порог 2 для Canny
MIN_SPORE_CONTOUR_LENGTH = 5 		# минимальная длина контура споры
INTENSITY_THRESHOLD = 50 			# порог интенсивности для споры