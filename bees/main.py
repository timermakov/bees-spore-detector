import os
from bees import config, io_utils, image_proc, spores, titr

def process_image(image_path, xml_path, debug_prefix=None):
    image = io_utils.load_image(image_path)
    metadata = io_utils.load_metadata(xml_path)
    # Сохраняем grayscale/contrast debug
    preproc_debug = debug_prefix + '_preproc' if debug_prefix else None
    img_arr = image_proc.preprocess_image(image, debug_path=preproc_debug)
    # Сохраняем бинаризацию/маску debug
    mask_debug = debug_prefix + '_mask' if debug_prefix else None
    spore_objs = image_proc.detect_spores(img_arr, 
                                          config.MIN_SPORE_AREA, config.MAX_SPORE_AREA, 
                                          config.CANNY_THRESHOLD1, config.CANNY_THRESHOLD2, 
                                          config.MIN_SPORE_CONTOUR_LENGTH, 
                                          debug_path=mask_debug)
    count = spores.count_spores(spore_objs)
    t = titr.calculate_titr(count)
    return {
        'image': image,
        'spore_objs': spore_objs,
        'count': count,
        'titr': t
    }

def main():
    data_dir = config.DATA_DIR
    res_dir = config.RESULTS_DIR
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    pairs = io_utils.list_image_pairs(data_dir)
    for image_path, xml_path in pairs:
        base = os.path.splitext(os.path.basename(image_path))[0]
        md_path = os.path.join(res_dir, base + '.md')
        debug_prefix = os.path.join(res_dir, base)
        debug_path = debug_prefix + '_debug'
        result = process_image(image_path, xml_path, debug_prefix=debug_prefix)
        md = f"""# Результаты анализа изображения {os.path.basename(image_path)}\n\n- Количество спор: {result['count']}\n- Титр (млн спор/мл): {result['titr']:.2f}\n"""
        io_utils.save_markdown(md_path, md)
        image_proc.save_debug_image(result['image'], result['spore_objs'], debug_path)

if __name__ == '__main__':
    main() 