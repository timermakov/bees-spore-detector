import os
from bees import config, io_utils, image_proc, spores, titr

def process_image(image_path, xml_path):
    image = io_utils.load_image(image_path)
    metadata = io_utils.load_metadata(xml_path)
    img_arr = image_proc.preprocess_image(image)
    spore_objs = image_proc.detect_spores(img_arr, config.MIN_SPORE_AREA, config.MAX_SPORE_AREA)
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
        debug_path = os.path.join(res_dir, base + '_debug.jpg')
        result = process_image(image_path, xml_path)
        md = f"""# Результаты анализа изображения {os.path.basename(image_path)}\n\n- Количество спор: {result['count']}\n- Титр (млн спор/мл): {result['titr']:.2f}\n"""
        io_utils.save_markdown(md_path, md)
        image_proc.save_debug_image(result['image'], result['spore_objs'], debug_path)

if __name__ == '__main__':
    main() 