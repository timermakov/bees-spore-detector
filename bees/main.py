import os
from bees import config, io_utils, image_proc, spores, titr

def process_image(image_path, xml_path, result_path):
    image = io_utils.load_image(image_path)
    metadata = io_utils.load_metadata(xml_path)
    img_arr = image_proc.preprocess_image(image)
    spore_objs = image_proc.detect_spores(img_arr)
    count = spores.count_spores(spore_objs)
    t = titr.calculate_titr(count)
    md = f"""# Результаты анализа изображения {os.path.basename(image_path)}\n\n- Количество спор: {count}\n- Титр (млн спор/мл): {t:.2f}\n"""
    io_utils.save_markdown(result_path, md)

def main():
    data_dir = config.DATA_DIR
    res_dir = config.RESULTS_DIR
    for fname in os.listdir(data_dir):
        if fname.lower().endswith('.jpg'):
            image_path = os.path.join(data_dir, fname)
            xml_path = image_path + '_meta.xml'
            if not os.path.exists(xml_path):
                continue
            result_path = os.path.join(res_dir, fname.replace('.jpg', '.md'))
            process_image(image_path, xml_path, result_path)

if __name__ == '__main__':
    main() 