import os
from bees import config, io_utils, image_proc, spores, titr
import shutil
import zipfile

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

def make_cvat_export(task_name, image_files, spore_objs_list, output_dir):
    """
    Create a folder with CVAT export structure and zip it:
    taskname/
      images/
        img1.jpg ...
      annotations.xml
    """
    import os
    from xml.etree import ElementTree as ET
    from bees.io_utils import export_cvat_xml_elements
    export_dir = os.path.join(output_dir, task_name)
    images_dir = os.path.join(export_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    # Copy images
    for img in image_files:
        shutil.copy(img, images_dir)
    # Build XML: <annotations><version>1.1</version><meta>...</meta><image>...</image>...</annotations>
    root = ET.Element('annotations')
    ET.SubElement(root, 'version').text = '1.1'
    metas = []
    for i, (img, spore_objs) in enumerate(zip(image_files, spore_objs_list)):
        meta, image_elem = export_cvat_xml_elements(img, spore_objs, image_id=i)
        metas.append(meta)
        root.append(image_elem)
    # Use the first meta (or merge if needed)
    if metas:
        root.insert(1, metas[0])
    # Pretty-print XML before saving
    from bees.io_utils import indent_xml
    indent_xml(root)
    merged_xml_path = os.path.join(export_dir, 'annotations.xml')
    ET.ElementTree(root).write(merged_xml_path, encoding='utf-8', xml_declaration=True)
    # Zip the folder
    zip_path = os.path.join(output_dir, f'{task_name}.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root_dir, _, files in os.walk(export_dir):
            for file in files:
                abs_path = os.path.join(root_dir, file)
                rel_path = os.path.relpath(abs_path, output_dir)
                zipf.write(abs_path, rel_path)
    return zip_path

def main():
    data_dir = config.DATA_DIR
    res_dir = config.RESULTS_DIR
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    pairs = io_utils.list_image_pairs(data_dir)
    image_files = []
    spore_objs_list = []
    for image_path, xml_path in pairs:
        base = os.path.splitext(os.path.basename(image_path))[0]
        md_path = os.path.join(res_dir, base + '.md')
        debug_prefix = os.path.join(res_dir, base)
        debug_path = debug_prefix + '_debug'
        result = process_image(image_path, xml_path, debug_prefix=debug_prefix)
        md = f"""# Результаты анализа изображения {os.path.basename(image_path)}\n\n- Количество спор: {result['count']}\n- Титр (млн спор/мл): {result['titr']:.2f}\n"""
        io_utils.save_markdown(md_path, md)
        image_proc.save_debug_image(result['image'], result['spore_objs'], debug_path)
        image_files.append(image_path)
        spore_objs_list.append(result['spore_objs'])
    # After all images processed, make CVAT export
    task_name = 'bees_task'
    make_cvat_export(task_name, image_files, spore_objs_list, res_dir)

if __name__ == '__main__':
    main() 