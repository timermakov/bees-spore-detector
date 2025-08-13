import os
import argparse
from bees import io_utils, image_proc, spores, titr
from bees.config_loader import load_config, get_param
from bees.grouping import list_grouped_images
from bees.reporting import write_markdown_report, export_excel
import shutil
import zipfile
from datetime import datetime

def process_image(image_path, xml_path, params, debug_prefix=None):
    image = io_utils.load_image(image_path)
    metadata = io_utils.load_metadata(xml_path)
    # Сохраняем grayscale/contrast debug
    preproc_debug = debug_prefix + '_preproc' if debug_prefix else None
    img_arr = image_proc.preprocess_image(image, debug_path=preproc_debug)
    # Сохраняем бинаризацию/маску debug
    mask_debug = debug_prefix + '_mask' if debug_prefix else None
    spore_objs = image_proc.detect_spores(
        img_arr,
        min_area=params.get('min_spore_area'),
        max_area=params.get('max_spore_area'),
        canny_threshold1=params.get('canny_threshold1'),
        canny_threshold2=params.get('canny_threshold2'), 
        min_spore_contour_length=params.get('min_spore_contour_length'),
        intensity_threshold=params.get('intensity_threshold'),
        debug_path=mask_debug
    )
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
    parser = argparse.ArgumentParser(description='Bees Spore Counter CLI')
    parser.add_argument('-c', '--config', required=False, default='config.yaml', help='Path to YAML config')
    parser.add_argument('-d', '--data', required=False, help='Override data directory (images)')
    parser.add_argument('-o', '--output', required=False, help='Override results directory')
    parser.add_argument('--export-cvat-zip', action='store_true', help='Export CVAT 1.1 ZIP with ellipses')
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = args.data or get_param(cfg, 'data_dir', 'dataset2')
    res_dir = args.output or get_param(cfg, 'results_dir', 'results2')
    
    # Создаём папки, если не существуют
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
        
    # Загружаем параметры из конфига
    params = {
        'min_spore_area': get_param(cfg, 'min_spore_area', 25),
        'max_spore_area': get_param(cfg, 'max_spore_area', 500),
        'canny_threshold1': get_param(cfg, 'canny_threshold1', 40),
        'canny_threshold2': get_param(cfg, 'canny_threshold2', 125),
        'min_spore_contour_length': get_param(cfg, 'min_spore_contour_length', 5),
        'intensity_threshold': get_param(cfg, 'intensity_threshold', 50),
    }
    # Debug: print config, paths, and parameters in a clear, readable format
    print("\n===== Bees Spore Counter Run Info =====")
    print(f"Config file:   {os.path.abspath(args.config)}")
    print(f"Data dir:      {os.path.abspath(data_dir)} (exists: {os.path.isdir(data_dir)})")
    print(f"Results dir:   {os.path.abspath(res_dir)}")
    print("Parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    print("=======================================\n")

    # Save run parameters to results directory for reproducibility
    params_txt = os.path.join(res_dir, 'params_used.txt')
    log_lines = [
        "--------------------------------",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Config file:   {os.path.abspath(args.config)}",
        f"Data dir:      {os.path.abspath(data_dir)} (exists: {os.path.isdir(data_dir)})",
        f"Results dir:   {os.path.abspath(res_dir)}",
        "Parameters:"
    ] + [f"  {k}: {v}" for k, v in params.items()] + [
        "--------------------------------"
    ]
    try:
        with open(params_txt, 'a', encoding='utf-8') as f:
            f.write('\n'.join(log_lines) + '\n')
    except Exception as e:
        print(f"Warning: Could not write params log: {e}")

    # Validate and group images
    groups, errors = list_grouped_images(data_dir)
    if errors:
        print("\n".join(errors))
        if not groups:
            return

    # Process images and collect results per group
    groups_results = {}
    image_files = []
    spore_objs_list = []
    for prefix, image_paths in groups.items():
        group_counts = []
        md_records = []  # (md_path, image_path, count)
        for idx, image_path in enumerate(image_paths, start=1):
            xml_path = image_path + '_meta.xml'
            if not os.path.exists(xml_path):
                print(f"Не найден мета-файл: {os.path.basename(image_path)}_meta.xml")
                continue
            base = os.path.splitext(os.path.basename(image_path))[0]
            md_path = os.path.join(res_dir, base + '.md')
            debug_prefix = os.path.join(res_dir, base)
            debug_path = debug_prefix + '_debug'
            result = process_image(image_path, xml_path, params, debug_prefix=debug_prefix)
            # defer markdown until group titr is known
            md_records.append((md_path, image_path, result['count']))
            image_proc.save_debug_image(result['image'], result['spore_objs'], debug_path)
            image_files.append(image_path)
            spore_objs_list.append(result['spore_objs'])
            group_counts.append(result['count'])
        # store rows as (count, group_titr placeholder)
        group_titr = titr.calculate_titr(group_counts)
        groups_results[prefix] = [(c, group_titr) for c in group_counts]
        # write markdown for each image in the group using group titr
        for md_path, image_path, count in md_records:
            write_markdown_report(md_path, image_path, count, group_titr)

    # Excel report
    export_excel(groups_results, os.path.join(res_dir, 'report.xlsx'))

    # Optional CVAT export
    if args.export_cvat_zip and image_files:
        task_name = 'bees_task'
        make_cvat_export(task_name, image_files, spore_objs_list, res_dir)

if __name__ == '__main__':
    main() 