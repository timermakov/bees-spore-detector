import os
from typing import Dict, List, Tuple

from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from openpyxl.styles import Alignment

from bees.titr import calculate_titr


def write_markdown_report(md_path: str, image_name: str, count: int, titr_value: float, analysis_square_size: int = None, total_count: int = None) -> None:
    content = (
        f"# Результаты анализа изображения {os.path.basename(image_name)}\n\n"
        f"- Количество спор: {count}\n"
        f"- Титр (млн спор/мл): {titr_value:.2f}\n"
    )
    if total_count is not None:
        content += f"- Общее количество спор на фотографии: {total_count}\n"
    if analysis_square_size:
        content += f"- Зона анализа: квадрат {analysis_square_size}x{analysis_square_size} пикселей в центре изображения\n"
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(content)


def export_excel(groups_results: Dict[str, List[Tuple[int, int, float]]], output_xlsx: str, analysis_square_size: int = None) -> None:
    """
    groups_results: {prefix: [(count_inside, total_count, group_titr), ...]}
    analysis_square_size: (не используется, оставлен для совместимости и потенциального заголовка)
    """
    wb = Workbook()
    ws = wb.active
    ws.title = 'Отчет'
    
    # Header
    ws.append(["Проба", "Сэмпл", "Количество спор (внутри квадрата)", "Общее количество спор на фотографии", "Титр"])
    # Content
    start_row = 2
    for prefix, rows in groups_results.items():
        group_titr = calculate_titr([c_in for c_in, _, _ in rows])
        for idx, (count_inside, total_count, _) in enumerate(rows, start=1):
            ws.append([prefix, idx, count_inside, total_count, group_titr])
        # Merge 'Проба' and 'Титр' cells for the rows of the group
        end_row = start_row + len(rows) - 1
        if end_row >= start_row:
            ws.merge_cells(start_row=start_row, start_column=1, end_row=end_row, end_column=1)
            ws.merge_cells(start_row=start_row, start_column=5, end_row=end_row, end_column=5)
            # Center alignment for merged cells
            ws.cell(row=start_row, column=1).alignment = Alignment(vertical='center', horizontal='center')
            ws.cell(row=start_row, column=5).alignment = Alignment(vertical='center', horizontal='center')
        start_row = end_row + 1
    # Column widths
    widths = [20, 10, 30, 34, 20]
    for col_idx, width in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(col_idx)].width = width
    os.makedirs(os.path.dirname(output_xlsx), exist_ok=True)
    wb.save(output_xlsx)


