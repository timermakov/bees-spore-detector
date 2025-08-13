import os
from typing import Dict, List, Tuple

from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from openpyxl.styles import Alignment

from bees.titr import calculate_titr


def write_markdown_report(md_path: str, image_name: str, count: int, titr_value: float) -> None:
    content = (
        f"# Результаты анализа изображения {os.path.basename(image_name)}\n\n"
        f"- Количество спор: {count}\n"
        f"- Титр (млн спор/мл): {titr_value:.2f}\n"
    )
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(content)


def export_excel(groups_results: Dict[str, List[Tuple[int, float]]], output_xlsx: str) -> None:
    """
    groups_results: {prefix: [(count1, titr_group), (count2, titr_group), (count3, titr_group)]}
    """
    wb = Workbook()
    ws = wb.active
    ws.title = 'Отчет'
    # Header
    ws.append(["Проба", "Сэмпл", "Количество спор", "Титр"])
    # Content
    start_row = 2
    for prefix, rows in groups_results.items():
        group_titr = calculate_titr([c for c, _ in rows])
        for idx, (count, _) in enumerate(rows, start=1):
            ws.append([prefix, idx, count, group_titr])
        # Merge 'Проба' and 'Титр' cells for the 3 rows
        end_row = start_row + len(rows) - 1
        if end_row >= start_row:
            ws.merge_cells(start_row=start_row, start_column=1, end_row=end_row, end_column=1)
            ws.merge_cells(start_row=start_row, start_column=4, end_row=end_row, end_column=4)
            # Center alignment for merged cells
            ws.cell(row=start_row, column=1).alignment = Alignment(vertical='center', horizontal='center')
            ws.cell(row=start_row, column=4).alignment = Alignment(vertical='center', horizontal='center')
        start_row = end_row + 1
    # Column widths
    widths = [20, 10, 20, 20]
    for col_idx, width in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(col_idx)].width = width
    os.makedirs(os.path.dirname(output_xlsx), exist_ok=True)
    wb.save(output_xlsx)


