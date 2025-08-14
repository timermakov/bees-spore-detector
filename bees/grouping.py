import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple


GROUP_SUFFIXES = {"_1", "_2", "_3"}


def list_grouped_images(data_dir: str) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Scan directory for files named in format <prefix>_<1|2|3>.jpg and group them by prefix.
    Returns: (groups, errors)
        groups: {prefix: [path_1, path_2, path_3]} ordered by suffix number if present
        errors: list of human-readable error strings
    """
    errors: List[str] = []
    pattern = re.compile(r"^(?P<prefix>.+)_(?P<idx>[123])\.jpg$", re.IGNORECASE)
    candidates: Dict[str, Dict[str, str]] = defaultdict(dict)
    if not os.path.isdir(data_dir):
        return {}, [f"Папка не найдена: {data_dir}"]

    for fname in os.listdir(data_dir):
        if not fname.lower().endswith('.jpg'):
            continue
        match = pattern.match(fname)
        if not match:
            # report non-conforming file names succinctly
            errors.append(
                f"Неверное имя файла: {fname}. Ожидается формат <имя>_1.jpg, <имя>_2.jpg, <имя>_3.jpg"
            )
            continue
        prefix = match.group('prefix')
        idx = match.group('idx')
        if f"_{idx}" in candidates[prefix]:
            errors.append(
                f"Дубликат для {prefix}_{idx}.jpg. Должен быть ровно один файл для каждого из _1, _2, _3"
            )
            continue
        candidates[prefix][f"_{idx}"] = os.path.join(data_dir, fname)

    groups: Dict[str, List[str]] = {}
    for prefix, parts in candidates.items():
        missing = sorted(list(GROUP_SUFFIXES - set(parts.keys())))
        extra = sorted([k for k in parts.keys() if k not in GROUP_SUFFIXES])
        if missing:
            errors.append(
                f"Для группы '{prefix}' не хватает: {', '.join(missing)}. Должны быть файлы: {prefix}_1.jpg, {prefix}_2.jpg, {prefix}_3.jpg"
            )
            continue
        if extra:
            errors.append(
                f"Для группы '{prefix}' обнаружены лишние суффиксы: {', '.join(extra)}. Разрешены только _1, _2, _3"
            )
            continue
        groups[prefix] = [parts["_1"], parts["_2"], parts["_3"]]

    return groups, errors


