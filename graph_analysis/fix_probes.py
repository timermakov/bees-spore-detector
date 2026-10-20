"""
fix_probes.py — автоматическая корректировка названий проб разведения.

Логика:
  1. Для каждой (версия, группа) считается среднее control.
  2. Для каждой строки разведения вычисляется теоретический титр
     = control / dilution.
  3. Считается ошибка |факт − теоретическое| / теоретическое отдельно
     для CV (normalized) и Hand Analysis.
  4. Внутри каждой кратности (:10, :20, :50) серии разделяются на
     "точные" (малая ошибка) и "ошибочные" (большая).
  5. Точные получают большую аликвоту (50+...),
     ошибочные — малую аликвоту (1+..., 10+...).

Использование:
    python -m graph_analysis.fix_probes \
        -i results/.../concentration.xlsx \
        -s "concentration" \
        -o results/.../concentration_fixed.xlsx \
        --groups "Nosema bombycis" "Paranosema locustae" \
        --error-threshold 0.5
"""

import argparse
import re
from pathlib import Path

import pandas as pd

from .config import COLS, COL_ALIASES
from .data_loader import load_excel, ensure_numeric, normalize_columns


# ============ Параметры аликвот ============

# Для каждой кратности: какая аликвота "точная" (большая),
# какая "ошибочная" (малая).
ALIQUOT_PAIRS = {
    ":10": {"good": "(50+450)", "bad": "(1+9)"},
    ":20": {"good": "(50+50)",  "bad": "(10+10)"},
    ":50": {"good": "(50+75)",  "bad": "(10+15)"},
}

CONTROL_LABELS = {"control", "контроль"}


# ============ Разбор и сборка имени пробы ============

_PROBE_RE = re.compile(r":\s*(\d+)\s*\(\s*(\d+)\s*\+\s*(\d+)\s*\)")
_PROBE_SIMPLE_RE = re.compile(r":\s*(\d+)\s*$")


def parse_probe(probe_str):
    """
    Разбирает имя пробы.
    Возвращает dict: {dilution, a, b, aliquot}
      ':10 (1+9)'    → {'dilution': 10, 'a': 1,  'b': 9,  'aliquot': '(1+9)'}
      ':20 (50+50)'  → {'dilution': 20, 'a': 50, 'b': 50, 'aliquot': '(50+50)'}
      ':100'         → {'dilution': 100, 'a': None, 'b': None, 'aliquot': None}
      'control'      → {'dilution': 1, 'a': None, 'b': None, 'aliquot': None}
    """
    s = str(probe_str).strip().lower()
    if s in {lbl.lower() for lbl in CONTROL_LABELS}:
        return {"dilution": 1, "a": None, "b": None, "aliquot": None}

    m = _PROBE_RE.match(s)
    if m:
        return {
            "dilution": int(m.group(1)),
            "a": int(m.group(2)),
            "b": int(m.group(3)),
            "aliquot": f"({m.group(2)}+{m.group(3)})",
        }

    m = _PROBE_SIMPLE_RE.match(s)
    if m:
        return {"dilution": int(m.group(1)), "a": None, "b": None, "aliquot": None}

    return {"dilution": None, "a": None, "b": None, "aliquot": None}


def build_probe_name(dilution, aliquot=None):
    if dilution is None:
        return None
    if dilution == 1:
        return "control"
    if aliquot:
        return f":{dilution} {aliquot}"
    return f":{dilution}"


# ============ Расчёт ошибок ============

def compute_control_means(df, cv_col, hand_col):
    """
    Для каждой (версия, группа) считает среднее CV и Hand по контролю.
    Возвращает dict: {(версия, группа): (mean_cv, mean_hand)}
    """
    result = {}
    for (v, g), sub in df.groupby(["версия", "группа"], dropna=False):
        ctrl = sub[sub["_dilution"] == 1]
        if ctrl.empty:
            continue
        m_cv = pd.to_numeric(ctrl[cv_col], errors="coerce").mean()
        m_hand = pd.to_numeric(ctrl[hand_col], errors="coerce").mean()
        if pd.notna(m_cv) and pd.notna(m_hand):
            result[(v, g)] = (m_cv, m_hand)
    return result


def add_error_columns(df, cv_col, hand_col):
    """
    Добавляет колонки _error_cv, _error_hand.
    Ошибка = |факт − теоретическое| / теоретическое.
    Теоретическое = control / dilution.
    """
    df = df.copy()
    df["_error_cv"] = None
    df["_error_hand"] = None

    ctrl_map = compute_control_means(df, cv_col, hand_col)

    for idx, row in df.iterrows():
        key = (row["версия"], row["группа"])
        if key not in ctrl_map:
            continue
        ctrl_cv, ctrl_hand = ctrl_map[key]
        dilution = row["_dilution"]
        if dilution is None or dilution <= 1:
            continue

        theo_cv = ctrl_cv / dilution
        theo_hand = ctrl_hand / dilution

        cv_val = pd.to_numeric(row[cv_col], errors="coerce")
        hand_val = pd.to_numeric(row[hand_col], errors="coerce")

        if pd.notna(cv_val) and theo_cv > 0:
            df.at[idx, "_error_cv"] = abs(cv_val - theo_cv) / theo_cv
        if pd.notna(hand_val) and theo_hand > 0:
            df.at[idx, "_error_hand"] = abs(hand_val - theo_hand) / theo_hand

    return df


# ============ Основная логика ============

def _classify_probe_rows(probe_rows, threshold):
    """
    Для набора строк одной пробы возвращает маску "плохих".
    Плохая — если хотя бы по одной метрике ошибка > threshold.
    """
    err_cv = pd.to_numeric(probe_rows["_error_cv"], errors="coerce")
    err_hand = pd.to_numeric(probe_rows["_error_hand"], errors="coerce")
    combined = pd.concat([err_cv, err_hand], axis=1).max(axis=1)
    return combined > threshold


def fix_probes(df, error_threshold=0.5):
    """
    Основная функция. Принимает df (уже с _dilution),
    возвращает (fixed_df, report_df).
    """
    df = df.copy()
    report_rows = []

    # Смотрим каждую (версия, группа, кратность) отдельно
    for (v, g, d), sub in df.groupby(["версия", "группа", "_dilution"], dropna=False):
        if d is None or d <= 1:
            continue

        d_key = f":{int(d)}"
        if d_key not in ALIQUOT_PAIRS:
            continue

        good_aliq = ALIQUOT_PAIRS[d_key]["good"]
        bad_aliq = ALIQUOT_PAIRS[d_key]["bad"]

        # Смотрим каждую пробу отдельно (могут быть разные пробы
        # одной кратности — :10 (1+9) и :10 (50+450))
        for probe_name, probe_rows in sub.groupby("проба", dropna=False):
            if len(probe_rows) == 0:
                continue

            parsed = parse_probe(probe_name)
            current_aliquot = parsed["aliquot"]

            # Если аликвота не из известного списка — пропускаем
            if current_aliquot not in (good_aliq, bad_aliq):
                continue

            # Определяем "плохие" строки
            is_bad = _classify_probe_rows(probe_rows, error_threshold)
            n_bad = int(is_bad.sum())
            n_total = len(probe_rows)

            if n_bad == 0:
                # Все точные. Если проба названа как "плохая" —
                # переименовать в "хорошую".
                if current_aliquot == bad_aliq:
                    new_name = build_probe_name(int(d), good_aliq)
                    df.loc[probe_rows.index, "проба"] = new_name
                    report_rows.append({
                        "версия": v, "группа": g, "кратность": d,
                        "старое_имя": probe_name, "новое_имя": new_name,
                        "n_всего": n_total, "n_плохих": 0,
                        "действие": "переименовано в 'хорошую' (все точные)",
                    })
                else:
                    report_rows.append({
                        "версия": v, "группа": g, "кратность": d,
                        "старое_имя": probe_name, "новое_имя": probe_name,
                        "n_всего": n_total, "n_плохих": 0,
                        "действие": "не изменено (уже 'хорошая', все точные)",
                    })

            elif n_bad == n_total:
                # Все плохие. Переименовать в "плохую".
                if current_aliquot == good_aliq:
                    new_name = build_probe_name(int(d), bad_aliq)
                    df.loc[probe_rows.index, "проба"] = new_name
                    report_rows.append({
                        "версия": v, "группа": g, "кратность": d,
                        "старое_имя": probe_name, "новое_имя": new_name,
                        "n_всего": n_total, "n_плохих": n_bad,
                        "действие": "переименовано в 'плохую' (все ошибочные)",
                    })
                else:
                    report_rows.append({
                        "версия": v, "группа": g, "кратность": d,
                        "старое_имя": probe_name, "новое_имя": probe_name,
                        "n_всего": n_total, "n_плохих": n_bad,
                        "действие": "не изменено (уже 'плохая', все ошибочные)",
                    })

            else:
                # Смешанные: часть точных, часть плохих.
                good_idx = probe_rows.index[~is_bad]
                bad_idx = probe_rows.index[is_bad]

                good_name = build_probe_name(int(d), good_aliq)
                bad_name = build_probe_name(int(d), bad_aliq)

                df.loc[good_idx, "проба"] = good_name
                df.loc[bad_idx, "проба"] = bad_name

                report_rows.append({
                    "версия": v, "группа": g, "кратность": d,
                    "старое_имя": probe_name,
                    "новое_имя": f"{good_name} | {bad_name}",
                    "n_всего": n_total, "n_плохих": n_bad,
                    "действие": (
                        f"разделено: {len(good_idx)} → {good_name}, "
                        f"{len(bad_idx)} → {bad_name}"
                    ),
                })

    return df, pd.DataFrame(report_rows)


# ============ CLI ============

def parse_args():
    p = argparse.ArgumentParser(
        description="Автокорректировка названий проб разведения."
    )
    p.add_argument(
        "-i", "--input", required=True,
        help="Путь к xlsx с концентрациями",
    )
    p.add_argument(
        "-s", "--sheet", default="concentration",
        help="Имя листа (по умолчанию: concentration)",
    )
    p.add_argument(
        "-o", "--output", default="concentration_fixed.xlsx",
        help="Куда сохранить исправленный xlsx",
    )
    p.add_argument(
        "--report", default="fix_probes_report.csv",
        help="Куда сохранить отчёт об изменениях",
    )
    p.add_argument(
        "--groups", nargs="+", default=None,
        help="Группы для обработки (например: Nosema bombycis Paranosema locustae). "
             "Если не указано — обрабатываются все.",
    )
    p.add_argument(
        "--error-threshold", type=float, default=0.5,
        help="Порог ошибки: |факт − теор| / теор. "
             "Всё, что выше — 'плохая' серия (по умолчанию: 0.5 = 50%%).",
    )
    p.add_argument(
        "--value-mode", choices=["normalized", "raw"], default="normalized",
        help="Какую CV-колонку использовать для расчёта ошибки "
             "(по умолчанию: normalized)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Читаю: {args.input} (лист: {args.sheet})")
    df = load_excel(args.input, args.sheet)
    df = normalize_columns(df)

    # Убираем пустые строки
    df = df.dropna(how="all").reset_index(drop=True)

    # Числовая нормализация
    df = ensure_numeric(df)

    # Какую CV-колонку использовать для расчёта
    if args.value_mode == "normalized":
        cv_col = COLS["fixed"]["cv_norm"]
    else:
        cv_col = COLS["fixed"]["cv_raw"]
    hand_col = COLS["fixed"]["hand"]

    if cv_col not in df.columns:
        raise SystemExit(f"Колонка '{cv_col}' не найдена в файле")
    if hand_col not in df.columns:
        raise SystemExit(f"Колонка '{hand_col}' не найдена в файле")

    # Фильтр по группам
    if args.groups:
        before = len(df)
        df = df[df["группа"].isin(args.groups)].copy()
        print(f"Отфильтровано по группам: {before} → {len(df)} строк")
        print(f"  Группы: {args.groups}")

    if df.empty:
        raise SystemExit("После фильтрации по группам данных нет.")

    # Парсим кратность разведения
    df["_dilution"] = df["проба"].apply(lambda x: parse_probe(x)["dilution"])

    # Считаем ошибки
    print("Считаю ошибки относительно теоретического разведения...")
    df = add_error_columns(df, cv_col, hand_col)

    # Переименовываем
    print(f"Порог ошибки: {args.error_threshold:.0%}")
    fixed_df, report_df = fix_probes(df, error_threshold=args.error_threshold)

    # Убираем вспомогательные колонки перед сохранением
    drop_cols = ["_dilution", "_error_cv", "_error_hand"]
    fixed_df = fixed_df.drop(columns=[c for c in drop_cols if c in fixed_df.columns])

    # Сохраняем результат
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fixed_df.to_excel(out_path, sheet_name=args.sheet, index=False)
    print(f"\n✓ Исправленный файл: {out_path}")

    # Отчёт
    if not report_df.empty:
        report_df.to_csv(args.report, index=False, encoding="utf-8-sig")
        print(f"✓ Отчёт об изменениях: {args.report}")
        print("\nИзменения:")
        print(report_df.to_string(index=False))
    else:
        print("\nИзменений не требуется — названия уже корректны.")


if __name__ == "__main__":
    main()