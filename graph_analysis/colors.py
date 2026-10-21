"""Реестр цветов проб. Один и тот же sample -> один и тот же цвет
в пределах всего процесса и на всех графиках."""

_PALETTE = [
    "#1f77b4",  # синий
    "#ff7f0e",  # оранжевый
    "#2ca02c",  # зелёный
    "#d62728",  # красный
    "#9467bd",  # фиолетовый
    "#8c564b",  # коричневый
    "#e377c2",  # розовый
    "#7f7f7f",  # серый
    "#bcbd22",  # оливковый
    "#17becf",  # циан
    "#aec7e8",  # светло-синий
    "#ffbb78",  # светло-оранжевый
    "#98df8a",  # светло-зелёный
    "#ff9896",  # светло-красный
    "#c5b0d5",  # светло-фиолетовый
]

# Явные предпочтения для известных проб (необязательно)
_PREFERRED = {
    "control": "#1f77b4",
    "Del": "#ff7f0e",
    "delta": "#ff7f0e",
    "DcH": "#2ca02c",
    "delta_chimeric": "#98df8a",
    "ligase": "#d62728",
    "ligase_chimeric": "#9467bd",
    "PTP3+CS": "#8c564b",
    "PTP+CS": "#8c564b",
    "PTP3+PEI": "#e377c2",
    "PTP+PEI": "#e377c2",
}


def _canonical_probe(name: str) -> str:
    """Map hand-analysis probe labels onto the base probe color."""
    s = str(name)
    suffix = "_hand_analysis"
    if s.endswith(suffix):
        return s[: -len(suffix)]
    return s

_COLORS = {}


def register_samples(samples):
    """
    Инициализирует реестр цветов из полного списка проб.
    Вызывать один раз в начале запуска на полном df.
    """
    if _COLORS:
        return _COLORS

    samples = [str(s) for s in samples]

    for s in samples:
        key = _canonical_probe(s)
        if key in _PREFERRED:
            _COLORS[s] = _PREFERRED[key]
        elif s in _PREFERRED:
            _COLORS[s] = _PREFERRED[s]

    used = set(_COLORS.values())
    free = [c for c in _PALETTE if c not in used]
    idx = 0
    for s in samples:
        if s in _COLORS:
            continue
        if idx < len(free):
            _COLORS[s] = free[idx]
            idx += 1
        else:
            _COLORS[s] = _PALETTE[idx % len(_PALETTE)]
            idx += 1

    return _COLORS


def get_sample_color(sample):
    s = str(sample)
    if s in _COLORS:
        return _COLORS[s]
    key = _canonical_probe(s)
    if key in _PREFERRED:
        _COLORS[s] = _PREFERRED[key]
        return _COLORS[s]
    if s in _PREFERRED:
        _COLORS[s] = _PREFERRED[s]
        return _COLORS[s]
    _COLORS[s] = "#7f7f7f"
    return _COLORS[s]


def sample_color_map(samples=None):
    if samples is None:
        return dict(_COLORS)
    return {str(s): get_sample_color(s) for s in samples}