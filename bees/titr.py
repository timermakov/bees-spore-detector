def calculate_titr(spores, volume_factor=12):
    """
    Calculate titr (million spores per ml) for a group of three samples.

    spores: int | list[int]
        If a single integer is provided, it is treated as a count for one sample.
        If a list/tuple is provided, values are summed as (x + y + z).
    volume_factor: int | float
        Divisor according to the Goryaev chamber method (default 12).
    """
    if spores is None:
        return 0.0
    try:
        # Iterable of counts (x, y, z, ...)
        total_spores = sum(spores)  # type: ignore[arg-type]
    except TypeError:
        # Single integer
        total_spores = int(spores)  # type: ignore[arg-type]
    return float(total_spores) / float(volume_factor)