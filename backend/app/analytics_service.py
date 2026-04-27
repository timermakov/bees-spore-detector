import sys
from pathlib import Path
from statistics import mean, pstdev

from scipy import stats
from sqlalchemy.orm import Session

from app import crud, models
from app.config import get_settings
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bees.titer import TiterCalculator


def _calc_p_value(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    _t, p = stats.ttest_1samp(values, 0)
    return float(p)


def analyze_probe(db: Session, probe_id: int, mode: str | None = None) -> models.ProbeResult:
    # Import lazily so API startup does not fail if OpenCV libs are missing.
    from bees.opencv import SporeDetectionPipeline

    probe = crud.get_by_id(db, models.Probe, probe_id)
    if not probe:
        raise ValueError("Probe not found")

    settings = get_settings()
    method = mode or settings.analysis_mode
    titer_calculator = TiterCalculator()

    sample_dirs: list[Path] = []
    for sample in probe.samples:
        if not sample.images:
            continue
        first_image = Path(sample.images[0].file_path)
        sample_dirs.append(first_image.parent)

    if not sample_dirs:
        raise ValueError("No images found for probe")

    detector = SporeDetectionPipeline()
    params = {
        "min_contour_area": 45,
        "max_contour_area": 650,
        "min_ellipse_area": 100,
        "max_ellipse_area": 360,
        "canny_threshold1": 40,
        "canny_threshold2": 120,
        "min_spore_contour_length": 7,
        "intensity_threshold": 36,
        "analysis_square_size": 780,
        "analysis_square_line_width": 2,
    }

    # Collect titers per sample using the new hierarchical API
    all_titers: list[float] = []
    for sample_dir in sample_dirs:
        photo_data: list[tuple[int, int, int]] = []
        for image_path in sorted(sample_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            try:
                from PIL import Image

                image = Image.open(image_path)
                spores = detector.detect_spores(image, **params)
                width, height = image.size
                photo_data.append((len(spores), width, height))
            except Exception:
                continue

        if photo_data:
            titer = titer_calculator.calculate_sample_titer(photo_data)
            all_titers.append(float(titer))

    if not all_titers:
        # Fallback keeps API stable even on empty/failed detections.
        all_titers = [0.0]

    mean_titer = float(mean(all_titers))
    std_titer = float(pstdev(all_titers)) if len(all_titers) > 1 else 0.0
    p_value = _calc_p_value(all_titers)

    return crud.upsert_probe_result(
        db=db,
        probe_id=probe_id,
        mean_titer=mean_titer,
        std_titer=std_titer,
        n_measurements=len(all_titers),
        p_value=p_value,
        method=method.lower(),
    )
