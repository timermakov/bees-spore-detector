"""The titer count keeps spores whose center is inside the analysis square."""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bees.analysis_zone import (
    analysis_square_bounds,
    point_inside_analysis_square,
    split_by_analysis_square,
)
from bees.config_loader import create_config_manager
from bees.spore_analysis_pipeline import SporeAnalysisPipeline
from bees.titer import TiterCalculator
from bees.yolo.detector import Detection
from bees.yolo.hierarchical_analysis import HierarchicalAnalyzer, _inside_detection_count


def _circle(cx: float, cy: float, radius: float = 6.0) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    return np.stack(
        [cx + radius * np.cos(angles), cy + radius * np.sin(angles)],
        axis=1,
    ).astype(np.float32)


def _write_config(directory: Path, square_size: int = 40) -> Path:
    path = directory / "config.yaml"
    path.write_text(
        "\n".join(
            [
                f"data_dir: {directory / 'data'}",
                f"results_dir: {directory / 'results'}",
                "min_contour_area: 1",
                "max_contour_area: 10000",
                "min_ellipse_area: 1",
                "max_ellipse_area: 10000",
                "canny_threshold1: 40",
                "canny_threshold2: 120",
                "min_spore_contour_length: 5",
                "intensity_threshold: 50",
                "min_ellipse_axis_ratio: 0.45",
                "max_ellipse_axis_ratio: 0.92",
                "min_ellipse_eccentricity: 0.45",
                "max_ellipse_eccentricity: 0.95",
                "min_center_distance: 6.0",
                f"analysis_square_size: {square_size}",
                "analysis_square_line_width: 2",
                "volume_factor: 4.0",
                "morphology_confidence: 0.7",
                "morphology_padding_ratio: 0.15",
                "morphology_montage_columns: 4",
                "microns_per_pixel: null",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


class TestAnalysisSquare:
    def test_bounds_are_centered_and_clipped(self):
        # Act
        bounds = analysis_square_bounds(100, 80, 40)

        # Assert
        assert bounds == (30, 20, 70, 60)

    def test_non_positive_square_covers_the_image(self):
        # Act
        inside = point_inside_analysis_square(0, 0, 100, 80, 0)

        # Assert
        assert inside is True

    def test_split_keeps_the_center_and_drops_the_corner(self):
        # Arrange
        items = ["center", "corner"]
        centers = [(50.0, 40.0), (1.0, 1.0)]

        # Act
        inside, outside = split_by_analysis_square(items, centers, 100, 80, 40)

        # Assert
        assert inside == ["center"]
        assert outside == ["corner"]


class TestPipelineCount:
    def test_opencv_count_uses_the_square(self, tmp_path, monkeypatch):
        # Arrange
        config_path = _write_config(tmp_path, square_size=40)
        data_dir = tmp_path / "data"
        image_path = data_dir / "photo.png"
        image_path.parent.mkdir(parents=True)
        Image.new("RGB", (100, 100), color="white").save(image_path)
        image_path.with_suffix(".png_meta.xml").write_text("<meta/>", encoding="utf-8")
        manager = create_config_manager(str(config_path))
        pipeline = SporeAnalysisPipeline(manager, str(data_dir), str(tmp_path / "results"), use_yolo=False)

        def fake_detect(self, image, **params):
            return [_circle(50, 50), _circle(5, 5)]

        monkeypatch.setattr(
            "bees.opencv.image_proc.SporeDetectionPipeline.detect_spores",
            fake_detect,
        )

        # Act
        result = pipeline.process_image(str(image_path), str(image_path.with_suffix(".png_meta.xml")))

        # Assert
        assert result["count"] == 1
        assert result["count_outside_square"] == 1
        assert len(result["spore_objects"]) == 1

    def test_yolo_count_uses_inside_detections(self, tmp_path):
        # Arrange
        config_path = _write_config(tmp_path)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        image_path = data_dir / "photo.png"
        Image.new("RGB", (100, 100), color="white").save(image_path)
        meta = image_path.with_suffix(".png_meta.xml")
        meta.write_text("<meta/>", encoding="utf-8")
        manager = create_config_manager(str(config_path))
        pipeline = SporeAnalysisPipeline(manager, str(data_dir), str(tmp_path / "results"), use_yolo=False)
        inside = Detection((40, 40, 60, 60), confidence=0.9)
        outside = Detection((0, 0, 4, 4), confidence=0.9)

        class FakeCounter:
            def count_with_detections(self, path):
                return {"total": 2, "inside": 1, "outside": 1}, [inside], [outside]

            def visualize(self, path, output_path):
                return None

        pipeline.use_yolo = True
        pipeline.yolo_counter = FakeCounter()

        # Act
        result = pipeline._process_image_yolo(str(image_path), Image.open(image_path), None, tmp_path / "debug")

        # Assert
        assert result["count"] == 1
        assert result["count_outside_square"] == 1
        assert len(result["spore_objects"]) == 1
        assert len(pipeline.morphology_rows) == 1
        assert pipeline.morphology_rows[0].source == "bbox_ellipse"


class TestHierarchyCount:
    def test_sample_folder_counts_centers_inside_the_square(self, tmp_path):
        # Arrange
        sample = tmp_path / "Sample_1"
        sample.mkdir()
        Image.new("RGB", (100, 100), color="white").save(sample / "a.png")
        inside = Detection((40, 40, 60, 60), confidence=0.9)
        outside = Detection((0, 0, 4, 4), confidence=0.9)

        class FakeDetector:
            def detect(self, path):
                return [inside, outside]

        analyzer = HierarchicalAnalyzer.__new__(HierarchicalAnalyzer)
        analyzer.use_yolo = True
        analyzer.use_sahi = False
        analyzer.titer_calculator = TiterCalculator(square_size=40, volume_factor=4.0)
        analyzer.detector = FakeDetector()

        # Act
        result = analyzer.analyze_sample_folder(sample, "Probe_1", "Type_A")

        # Assert
        assert result.photo_data == [(1, 100, 100)]
        assert _inside_detection_count([inside, outside], 100, 100, 40) == 1

    def test_opencv_sample_folder_records_the_inside_count(self, tmp_path, monkeypatch):
        # Arrange
        config_path = _write_config(tmp_path, square_size=40)
        sample = tmp_path / "Sample_1"
        sample.mkdir()
        Image.new("RGB", (100, 100), color="white").save(sample / "a.png")
        analyzer = HierarchicalAnalyzer(config_path=str(config_path), use_yolo=False)

        def fake_detect(self, image, **params):
            return [_circle(50, 50), _circle(5, 5)]

        monkeypatch.setattr(
            "bees.opencv.image_proc.SporeDetectionPipeline.detect_spores",
            fake_detect,
        )

        # Act
        result = analyzer.analyze_sample_folder(sample, "Probe_1", "Type_A")

        # Assert
        assert result.photo_data == [(1, 100, 100)]
        assert analyzer.opencv_params["analysis_square_size"] == 40
        assert len(analyzer.morphology_rows) == 1
        assert analyzer.morphology_rows[0].population == "opencv_accepted"
        assert analyzer.morphology_rows[0].source == "contour"


def test_sahi_dict_uses_bbox_center():
    # Arrange
    detections = [{"bbox": [40, 40, 60, 60]}, {"bbox": [0, 0, 4, 4]}]

    # Act
    count = _inside_detection_count(detections, 100, 100, 40)

    # Assert
    assert count == 1
