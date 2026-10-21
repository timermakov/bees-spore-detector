#!/usr/bin/env python3
"""Integration checks for the public bees package API."""

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from bees import (
    CVATExporter,
    ConfigurationLoader,
    ConfigurationManager,
    ConfigurationValidator,
    ExcelReporter,
    FilePairFinder,
    HierarchicalStructure,
    ImageInfo,
    ImageLoader,
    MarkdownReporter,
    MetadataLoader,
    ReportManager,
    TiterCalculator,
    XMLFormatter,
    create_config_manager,
    load_config_context,
)
from bees.config_loader import ConfigurationError
from bees.grouping import find_images, scan_folder_with_metadata, scan_hierarchy


def _repo_config_path() -> Path:
    return _REPO_ROOT / "config.yaml"


def _write_image(path: Path, size=(20, 10)) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color="white").save(path)


def _valid_config_text(data_dir: str = "data", results_dir: str = "results") -> str:
    return f"data_dir: {data_dir}\nresults_dir: {results_dir}\nmin_contour_area: 10\n"


class TestPublicApi:
    def test_imports(self):
        # Arrange / Act are the module import above.

        # Assert
        assert TiterCalculator is not None
        assert ConfigurationManager is not None
        assert HierarchicalStructure is not None

    def test_count_spores_is_callable(self):
        from bees import count_spores

        # Arrange
        spores = []

        # Act
        count = count_spores(spores)

        # Assert
        assert count == 0


class TestConfiguration:
    def test_repository_config_loads(self):
        # Arrange
        path = _repo_config_path()

        # Act
        config = create_config_manager(str(path))

        # Assert
        assert isinstance(config.get_param("data_dir"), str) and config.get_param("data_dir")
        assert isinstance(config.get_int_param("min_contour_area"), int)

    def test_loader_caches_until_reload(self, tmp_path):
        # Arrange
        path = tmp_path / "config.yaml"
        path.write_text(_valid_config_text("first"), encoding="utf-8")
        loader = ConfigurationLoader(path)

        # Act
        first = loader.load_config()
        path.write_text(_valid_config_text("second"), encoding="utf-8")
        cached = loader.load_config()
        reloaded = loader.reload_config()

        # Assert
        assert loader.get_config_path() == path
        assert first["data_dir"] == "first"
        assert cached["data_dir"] == "first"
        assert reloaded["data_dir"] == "second"
        assert loader.is_valid() is True

    def test_invalid_file_is_rejected(self, tmp_path):
        # Arrange
        missing = tmp_path / "missing.yaml"
        broken = tmp_path / "broken.yaml"
        broken.write_text(":\n  - [", encoding="utf-8")
        invalid = tmp_path / "invalid.yaml"
        invalid.write_text("data_dir: data\nresults_dir: results\nmin_contour_area: -1\n", encoding="utf-8")

        # Act / Assert
        with pytest.raises(ConfigurationError):
            ConfigurationLoader(missing)
        with pytest.raises(ConfigurationError):
            ConfigurationLoader(broken).load_config()
        assert ConfigurationLoader(invalid).is_valid() is False

    def test_validator_names_bad_parameters(self):
        # Arrange
        config = {
            "min_contour_area": 0,
            "canny_threshold1": -1,
            "min_spore_contour_length": 2,
        }

        # Act
        errors = ConfigurationValidator.validate_config(config)

        # Assert
        message = " ".join(errors)
        assert "data_dir" in message
        assert "min_contour_area" in message
        assert "canny_threshold1" in message
        assert "min_spore_contour_length" in message

    def test_manager_helpers_and_context(self, tmp_path):
        # Arrange
        path = tmp_path / "config.yaml"
        path.write_text(
            "data_dir: data\nresults_dir: results\nlabel: spore\nready: true\ncount: 3\n",
            encoding="utf-8",
        )

        # Act
        with load_config_context(path) as config:
            present = config.has_param("label")
            missing = config.validate_required_params(["label", "absent"])
            copied = config.get_all_params()
            text = config.get_string_param("label")
            flag = config.get_bool_param("ready")

        # Assert
        assert present is True
        assert missing == ["absent"]
        assert copied["count"] == 3
        assert text == "spore"
        assert flag is True

    def test_typed_getters_reject_bad_values(self):
        # Arrange
        manager = ConfigurationManager({"count": True, "ratio": "nope", "flag": "yes", "name": "  "})

        # Act / Assert
        with pytest.raises(ConfigurationError, match="count"):
            manager.get_int_param("count")
        with pytest.raises(ConfigurationError, match="ratio"):
            manager.get_float_param("ratio")
        with pytest.raises(ConfigurationError, match="flag"):
            manager.get_bool_param("flag")
        with pytest.raises(ConfigurationError, match="name"):
            manager.get_string_param("name")


class TestTiterCalculation:
    def test_sample_titer_uses_square_and_volume(self):
        # Arrange
        calculator = TiterCalculator(square_size=10, volume_factor=4.0)
        photos = [(80, 20, 20)]

        # Act
        titer, breakdown = calculator.calculate_sample_titer_safe(photos)

        # Assert
        assert calculator.calculate_sample_titer(photos) == pytest.approx(5.0)
        assert titer == pytest.approx(5.0)
        assert breakdown["total_spores"] == 80
        assert breakdown["n_photos"] == 1
        assert calculator.get_square_size() == 10

    def test_empty_photos_return_zero(self):
        # Arrange
        calculator = TiterCalculator()

        # Act
        titer = calculator.calculate_sample_titer([])
        safe_titer, breakdown = calculator.calculate_sample_titer_safe([])

        # Assert
        assert titer == 0.0
        assert safe_titer == 0.0
        assert breakdown["error"] == "No photos"

    def test_square_size_must_stay_positive(self):
        # Arrange
        calculator = TiterCalculator(square_size=10, volume_factor=2.0)

        # Act
        calculator.set_square_size(20)

        # Assert
        assert calculator.get_square_size() == 20
        assert calculator.square_area == 400
        with pytest.raises(ValueError):
            TiterCalculator(square_size=0)
        with pytest.raises(ValueError):
            TiterCalculator(volume_factor=0)
        with pytest.raises(ValueError):
            calculator.set_square_size(0)

    def test_t_tests_need_two_values_in_each_group(self):
        # Arrange
        calculator = TiterCalculator()

        # Act
        short_t, short_p = calculator.one_sample_ttest([1.0], 1.0)
        t_stat, p_value = calculator.one_sample_ttest([1.0, 3.0], 2.0)
        short_two = calculator.two_sample_ttest([1.0], [1.0, 2.0])
        two_t, two_p = calculator.two_sample_ttest([1.0, 2.0], [10.0, 12.0])

        # Assert
        assert (short_t, short_p) == (0.0, 1.0)
        assert isinstance(t_stat, float)
        assert 0.0 <= p_value <= 1.0
        assert short_two == (0.0, 1.0)
        assert isinstance(two_t, float)
        assert 0.0 <= two_p <= 1.0


class TestGrouping:
    def test_image_info_and_structure(self, tmp_path):
        # Arrange
        image = ImageInfo(path=tmp_path / "photo.png", width=20, height=10)
        structure = HierarchicalStructure()

        # Act
        structure.add_sample("Type_A", "Probe_1", "Sample_1", [image])
        rows = list(structure.iter_samples())

        # Assert
        assert image.area == 200
        assert image.stem == "photo"
        assert structure.n_types == 1
        assert structure.get_probe_names("Type_A") == ["Probe_1"]
        assert structure.get_sample_names("Type_A", "Probe_1") == ["Sample_1"]
        assert structure.get_probe_names("missing") == []
        assert rows == [("Type_A", "Probe_1", "Sample_1", [image])]

    def test_find_and_scan_images(self, tmp_path):
        # Arrange
        _write_image(tmp_path / "keep.png")
        (tmp_path / "notes.txt").write_text("skip", encoding="utf-8")
        (tmp_path / "broken.png").write_text("not-an-image", encoding="utf-8")

        # Act
        found = find_images(tmp_path, {".png"})
        scanned = scan_folder_with_metadata(tmp_path, {".png"})
        missing = find_images(tmp_path / "absent")

        # Assert
        assert [path.name for path in found] == ["broken.png", "keep.png"]
        assert [item.stem for item in scanned] == ["keep"]
        assert scanned[0].width == 20
        assert scanned[0].height == 10
        assert missing == []

    def test_scan_hierarchy(self, tmp_path):
        # Arrange
        _write_image(tmp_path / "Type_A" / "Probe_1" / "Sample_1" / "a.png")
        (tmp_path / "readme.txt").write_text("ignore", encoding="utf-8")

        # Act
        structure = scan_hierarchy(tmp_path, {".png"})

        # Assert
        assert structure.n_types == 1
        assert structure.get_sample_names("Type_A", "Probe_1") == ["Sample_1"]
        with pytest.raises(FileNotFoundError):
            scan_hierarchy(tmp_path / "absent")


class TestIoUtils:
    def test_image_loader(self, tmp_path):
        # Arrange
        image_path = tmp_path / "photo.png"
        _write_image(image_path)
        text_path = tmp_path / "notes.txt"
        text_path.write_text("nope", encoding="utf-8")

        # Act
        image = ImageLoader.load_image(image_path)

        # Assert
        assert image.size == (20, 10)
        assert ImageLoader.validate_image(image_path) is True
        assert ImageLoader.validate_image(text_path) is False
        with pytest.raises(FileNotFoundError):
            ImageLoader.load_image(tmp_path / "missing.png")
        with pytest.raises(ValueError):
            ImageLoader.load_image(text_path)

    def test_metadata_and_pairs(self, tmp_path):
        # Arrange
        xml_path = tmp_path / "photo.jpg_meta.xml"
        xml_path.write_text("<meta><name>photo</name></meta>", encoding="utf-8")
        _write_image(tmp_path / "photo.jpg")
        _write_image(tmp_path / "alone.jpg")
        broken = tmp_path / "broken.xml"
        broken.write_text("<meta>", encoding="utf-8")

        # Act
        root = MetadataLoader.load_metadata(xml_path)
        pairs = FilePairFinder.list_image_pairs(tmp_path)
        missing_dir = FilePairFinder.list_image_pairs(tmp_path / "absent")

        # Assert
        assert root.tag == "meta"
        assert pairs == [(tmp_path / "photo.jpg", xml_path)]
        assert missing_dir == []
        with pytest.raises(FileNotFoundError):
            MetadataLoader.load_metadata(tmp_path / "missing.xml")
        with pytest.raises(Exception):
            MetadataLoader.load_metadata(broken)

    def test_cvat_export_and_indent(self, tmp_path):
        import numpy as np

        # Arrange
        image_path = tmp_path / "photo.png"
        _write_image(image_path, size=(40, 40))
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        contour = np.stack(
            [20 + 8 * np.cos(angles), 20 + 6 * np.sin(angles)],
            axis=1,
        ).astype(np.float32)
        exporter = CVATExporter(label_name="spore")

        # Act
        meta, image_elem = exporter.export_image_elements(
            image_path,
            [np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float32), contour],
            image_id=3,
        )
        XMLFormatter.indent_xml(meta)

        # Assert
        assert image_elem.get("id") == "3"
        assert image_elem.get("name") == "photo.png"
        assert len(image_elem.findall("ellipse")) == 1
        assert meta.find("./task/labels/label/name").text == "spore"
        assert "\n" in (meta.text or "")
        with pytest.raises(FileNotFoundError):
            exporter.export_image_elements(tmp_path / "missing.png", [])


class TestReporting:
    def test_markdown_and_excel_reports(self, tmp_path):
        # Arrange
        markdown = MarkdownReporter(tmp_path / "reports")
        excel = ExcelReporter(tmp_path / "reports")
        groups = {"A": [(10, 1.5), (12, 1.5)]}

        # Act
        image_report = markdown.write_image_report("photo.jpg", 10, 1.5)
        group_report = markdown.write_group_report("A", [10, 12], 1.5)
        workbook = excel.export_results(groups, "report.xlsx")

        # Assert
        image_text = image_report.read_text(encoding="utf-8")
        group_text = group_report.read_text(encoding="utf-8")
        assert "photo.jpg" in image_text
        assert "Spore Count:" in image_text
        assert "Group:** A" in group_text or "**Group:** A" in group_text
        assert "Total Spores:** 22" in group_text or "**Total Spores:** 22" in group_text
        assert workbook.exists()
        assert workbook.suffix == ".xlsx"

    def test_report_manager_writes_every_kind(self, tmp_path):
        # Arrange
        manager = ReportManager(tmp_path / "reports")
        groups = {"A": [(4, 2.0)]}
        images = {"photo.jpg": {"count": 4, "titer": 2.0, "debug_images": ["debug.jpg"]}}

        # Act
        reports = manager.generate_all_reports(groups, images)

        # Assert
        assert reports["excel"].exists()
        assert reports["group_reports"]["A"].exists()
        assert reports["individual_reports"]["photo.jpg"]["report"].exists()
        assert reports["individual_reports"]["photo.jpg"]["debug_images"] == ["debug.jpg"]


def main() -> int:
    return pytest.main([__file__, "-v"])


if __name__ == "__main__":
    sys.exit(main())
