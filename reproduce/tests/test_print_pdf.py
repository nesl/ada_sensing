from __future__ import annotations

import csv
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from reproduce_in_ae.print_pdf import (
    PAGE_HEIGHT,
    PAGE_WIDTH,
    POINTS_PER_INCH,
    discover_images,
    fit_label,
    generate_print_pdfs,
    image_placement,
    inspect_image,
)


class PrintPdfTests(unittest.TestCase):
    def test_landscape_uses_sixty_percent_letter_width_and_is_centered(self) -> None:
        placement = image_placement(600, 400)
        self.assertAlmostEqual(placement.width / POINTS_PER_INCH, 5.1)
        self.assertAlmostEqual(placement.height / POINTS_PER_INCH, 3.4)
        self.assertAlmostEqual(placement.x + placement.width / 2.0, PAGE_WIDTH / 2.0)
        self.assertAlmostEqual(placement.y + placement.height / 2.0, PAGE_HEIGHT / 2.0)

    def test_square_and_portrait_keep_aspect_ratio_without_overflow(self) -> None:
        square = image_placement(100, 100)
        portrait = image_placement(100, 400)
        self.assertAlmostEqual(square.width, square.height)
        self.assertAlmostEqual(portrait.height / portrait.width, 4.0)
        self.assertLessEqual(portrait.height, 9.5 * POINTS_PER_INCH)
        self.assertGreaterEqual(portrait.y, 0.0)

    def test_discovery_is_relative_path_sorted_and_case_insensitive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b").mkdir()
            (root / "a").mkdir()
            Image.new("RGB", (10, 10)).save(root / "b" / "z.PNG")
            Image.new("RGB", (10, 10)).save(root / "a" / "x.JPEG")
            (root / "a" / "ignored.txt").write_text("not an image")
            paths = discover_images(root)
            self.assertEqual(
                [path.relative_to(root).as_posix() for path in paths],
                ["a/x.JPEG", "b/z.PNG"],
            )

    def test_exif_rotation_changes_layout_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "rotated.JPEG"
            image = Image.new("RGB", (40, 20), "red")
            exif = image.getexif()
            exif[274] = 6
            image.save(path, exif=exif)
            record = inspect_image(path, root)
            self.assertTrue(record.valid)
            self.assertEqual((record.width, record.height), (20, 40))

    def test_long_label_is_fitted(self) -> None:
        fitted, size = fit_label("class/" + "very-long-name-" * 100 + ".JPEG")
        self.assertIn("...", fitted)
        self.assertEqual(size, 5.0)

    @unittest.skipUnless(shutil.which("pdfinfo"), "pdfinfo is required")
    def test_pdf_manifest_splitting_transparency_and_invalid_skip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "images"
            root.mkdir()
            Image.new("RGB", (60, 40), "red").save(root / "a.JPEG")
            Image.new("L", (40, 40), 127).save(root / "b.jpeg")
            Image.new("RGBA", (30, 60), (0, 255, 0, 128)).save(root / "c.png")
            (root / "broken.jpg").write_bytes(b"not a JPEG")
            output = Path(directory) / "print.pdf"

            result = generate_print_pdfs(
                root,
                output,
                skip_invalid=True,
                max_pages_per_pdf=2,
            )

            self.assertEqual(result.discovered, 4)
            self.assertEqual(result.rendered, 3)
            self.assertEqual(result.skipped, 1)
            self.assertEqual(len(result.pdf_paths), 2)
            expected_pages = [2, 1]
            for path, pages in zip(result.pdf_paths, expected_pages):
                info = subprocess.run(
                    ["pdfinfo", str(path)],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
                self.assertRegex(info, rf"(?m)^Pages:\s+{pages}$")
                self.assertRegex(
                    info, re.compile(r"(?m)^Page size:\s+612 x 792 pts \(letter\)$")
                )

            with result.manifest_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertEqual([row["status"] for row in rows].count("rendered"), 3)
            self.assertEqual([row["status"] for row in rows].count("skipped"), 1)
            self.assertEqual(
                [row["global_page"] for row in rows if row["status"] == "rendered"],
                ["1", "2", "3"],
            )

    def test_invalid_is_fail_fast_before_writing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "images"
            root.mkdir()
            Image.new("RGB", (20, 20)).save(root / "valid.JPEG")
            (root / "broken.png").write_bytes(b"broken")
            output = Path(directory) / "print.pdf"
            with self.assertRaises(ValueError):
                generate_print_pdfs(root, output)
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".manifest.csv").exists())


if __name__ == "__main__":
    unittest.main()
