from __future__ import annotations

import csv
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageChops

from reproduce_in_ae.print_bitmap_pdf import (
    centered_paste_box,
    generate_bitmap_print_pdf,
    letter_canvas_size,
    load_selections,
)


SELECTION_FIELDS = [
    "sample_id",
    "original_path",
    "rendered_path",
    "pdf_page",
    "class_index",
    "wnid",
    "class_name",
    "source_relative_path",
]


def write_selection_csv(root: Path, names: list[str]) -> Path:
    labels = root / "labels.csv"
    with labels.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SELECTION_FIELDS)
        writer.writeheader()
        for index, name in enumerate(names):
            writer.writerow(
                {
                    "sample_id": Path(name).stem,
                    "original_path": f"original/{name}",
                    "rendered_path": "",
                    "pdf_page": index + 1,
                    "class_index": index,
                    "wnid": f"n{index:08d}",
                    "class_name": f"class_{index}",
                    "source_relative_path": f"n{index:08d}/{name}",
                }
            )
    return labels


class PrintBitmapPdfTests(unittest.TestCase):
    def test_letter_canvas_and_integer_centering(self) -> None:
        self.assertEqual(letter_canvas_size(600), (5100, 6600))
        self.assertEqual(
            centered_paste_box((5100, 6600), (500, 375)),
            (2300, 3112, 2800, 3487),
        )

    def test_source_larger_than_canvas_fails_instead_of_resizing(self) -> None:
        with self.assertRaisesRegex(ValueError, "resizing is forbidden"):
            centered_paste_box((612, 792), (613, 100))

    def test_selection_csv_order_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original"
            original.mkdir()
            names = ["z.JPEG", "a.JPEG"]
            for name in names:
                Image.new("RGB", (10, 10), "white").save(original / name)
            labels = write_selection_csv(root, names)
            selections = load_selections(labels)
            self.assertEqual(
                [selection.metadata["sample_id"] for selection in selections],
                ["z", "a"],
            )

    @unittest.skipUnless(
        shutil.which("pdfinfo") and shutil.which("pdfimages"),
        "Poppler pdfinfo and pdfimages are required",
    )
    def test_pdf_contains_lossless_full_page_bitmap_and_exact_source_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original"
            original.mkdir()
            source_path = original / "sample.JPEG"
            image = Image.new("RGB", (11, 9))
            for y in range(image.height):
                for x in range(image.width):
                    image.putpixel((x, y), (x * 17, y * 23, (x + y) * 11))
            image.save(source_path, quality=95, subsampling=0)
            labels = write_selection_csv(root, [source_path.name])
            output = root / "bitmap.pdf"

            result = generate_bitmap_print_pdf(labels, output, dpi=72)
            self.assertEqual(result.pages, 1)

            info = subprocess.run(
                ["pdfinfo", str(output)],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            self.assertRegex(info, r"(?m)^Pages:\s+1$")
            self.assertRegex(info, r"(?m)^Page size:\s+612 x 792 pts \(letter\)$")

            images = subprocess.run(
                ["pdfimages", "-list", str(output)],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            self.assertEqual(len(re.findall(r"(?m)^\s*1\s+\d+\s+image\s+", images)), 1)
            self.assertRegex(images, r"(?m)^\s*1\s+\d+\s+image\s+612\s+792\s+rgb")
            self.assertRegex(images, r"\s+72\s+72\s+")

            extracted_prefix = root / "extracted"
            subprocess.run(
                ["pdfimages", "-png", str(output), str(extracted_prefix)],
                check=True,
                capture_output=True,
            )
            extracted_path = root / "extracted-000.png"
            with result.manifest_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            row = rows[0]
            box = tuple(
                int(row[field])
                for field in (
                    "paste_left_px",
                    "paste_top_px",
                    "paste_right_px",
                    "paste_bottom_px",
                )
            )
            with Image.open(source_path) as decoded, Image.open(extracted_path) as page:
                self.assertEqual(page.size, (612, 792))
                self.assertIsNone(ImageChops.difference(decoded, page.crop(box)).getbbox())
                self.assertEqual(page.getpixel((0, 0)), (255, 255, 255))
                top_region = page.crop((0, 0, page.width, page.height // 4))
                self.assertIsNotNone(
                    ImageChops.difference(
                        top_region, Image.new("RGB", top_region.size, "white")
                    ).getbbox()
                )


if __name__ == "__main__":
    unittest.main()
