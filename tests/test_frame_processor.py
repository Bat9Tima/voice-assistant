from __future__ import annotations

import unittest

from PIL import Image, ImageDraw

from app.frame_processor import FrameProcessor, dhash, pixel_difference


def bars(offset: int = 0, size: tuple[int, int] = (320, 180)) -> Image.Image:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    for x in range(-40 + offset, size[0], 40):
        draw.rectangle([x, 0, x + 20, size[1]], fill="black")
    return image


class FrameProcessorTest(unittest.TestCase):
    def test_identical_frames_are_unchanged(self) -> None:
        result = FrameProcessor().compare(
            Image.new("RGB", (320, 180), "white"),
            Image.new("RGB", (320, 180), "white"),
        )

        self.assertFalse(result.changed)
        self.assertEqual(result.reason, "same")
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.hash_distance, 0)

    def test_solid_color_change_is_content(self) -> None:
        result = FrameProcessor().compare(
            Image.new("RGB", (320, 180), "black"),
            Image.new("RGB", (320, 180), "white"),
        )

        self.assertTrue(result.changed)
        self.assertEqual(result.reason, "content")
        self.assertEqual(result.pixel_diff, 1.0)
        self.assertEqual(result.hash_ratio, 0.0)

    def test_shifted_pattern_is_changed(self) -> None:
        result = FrameProcessor().compare(bars(0), bars(6))

        self.assertTrue(result.changed)
        self.assertGreater(result.score, 0.0)

    def test_tiny_dot_is_ignored(self) -> None:
        clean = Image.new("RGB", (640, 360), "white")
        dotted = clean.copy()
        ImageDraw.Draw(dotted).point((10, 10), fill="black")

        result = FrameProcessor().compare(clean, dotted)

        self.assertFalse(result.changed)
        self.assertEqual(result.reason, "same")

    def test_size_change_is_changed(self) -> None:
        result = FrameProcessor().compare(
            Image.new("RGB", (320, 180), "white"),
            Image.new("RGB", (320, 181), "white"),
        )

        self.assertTrue(result.changed)
        self.assertEqual(result.reason, "size")
        self.assertEqual(result.score, 1.0)

    def test_thresholds_can_ignore_small_difference(self) -> None:
        white = Image.new("RGB", (320, 180), "white")
        gray = Image.new("RGB", (320, 180), (230, 230, 230))

        self.assertTrue(FrameProcessor().compare(white, gray).changed)
        strict = FrameProcessor(pixel_threshold=0.2, hash_threshold=1.0)
        self.assertFalse(strict.compare(white, gray).changed)

    def test_dhash_is_stable(self) -> None:
        first = bars(0)
        second = bars(0)
        gradient = Image.new("L", (256, 1))
        gradient.putdata(list(range(255, -1, -1)))
        gradient = gradient.resize((320, 180)).convert("RGB")

        self.assertEqual(dhash(first), dhash(second))
        self.assertNotEqual(dhash(gradient), 0)

    def test_pixel_difference_bounds(self) -> None:
        self.assertEqual(
            pixel_difference(
                Image.new("RGB", (10, 10), "black"),
                Image.new("RGB", (10, 10), "white"),
            ),
            1.0,
        )
        self.assertEqual(
            pixel_difference(
                Image.new("RGB", (10, 10), "white"),
                Image.new("RGB", (10, 10), "white"),
            ),
            0.0,
        )

    def test_invalid_hash_size(self) -> None:
        with self.assertRaises(ValueError):
            FrameProcessor(hash_size=0)

    def test_invalid_sample_size(self) -> None:
        with self.assertRaises(ValueError):
            FrameProcessor(sample_size=(0, 36))

    def test_invalid_thresholds(self) -> None:
        for kwargs in (
            {"pixel_threshold": -0.1},
            {"pixel_threshold": 1.1},
            {"hash_threshold": -0.1},
            {"hash_threshold": 1.1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                FrameProcessor(**kwargs)


if __name__ == "__main__":
    unittest.main()
