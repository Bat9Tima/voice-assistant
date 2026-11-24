"""Сравнение кадров без модели: изменился ли экран.

Отпечаток dHash ловит смену структуры, средняя разница яркости — заливку
и плавные изменения. Пороги вынесены в параметры: их нужно калибровать
по реальным сценариям, а не задавать на глаз.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageChops, ImageStat

DEFAULT_SAMPLE_SIZE = (64, 36)
DEFAULT_HASH_SIZE = 8
DEFAULT_PIXEL_THRESHOLD = 0.02
DEFAULT_HASH_THRESHOLD = 0.15


@dataclass(frozen=True)
class ChangeResult:
    """Насколько следующий кадр отличается от предыдущего."""

    changed: bool
    score: float
    pixel_diff: float
    hash_distance: int
    hash_ratio: float
    reason: str


def dhash(image: Image.Image, hash_size: int = DEFAULT_HASH_SIZE) -> int:
    """Разностный хеш кадра: 1, если левый пиксель ярче правого."""
    if hash_size < 1:
        raise ValueError("hash_size должен быть положительным")
    gray = image.convert("L").resize(
        (hash_size + 1, hash_size),
        Image.Resampling.BILINEAR,
    )
    data = gray.tobytes()
    bits = 0
    stride = hash_size + 1
    for row in range(hash_size):
        offset = row * stride
        for col in range(hash_size):
            bits = (bits << 1) | (1 if data[offset + col] > data[offset + col + 1] else 0)
    return bits


def pixel_difference(
    previous: Image.Image,
    current: Image.Image,
    *,
    sample_size: tuple[int, int] = DEFAULT_SAMPLE_SIZE,
) -> float:
    """Средняя разница яркости в долях от 255: от 0.0 до 1.0."""
    left = previous.convert("L").resize(sample_size, Image.Resampling.BILINEAR)
    right = current.convert("L").resize(sample_size, Image.Resampling.BILINEAR)
    diff = ImageChops.difference(left, right)
    return ImageStat.Stat(diff).mean[0] / 255.0


class FrameProcessor:
    """Решает, считается ли новый кадр изменившимся."""

    def __init__(
        self,
        *,
        sample_size: tuple[int, int] = DEFAULT_SAMPLE_SIZE,
        hash_size: int = DEFAULT_HASH_SIZE,
        pixel_threshold: float = DEFAULT_PIXEL_THRESHOLD,
        hash_threshold: float = DEFAULT_HASH_THRESHOLD,
    ) -> None:
        if hash_size < 1:
            raise ValueError("hash_size должен быть положительным")
        if sample_size[0] < 1 or sample_size[1] < 1:
            raise ValueError("sample_size должен быть положительным")
        pixel_threshold = float(pixel_threshold)
        hash_threshold = float(hash_threshold)
        if not 0 <= pixel_threshold <= 1:
            raise ValueError("pixel_threshold должен быть от 0 до 1")
        if not 0 <= hash_threshold <= 1:
            raise ValueError("hash_threshold должен быть от 0 до 1")
        self.sample_size = sample_size
        self.hash_size = hash_size
        self.hash_bits = hash_size * hash_size
        self.pixel_threshold = float(pixel_threshold)
        self.hash_threshold = float(hash_threshold)

    def compare(self, previous: Image.Image, current: Image.Image) -> ChangeResult:
        """Сравнивает два кадра и объясняет решение."""
        if previous.size != current.size:
            return ChangeResult(
                changed=True,
                score=1.0,
                pixel_diff=1.0,
                hash_distance=self.hash_bits,
                hash_ratio=1.0,
                reason="size",
            )
        pixel_diff = pixel_difference(previous, current, sample_size=self.sample_size)
        distance = (dhash(previous, self.hash_size) ^ dhash(current, self.hash_size)).bit_count()
        ratio = distance / self.hash_bits
        content = pixel_diff >= self.pixel_threshold
        structure = ratio >= self.hash_threshold
        if content and structure:
            reason = "content+structure"
        elif content:
            reason = "content"
        elif structure:
            reason = "structure"
        else:
            reason = "same"
        return ChangeResult(
            changed=content or structure,
            score=max(pixel_diff, ratio),
            pixel_diff=pixel_diff,
            hash_distance=distance,
            hash_ratio=ratio,
            reason=reason,
        )


__all__ = [
    "DEFAULT_HASH_SIZE",
    "DEFAULT_HASH_THRESHOLD",
    "DEFAULT_PIXEL_THRESHOLD",
    "DEFAULT_SAMPLE_SIZE",
    "ChangeResult",
    "FrameProcessor",
    "dhash",
    "pixel_difference",
]
