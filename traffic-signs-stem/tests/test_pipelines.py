import json
from pathlib import Path

import numpy as np

from methods.metrics import iou
from pipelines import pipeline1

CONFIG = json.loads((Path(__file__).parent.parent / "config.json")
                    .read_text(encoding="utf-8"))


def synthetic_sign(size=60, center=(30, 31), radius=22, noise=0.0):
    """Круглий знак: червоний обідок 5 px, біла середина, сіро-зелений фон."""
    y_coords, x_coords = np.mgrid[:size, :size]
    dist = np.hypot(y_coords - center[0], x_coords - center[1])
    image = np.full((size, size, 3), [90, 110, 90], dtype=np.float64)
    image[dist <= radius] = [190, 25, 35]
    image[dist <= radius - 5] = [235, 235, 235]
    image += np.random.default_rng(0).normal(0, noise, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def test_pipeline1_finds_sign():
    """Рамка знайденого знака збігається з еталонною (IoU >= 0.85)."""
    stages, box, score = pipeline1(synthetic_sign(), CONFIG["pipeline1"])
    truth = (31 - 22, 30 - 22, 31 + 22, 30 + 22)
    assert iou(box, truth) >= 0.85
    assert score > 0.35
    assert set(stages) == {"1_red_mask", "2_morphology", "2_edges", "3_hough"}


def test_pipeline1_noisy_sign():
    """Шум sigma = 25 не заважає знайти знак (IoU >= 0.7)."""
    _, box, _ = pipeline1(synthetic_sign(noise=25), CONFIG["pipeline1"])
    assert iou(box, (9, 8, 53, 52)) >= 0.7


def test_pipeline1_no_red():
    """Зображення без червоного: знак не знайдено, без помилки."""
    image = np.full((40, 40, 3), 120, dtype=np.uint8)
    _, box, score = pipeline1(image, CONFIG["pipeline1"])
    assert box is None and score == 0.0
