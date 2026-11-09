import numpy as np
import pytest

from methods.metrics import iou, mse, psnr, ssim


def test_iou_identical_boxes():
    """Однакові прямокутники мають IoU = 1."""
    assert iou((0, 0, 4, 4), (0, 0, 4, 4)) == pytest.approx(1.0)


def test_iou_partial_overlap():
    """Квадрати 3x3 (0..2) і (1..3): перетин 2x2 = 4, об'єднання 9 + 9 - 4 = 14."""
    assert iou((0, 0, 2, 2), (1, 1, 3, 3)) == pytest.approx(4 / 14)


def test_iou_no_overlap_and_none():
    """Прямокутники без перетину і відсутній результат дають 0."""
    assert iou((0, 0, 1, 1), (3, 3, 4, 4)) == 0.0
    assert iou(None, (0, 0, 1, 1)) == 0.0


def test_mse_3x3():
    """Різниця 3 в одному пікселі з 9: MSE = 9 / 9 = 1."""
    reference = np.zeros((3, 3))
    test = np.zeros((3, 3))
    test[1, 1] = 3
    assert mse(reference, test) == pytest.approx(1.0)


def test_psnr_3x3():
    """MSE = 1 -> PSNR = 10 * log10(255^2) = 48.13 дБ; однакові -> inf."""
    reference = np.zeros((3, 3))
    test = np.zeros((3, 3))
    test[1, 1] = 3
    assert psnr(reference, test) == pytest.approx(48.1308, abs=1e-3)
    assert psnr(reference, reference) == float("inf")


def test_ssim_identical_is_one():
    """SSIM однакових зображень дорівнює 1."""
    image = np.arange(25, dtype=float).reshape(5, 5) * 10
    assert ssim(image, image, window=3) == pytest.approx(1.0)


def test_ssim_constant_images_5x5():
    """Сталі зображення 100 і 110 (дисперсії 0): SSIM = (2*100*110 + C1) /
    (100^2 + 110^2 + C1), C1 = 6.5025."""
    expected = (2 * 100 * 110 + 6.5025) / (100 ** 2 + 110 ** 2 + 6.5025)
    result = ssim(np.full((5, 5), 100.0), np.full((5, 5), 110.0), window=3)
    assert result == pytest.approx(expected)


def test_ssim_noise_lowers_value():
    """Зашумлене зображення має SSIM < 1."""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (20, 20)).astype(float)
    noisy = np.clip(image + rng.normal(0, 30, image.shape), 0, 255)
    assert ssim(image, noisy) < 0.95


def test_different_shapes_raise():
    """Зображення різних розмірів викликають ValueError."""
    with pytest.raises(ValueError):
        mse(np.zeros((3, 3)), np.zeros((3, 4)))
