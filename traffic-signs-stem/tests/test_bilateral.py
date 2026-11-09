import numpy as np
import pytest

from methods.bilateral import bilateral_filter


def test_constant_image_unchanged():
    """Однотонне зображення після фільтра не змінюється."""
    image = np.full((5, 5, 3), 90, dtype=np.uint8)
    np.testing.assert_array_equal(bilateral_filter(image), image)


def test_single_pixel_3x3_large_sigma_r():
    """Центр 30 серед нулів, вікно 3x3, sigma_s = 2, sigma_r дуже велика."""
    image = np.zeros((3, 3), dtype=np.uint8)
    image[1, 1] = 30
    result = bilateral_filter(image, size=3, sigma_s=2.0, sigma_r=1e6)
    assert result[1, 1] == 4


def test_edge_is_preserved_5x5():
    """Різкий край 0 | 200 з малою sigma_r лишається різким."""
    image = np.zeros((5, 5), dtype=np.uint8)
    image[:, 3:] = 200
    result = bilateral_filter(image, size=5, sigma_s=2.0, sigma_r=10.0)
    np.testing.assert_array_equal(result, image)


def test_noise_is_reduced():
    """Шум зменшується: середньоквадратична похибка після фільтра менша."""
    rng = np.random.default_rng(1)
    clean = np.full((20, 20, 3), 120.0)
    noisy = np.clip(clean + rng.normal(0, 15, clean.shape), 0, 255)
    filtered = bilateral_filter(noisy.astype(np.uint8))
    assert np.mean((filtered - clean) ** 2) < np.mean((noisy - clean) ** 2)


def test_wrong_parameters():
    """Парне вікно або нульова сигма викликають ValueError."""
    image = np.zeros((5, 5), dtype=np.uint8)
    with pytest.raises(ValueError):
        bilateral_filter(image, size=4)
    with pytest.raises(ValueError):
        bilateral_filter(image, sigma_r=0)
