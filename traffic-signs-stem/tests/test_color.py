import numpy as np
import pytest

from methods.color import normalize_brightness, red_mask, rgb_to_hsv, to_rgb

# Зображення 3x3, кожен піксель - колір із відомим HSV
IMAGE_3X3 = np.array([
    [[255, 0, 0], [0, 255, 0], [0, 0, 255]],        # червоний, зелений, синій
    [[255, 255, 0], [255, 0, 255], [0, 255, 255]],  # жовтий, пурпуровий, блакитний
    [[0, 0, 0], [255, 255, 255], [128, 0, 0]],      # чорний, білий, темно-червоний
], dtype=np.uint8)

# Очікуваний результат, пораховано вручну за формулами
EXPECTED_HSV = np.array([
    [[0, 1, 1], [120, 1, 1], [240, 1, 1]],
    [[60, 1, 1], [300, 1, 1], [180, 1, 1]],
    [[0, 0, 0], [0, 0, 1], [0, 1, 128 / 255]],
])


def test_rgb_to_hsv_basic_colors():
    """Основні кольори перетворюються у правильні H, S, V."""
    result = rgb_to_hsv(IMAGE_3X3)
    np.testing.assert_allclose(result, EXPECTED_HSV, atol=1e-6)


def test_rgb_to_hsv_hue_near_360():
    """Червоний з домішкою синього має відтінок трохи менший за 360."""
    # R=255, G=0, B=51: H = 60 * ((0 - 0.2) / 1 mod 6) = 348
    pixel = np.array([[[255, 0, 51]]], dtype=np.uint8)
    np.testing.assert_allclose(rgb_to_hsv(pixel)[0, 0, 0], 348.0, atol=1e-6)


def test_rgb_to_hsv_grayscale_input():
    """Сіре зображення (H, W) обробляється: S = 0, V = яскравість."""
    gray = np.array([[0, 128, 255]], dtype=np.uint8)
    result = rgb_to_hsv(gray)
    np.testing.assert_allclose(result[:, :, 1], 0.0)
    np.testing.assert_allclose(result[:, :, 2], [[0, 128 / 255, 1]])


def test_to_rgb_rejects_wrong_shape():
    """Масив непідтримуваної форми викликає ValueError."""
    with pytest.raises(ValueError):
        to_rgb(np.zeros((3, 3, 2)))


def test_red_mask_3x3():
    """Маска позначає лише насичені, достатньо яскраві червоні пікселі."""
    image = np.array([
        [[255, 0, 0], [200, 20, 30], [255, 0, 51]],      # червоні
        [[0, 255, 0], [128, 128, 128], [255, 200, 200]],  # зелений, сірий, блідо-рожевий
        [[30, 0, 0], [0, 0, 255], [255, 255, 0]],         # дуже темний червоний, синій, жовтий
    ], dtype=np.uint8)
    expected = np.array([
        [True, True, True],
        [False, False, False],
        [False, False, False],
    ])
    np.testing.assert_array_equal(red_mask(rgb_to_hsv(image)), expected)


def test_normalize_brightness_3x3():
    """Найяскравіший піксель (V = 0.5) стає 1.0, решта масштабується в 2 рази."""
    hsv = np.zeros((3, 3, 3))
    hsv[:, :, 0] = 10.0
    hsv[:, :, 1] = 0.6
    hsv[:, :, 2] = [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.5, 0.25, 0.0]]
    result = normalize_brightness(hsv, percentile=100)
    expected_v = [[0.2, 0.4, 0.6], [0.2, 0.4, 0.6], [1.0, 0.5, 0.0]]
    np.testing.assert_allclose(result[:, :, 2], expected_v)
    np.testing.assert_allclose(result[:, :, :2], hsv[:, :, :2])


def test_normalize_brightness_black_image():
    """Повністю чорне зображення не змінюється (немає ділення на нуль)."""
    hsv = np.zeros((3, 3, 3))
    np.testing.assert_array_equal(normalize_brightness(hsv), hsv)