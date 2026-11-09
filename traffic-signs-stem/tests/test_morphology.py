import numpy as np
import pytest

from methods.morphology import (boundary, closing, dilate, disk_kernel,
                                erode, gradient, opening)

CROSS = np.array([[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]], dtype=bool)
SQUARE = np.ones((3, 3), dtype=bool)

# Квадрат 3x3 у центрі маски 5x5
BLOCK_5X5 = np.array([[0, 0, 0, 0, 0],
                      [0, 1, 1, 1, 0],
                      [0, 1, 1, 1, 0],
                      [0, 1, 1, 1, 0],
                      [0, 0, 0, 0, 0]], dtype=bool)


def test_disk_kernel_radius_1_is_cross():
    """Диск радіуса 1 - це хрест 3x3."""
    np.testing.assert_array_equal(disk_kernel(1), CROSS)


def test_disk_kernel_negative_radius():
    """Від'ємний радіус викликає ValueError."""
    with pytest.raises(ValueError):
        disk_kernel(-1)


def test_erode_block_leaves_center():
    """Ерозія квадрата 3x3 хрестом залишає лише центральний піксель."""
    expected = np.zeros((5, 5), dtype=bool)
    expected[2, 2] = True
    np.testing.assert_array_equal(erode(BLOCK_5X5, CROSS), expected)


def test_dilate_point_gives_cross():
    """Дилатація одного пікселя хрестом дає хрест."""
    point = np.zeros((5, 5), dtype=bool)
    point[2, 2] = True
    expected = np.zeros((5, 5), dtype=bool)
    expected[1:4, 1:4] = CROSS
    np.testing.assert_array_equal(dilate(point, CROSS), expected)


def test_opening_removes_isolated_pixel():
    """Відкриття квадратом прибирає одиночний піксель і зберігає блок 3x3."""
    mask = BLOCK_5X5.copy()
    mask[0, 4] = True  # одиночна «пляма»
    np.testing.assert_array_equal(opening(mask, SQUARE), BLOCK_5X5)


def test_closing_fills_hole():
    """Закриття квадратом заповнює дірку в центрі блоку 3x3 (маска 7x7)."""
    mask = np.zeros((7, 7), dtype=bool)
    mask[2:5, 2:5] = True
    mask[3, 3] = False  # дірка
    expected = np.zeros((7, 7), dtype=bool)
    expected[2:5, 2:5] = True
    np.testing.assert_array_equal(closing(mask, SQUARE), expected)


def test_boundary_of_block_is_ring():
    """Внутрішня межа блоку 3x3 - кільце з 8 пікселів без центру."""
    expected = BLOCK_5X5.copy()
    expected[2, 2] = False
    np.testing.assert_array_equal(boundary(BLOCK_5X5), expected)


def test_gradient_of_point():
    """Градієнт одного пікселя: дилатація (хрест) мінус порожня ерозія."""
    point = np.zeros((3, 3), dtype=bool)
    point[1, 1] = True
    np.testing.assert_array_equal(gradient(point, CROSS), CROSS)


def test_empty_mask_stays_empty():
    """Порожня маска лишається порожньою після всіх операцій."""
    empty = np.zeros((5, 5), dtype=bool)
    for operation in (erode, dilate, opening, closing, gradient):
        assert not operation(empty, CROSS).any()


def test_wrong_input_shapes():
    """3D-маска або парний структурний елемент викликають ValueError."""
    with pytest.raises(ValueError):
        erode(np.zeros((3, 3, 3)), CROSS)
    with pytest.raises(ValueError):
        dilate(BLOCK_5X5, np.ones((2, 2)))
