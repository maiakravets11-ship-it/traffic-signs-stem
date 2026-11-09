import numpy as np

from methods.histogram import equalize, equalize_brightness, histogram

# По три пікселі яскравостей 50, 100 і 150
CHANNEL_3X3 = np.array([[50, 50, 50],
                        [100, 100, 100],
                        [150, 150, 150]], dtype=np.uint8)


def test_histogram_3x3():
    """Гістограма рахує по 3 пікселі для значень 50, 100, 150."""
    hist = histogram(CHANNEL_3X3)
    assert hist[50] == hist[100] == hist[150] == 3
    assert hist.sum() == 9


def test_equalize_3x3():
    """cdf = 3, 6, 9; cdf_min = 3; N = 9:
    50 -> 0, 100 -> round(3 / 6 * 255) = 128, 150 -> 255."""
    expected = np.array([[0, 0, 0],
                         [128, 128, 128],
                         [255, 255, 255]], dtype=np.uint8)
    np.testing.assert_array_equal(equalize(CHANNEL_3X3), expected)


def test_s6_narrow_range_is_stretched():
    """S6: діапазон [100; 150] розтягується щонайменше до [5; 250]."""
    channel = np.random.default_rng(0).integers(100, 151, (20, 20))
    result = equalize(channel.astype(np.uint8))
    assert result.min() <= 5 and result.max() >= 250


def test_s7_uniform_image_unchanged():
    """S7: однотонне зображення не змінюється (немає ділення на нуль)."""
    channel = np.full((5, 5), 77, dtype=np.uint8)
    np.testing.assert_array_equal(equalize(channel), channel)


def test_equalize_brightness_keeps_color_ratio():
    """Вирівнювання яскравості зберігає пропорції R:G:B (тобто колір)."""
    image = np.zeros((3, 3, 3), dtype=np.uint8)
    image[0] = [40, 20, 10]
    image[1] = [80, 40, 20]
    image[2] = [120, 60, 30]
    result = equalize_brightness(image).astype(float)
    # Яскравості 40, 80, 120 стають 0, 128, 255
    np.testing.assert_array_equal(result[:, 0, 0], [0, 128, 255])
    # Співвідношення R : G : B = 4 : 2 : 1 зберігається (з точністю округлення)
    np.testing.assert_allclose(result[2, 0] / result[2, 0, 0],
                               [1, 0.5, 0.25], atol=0.01)
