import numpy as np


def iou(box_a, box_b):
    """Обчислює IoU (Intersection over Union) двох прямокутників.

    Прямокутник задається як (x1, y1, x2, y2), координати включні,
    тобто прямокутник (0, 0, 1, 1) займає 2x2 пікселі."""
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = min(ax2, bx2) - max(ax1, bx1) + 1
    inter_h = min(ay2, by2) - max(ay1, by1) + 1
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    intersection = inter_w * inter_h
    area_a = (ax2 - ax1 + 1) * (ay2 - ay1 + 1)
    area_b = (bx2 - bx1 + 1) * (by2 - by1 + 1)
    return intersection / (area_a + area_b - intersection)


def mse(reference, test):
    """Середньоквадратична похибка: середнє (reference - test)^2."""
    reference = np.asarray(reference, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    if reference.shape != test.shape:
        raise ValueError(f"Різні розміри: {reference.shape} і {test.shape}")
    return float(np.mean((reference - test) ** 2))


def psnr(reference, test, max_value=255.0):
    """Пікове відношення сигнал/шум у децибелах."""
    error = mse(reference, test)
    if error == 0:
        return float("inf")
    return float(10.0 * np.log10(max_value ** 2 / error))


def _local_mean(image, size):
    """Середнє у вікні size x size навколо кожного пікселя (через інтегральне зображення). """
    half = size // 2
    height, width = image.shape
    integral = np.zeros((height + 1, width + 1))
    integral[1:, 1:] = image.cumsum(axis=0).cumsum(axis=1)

    rows = np.arange(height)
    cols = np.arange(width)
    top = np.clip(rows - half, 0, height)[:, None]
    bottom = np.clip(rows + half + 1, 0, height)[:, None]
    left = np.clip(cols - half, 0, width)[None, :]
    right = np.clip(cols + half + 1, 0, width)[None, :]

    window_sum = (integral[bottom, right] - integral[top, right]
                  - integral[bottom, left] + integral[top, left])
    return window_sum / ((bottom - top) * (right - left))


def ssim(reference, test, max_value=255.0, window=7):
    """Індекс структурної схожості (SSIM), середній по зображенню."""
    reference = np.asarray(reference, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    if reference.shape != test.shape:
        raise ValueError(f"Різні розміри: {reference.shape} і {test.shape}")
    if reference.ndim == 3:
        return float(np.mean([ssim(reference[:, :, c], test[:, :, c],
                                   max_value, window)
                              for c in range(reference.shape[2])]))

    c_1 = (0.01 * max_value) ** 2
    c_2 = (0.03 * max_value) ** 2
    mu_x = _local_mean(reference, window)
    mu_y = _local_mean(test, window)
    var_x = _local_mean(reference ** 2, window) - mu_x ** 2
    var_y = _local_mean(test ** 2, window) - mu_y ** 2
    cov_xy = _local_mean(reference * test, window) - mu_x * mu_y

    numerator = (2 * mu_x * mu_y + c_1) * (2 * cov_xy + c_2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c_1) * (var_x + var_y + c_2)
    return float(np.mean(numerator / denominator))
