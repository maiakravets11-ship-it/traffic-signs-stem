import numpy as np


def bilateral_filter(image, size=5, sigma_s=2.0, sigma_r=25.0):
    """Білатеральна фільтрація зображення"""
    if size < 1 or size % 2 == 0:
        raise ValueError("Розмір вікна має бути непарним додатним числом")
    if sigma_s <= 0 or sigma_r <= 0:
        raise ValueError("Сигми мають бути додатними")

    image = np.asarray(image, dtype=np.float64)
    gray = image.ndim == 2
    if gray:
        image = image[:, :, None]
    result = _weighted_average(image, size, sigma_s, sigma_r)
    result = np.clip(np.round(result), 0, 255).astype(np.uint8)
    return result[:, :, 0] if gray else result


def _weighted_average(image, size, sigma_s, sigma_r):
    """Обчислює білатеральне зважене середнє для масиву форми (H, W, C)."""
    half = size // 2
    height, width = image.shape[:2]
    padded = np.pad(image, ((half, half), (half, half), (0, 0)), mode="edge")
    numerator = np.zeros_like(image)
    denominator = np.zeros((height, width, 1))

    for d_y in range(-half, half + 1):
        for d_x in range(-half, half + 1):
            neighbour = padded[half + d_y:half + d_y + height,
                               half + d_x:half + d_x + width]
            # вага за відстанню * вага за різницею кольору
            weight = (np.exp(-(d_y ** 2 + d_x ** 2) / (2.0 * sigma_s ** 2))
                      * np.exp(-np.sum((neighbour - image) ** 2, axis=2,
                                       keepdims=True) / (2.0 * sigma_r ** 2)))
            numerator += weight * neighbour
            denominator += weight
    return numerator / denominator