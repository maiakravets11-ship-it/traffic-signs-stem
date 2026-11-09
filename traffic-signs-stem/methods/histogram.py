import numpy as np


def histogram(channel):
    """Рахує, скільки пікселів має кожну яскравість 0..255."""
    return np.bincount(np.asarray(channel, dtype=np.uint8).ravel(),
                       minlength=256)


def equalize(channel):
    """Гістограмне вирівнювання одного каналу (значення 0..255)."""
    channel = np.asarray(channel, dtype=np.uint8)
    cdf = np.cumsum(histogram(channel))
    total = cdf[-1]
    cdf_min = cdf[cdf > 0][0]
    if total == cdf_min:
        return channel.copy()
    lookup = np.round((cdf - cdf_min) / (total - cdf_min) * 255.0)
    lookup = np.clip(lookup, 0, 255).astype(np.uint8)
    return lookup[channel]


def equalize_brightness(image):
    """Вирівнює яскравість кольорового зображення, не змінюючи кольорів."""
    image = np.asarray(image, dtype=np.uint8)
    if image.ndim == 2:
        return equalize(image)
    value = image.max(axis=2)
    new_value = equalize(value).astype(np.float64)
    ratio = new_value / np.maximum(value, 1)
    result = image.astype(np.float64) * ratio[:, :, None]
    return np.clip(np.round(result), 0, 255).astype(np.uint8)
