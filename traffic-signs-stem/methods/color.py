import numpy as np


def to_rgb(image):
    """Приводить зображення до формату RGB з трьома каналами.

    Зображення у відтінках сірого (H, W) копіюється в три канали,
    у зображення RGBA (H, W, 4) відкидається канал прозорості. """
    image = np.asarray(image)
    if image.ndim == 2:
        return np.stack([image, image, image], axis=2)
    if image.ndim == 3 and image.shape[2] in (3, 4):
        return image[:, :, :3]
    raise ValueError(f"Непідтримувана форма зображення: {image.shape}")


def rgb_to_hsv(image):
    """Перетворює зображення з RGB у HSV.

    Формули:
        V = max(R, G, B)
        S = (max - min) / max, або 0, якщо max = 0
        H = 60 * ((G - B) / delta mod 6), якщо max = R
            60 * ((B - R) / delta + 2),   якщо max = G
            60 * ((R - G) / delta + 4),   якщо max = B
            0, якщо delta = 0 (сірий колір)  """
    rgb = to_rgb(image).astype(np.float64) / 255.0
    c_max = rgb.max(axis=2)
    delta = c_max - rgb.min(axis=2)

    hue = _hue(rgb, c_max, delta)
    safe_max = np.where(c_max > 0, c_max, 1.0)
    saturation = np.where(c_max > 0, delta / safe_max, 0.0)

    return np.stack([hue, saturation, c_max], axis=2)


def _hue(rgb, c_max, delta):
    """Обчислює відтінок H у градусах для нормованого RGB-зображення. """
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    # Для сірих пікселів (delta = 0) відтінок не визначений, лишаємо 0.
    # Ділимо на safe_delta, щоб не було ділення на нуль.
    colored = delta > 0
    safe_delta = np.where(colored, delta, 1.0)

    is_red = colored & (c_max == red)
    is_green = colored & (c_max == green) & ~is_red
    is_blue = colored & ~is_red & ~is_green

    hue = np.zeros_like(c_max)
    hue[is_red] = (60.0 * (green - blue) / safe_delta)[is_red] % 360.0
    hue[is_green] = (60.0 * ((blue - red) / safe_delta + 2.0))[is_green]
    hue[is_blue] = (60.0 * ((red - green) / safe_delta + 4.0))[is_blue]
    return hue


def normalize_brightness(hsv, percentile=99.0):
    """Розтягує яскравість V зображення до повного діапазону [0; 1]."""
    result = np.array(hsv, dtype=np.float64)
    reference = np.percentile(result[:, :, 2], percentile)
    if reference > 0:
        result[:, :, 2] = np.clip(result[:, :, 2] / reference, 0.0, 1.0)
    return result


def red_mask(hsv, hue_low=15.0, hue_high=340.0, min_sat=0.4, min_val=0.2):
    """Повертає бінарну маску червоних пікселів."""
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    red_hue = (hue <= hue_low) | (hue >= hue_high)
    return red_hue & (sat >= min_sat) & (val >= min_val)