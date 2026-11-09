import numpy as np

from methods.color import normalize_brightness, red_mask, rgb_to_hsv, to_rgb
from methods.hough import circle_offsets, circle_to_box, detect_circles
from methods.morphology import (boundary, closing, disk_kernel, fill_holes,
                                opening)


def mask_to_image(mask):
    """Перетворює бінарну маску на чорно-біле зображення uint8 (0 або 255)."""
    return np.asarray(mask, dtype=np.uint8) * 255


def draw_circle(image, center_y, center_x, radius, color=(0, 255, 0)):
    """Малює контур кола поверх копії RGB-зображення (для візуалізації)."""
    result = to_rgb(image).copy()
    height, width = result.shape[:2]
    for d_y, d_x in circle_offsets(radius):
        y_pos, x_pos = center_y + d_y, center_x + d_x
        if 0 <= y_pos < height and 0 <= x_pos < width:
            result[y_pos, x_pos] = color
    return result


def pipeline1(image, params):
    """Пайплайн 1: виявлення знака за кольором і формою.

    Стадії:
        1. RGB -> HSV, нормування яскравості й червона маска
           (колірна корекція);
        2. закриття, відкриття, заповнення дірок, внутрішня межа
           (морфологічна обробка);
        3. перетворення Хафа для кіл (перетворення Хафа). """
    image = to_rgb(image)
    stages = {}

    # Стадія 1: колірна корекція
    hsv = normalize_brightness(rgb_to_hsv(image),
                               params["brightness_percentile"])
    mask = red_mask(hsv, **params["red"])
    stages["1_red_mask"] = mask_to_image(mask)

    # Стадія 2: морфологія
    mask = closing(mask, disk_kernel(params["closing_radius"]))
    # opening_radius = 0 вимикає відкриття (диск 1x1 нічого не змінює)
    mask = opening(mask, disk_kernel(params["opening_radius"]))
    mask = fill_holes(mask)
    stages["2_morphology"] = mask_to_image(mask)
    edges = boundary(mask)
    stages["2_edges"] = mask_to_image(edges)

    # Стадія 3: перетворення Хафа
    side = min(image.shape[:2])
    circles = detect_circles(edges,
                             r_min=params["r_min_frac"] * side,
                             r_max=params["r_max_frac"] * side,
                             threshold=params["hough_threshold"])
    if not circles:
        stages["3_hough"] = image.copy()
        return stages, None, 0.0

    center_y, center_x, radius, score = circles[0]
    stages["3_hough"] = draw_circle(image, center_y, center_x, radius)
    box = circle_to_box(center_y, center_x, radius, image.shape)
    return stages, box, score


PIPELINES = {"1": pipeline1}