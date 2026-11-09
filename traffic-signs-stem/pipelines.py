import numpy as np

from methods.bilateral import bilateral_filter
from methods.color import normalize_brightness, red_mask, rgb_to_hsv, to_rgb
from methods.geometry import normalize_sign
from methods.histogram import equalize_brightness
from methods.hough import circle_offsets, circle_to_box, detect_circles
from methods.morphology import (boundary, closing, disk_kernel, fill_holes,
                                opening)
from methods.region_growing import mask_to_box, segment_sign


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


def pipeline2(image, params):
    """Пайплайн 2: виділення знака як області та нормалізація розміру.
 
    Стадії:
        1. гістограмне вирівнювання яскравості (гістограмна корекція);
        2. білатеральний фільтр (фільтрація);
        3. регіональне зростання фону від країв, знак - область
           навколо центру, до якої фон не дійшов (сегментація);
        4. афінне нормування знайденого знака до квадрата out_size
           (геометричні перетворення)"""
    image = to_rgb(image)
    stages = {}
 
    # Стадія 1: гістограмна корекція
    image = equalize_brightness(image)
    stages["1_equalized"] = image
 
    # Стадія 2: фільтрація
    image = bilateral_filter(image, **params["bilateral"])
    stages["2_bilateral"] = image
 
    # Стадія 3: сегментація
    region = params["region"]
    thresholds = range(region["threshold_min"], region["threshold_max"] + 1,
                       region["threshold_step"])
    mask, _ = segment_sign(image, thresholds, region["min_area"])
    stages["3_region"] = mask_to_image(mask)
    box = mask_to_box(mask)
    if box is None:
        return stages, None, 0.0
 
    # Стадія 4: геометричне нормування
    stages["4_normalized"] = normalize_sign(image, box, params["out_size"])
    return stages, box, float(mask.mean())
 
 
PIPELINES = {"1": pipeline1, "2": pipeline2}