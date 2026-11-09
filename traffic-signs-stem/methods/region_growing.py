from collections import deque

import numpy as np

# Зсуви до 4 сусідів: вгору, вниз, ліворуч, праворуч
NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def region_grow(image, seeds, threshold=20.0):
    """Вирощує область від початкових пікселів (seeds). """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim == 2:
        image = image[:, :, None]
    height, width = image.shape[:2]
    region = np.zeros((height, width), dtype=bool)
    queue = deque()
    for y_pos, x_pos in seeds:
        if 0 <= y_pos < height and 0 <= x_pos < width and not region[y_pos, x_pos]:
            region[y_pos, x_pos] = True
            queue.append((y_pos, x_pos))

    limit = threshold ** 2
    while queue:
        y_pos, x_pos = queue.popleft()
        color = image[y_pos, x_pos]
        for d_y, d_x in NEIGHBOURS:
            n_y, n_x = y_pos + d_y, x_pos + d_x
            if (0 <= n_y < height and 0 <= n_x < width
                    and not region[n_y, n_x]
                    and np.sum((image[n_y, n_x] - color) ** 2) < limit):
                region[n_y, n_x] = True
                queue.append((n_y, n_x))
    return region


def border_seeds(shape):
    """Повертає координати всіх пікселів на краях зображення."""
    height, width = shape[:2]
    seeds = [(0, x) for x in range(width)] + [(height - 1, x) for x in range(width)]
    seeds += [(y, 0) for y in range(height)] + [(y, width - 1) for y in range(height)]
    return seeds


def mask_to_box(mask):
    """Повертає обмежувальний прямокутник (x1, y1, x2, y2) маски або None."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def connected_component(mask, seed):
    """Повертає зв'язну (за 4 сусідами) частину маски, що містить точку seed. """
    mask = np.asarray(mask, dtype=bool)
    if not mask[seed]:
        return np.zeros_like(mask)
    return region_grow(mask.astype(np.float64), [seed], threshold=0.5)


def segment_sign(image, thresholds, min_area=0.35):
    """Виділяє знак: фон росте від країв, знак - те, куди фон не дійшов. """
    image = np.asarray(image)
    center = (image.shape[0] // 2, image.shape[1] // 2)
    seeds = border_seeds(image.shape)
    chosen = None
    for threshold in thresholds:
        background = region_grow(image, seeds, threshold)
        sign = connected_component(~background, center)
        if chosen is None or sign.mean() >= min_area:
            chosen = (sign, threshold)
        else:
            break
    return chosen
