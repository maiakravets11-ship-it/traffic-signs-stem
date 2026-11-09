import numpy as np
 
 
def circle_offsets(radius):
    """Повертає цілочисельні зсуви (dy, dx) точок цифрового кола."""
    def inside(d_y, d_x):
        return d_y ** 2 + d_x ** 2 <= radius ** 2
 
    d_y, d_x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    on_edge = inside(d_y, d_x) & ~(inside(d_y - 1, d_x) & inside(d_y + 1, d_x)
                                   & inside(d_y, d_x - 1) & inside(d_y, d_x + 1))
    return np.stack([d_y[on_edge], d_x[on_edge]], axis=1)
 
 
def hough_circles(edges, radii):
    """Будує акумулятор Хафа для заданих радіусів."""
    edges = np.asarray(edges, dtype=bool)
    if edges.ndim != 2:
        raise ValueError(f"Контур має бути двовимірним, отримано {edges.shape}")
    if min(radii) < 1:
        raise ValueError("Радіуси мають бути не меншими за 1")
 
    height, width = edges.shape
    edge_y, edge_x = np.nonzero(edges)
    accumulator = np.zeros((len(radii), height, width))
 
    for k, radius in enumerate(radii):
        offsets = circle_offsets(radius)
        # Кандидати в центри: кожна точка контуру мінус кожен зсув
        center_y = (edge_y[:, None] - offsets[None, :, 0]).ravel()
        center_x = (edge_x[:, None] - offsets[None, :, 1]).ravel()
        inside = ((center_y >= 0) & (center_y < height)
                  & (center_x >= 0) & (center_x < width))
        flat = center_y[inside] * width + center_x[inside]
        votes = np.bincount(flat, minlength=height * width)
        accumulator[k] = votes.reshape(height, width) / len(offsets)
 
    return accumulator
 
 
def find_circles(accumulator, radii, threshold=0.35, max_circles=1,
                 min_distance=None):
    """Знаходить найсильніші кола в акумуляторі."""
    acc = accumulator.copy()
    circles = []
    while len(circles) < max_circles and acc.size > 0:
        k, center_y, center_x = map(int, np.unravel_index(np.argmax(acc),
                                                         acc.shape))
        score = float(acc[k, center_y, center_x])
        if score < threshold:
            break
        radius = int(radii[k])
        circles.append((center_y, center_x, radius, score))
        dist = radius if min_distance is None else min_distance
        acc[:, max(center_y - dist, 0):center_y + dist + 1,
            max(center_x - dist, 0):center_x + dist + 1] = 0.0
    return circles
 
 
def detect_circles(edges, r_min, r_max, threshold=0.35, max_circles=1):
    """Повний цикл: акумулятор для радіусів r_min..r_max і пошук кіл."""
    radii = list(range(max(1, int(r_min)), int(r_max) + 1))
    if not radii or not np.any(edges):
        return []
    accumulator = hough_circles(edges, radii)
    return find_circles(accumulator, radii, threshold, max_circles)
 
 
def circle_to_box(center_y, center_x, radius, shape):
    """Перетворює коло на обмежувальний прямокутник (x1, y1, x2, y2).
    Координати обрізаються межами зображення форми shape = (H, W). """
    height, width = shape[:2]
    x_1 = max(center_x - radius, 0)
    y_1 = max(center_y - radius, 0)
    x_2 = min(center_x + radius, width - 1)
    y_2 = min(center_y + radius, height - 1)
    return x_1, y_1, x_2, y_2
