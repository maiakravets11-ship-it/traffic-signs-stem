
import numpy as np
import pytest
 
from methods.hough import (circle_offsets, circle_to_box, detect_circles,
                           hough_circles)
from methods.color import red_mask, rgb_to_hsv
from methods.morphology import (boundary, closing, disk_kernel, fill_holes,
                                opening)
 
# Цифрове коло радіуса 1 навколо центру (2, 2): 4 сусіди-хрест
CIRCLE_R1_5X5 = np.array([[0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0],
                          [0, 1, 0, 1, 0],
                          [0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0]], dtype=bool)
 
 
def make_sign(size, rings, noise_sigma=0.0, seed=0):
    """Малює червоні кільця (обідок 4 px) на сірому фоні, за потреби з шумом.
 
    rings - список (center_y, center_x, radius) зовнішнього краю обідка.
    """
    y_coords, x_coords = np.mgrid[:size, :size]
    image = np.full((size, size, 3), [100, 110, 100], dtype=np.float64)
    for center_y, center_x, radius in rings:
        dist = np.hypot(y_coords - center_y, x_coords - center_x)
        image[(dist <= radius) & (dist > radius - 4)] = [200, 30, 40]
    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        image += rng.normal(0.0, noise_sigma, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)
 
 
def pipeline1_edges(image):
    """Стадії 1-2 пайплайна 1: червона маска -> закриття -> відкриття -> межа.
 
    Після закриття обідок заповнюється в суцільний диск, тому межа -
    це зовнішній край знака.
    """
    mask = red_mask(rgb_to_hsv(image))
    mask = closing(mask, disk_kernel(2))
    mask = opening(mask, disk_kernel(1))
    return boundary(fill_holes(mask))
 
 
def test_circle_offsets_radius_1():
    """Цифрове коло радіуса 1 - це 4 сусіди центру (межа диска-хреста)."""
    offsets = {tuple(o) for o in circle_offsets(1)}
    assert offsets == {(-1, 0), (1, 0), (0, -1), (0, 1)}
 
 
def test_accumulator_circle_5x5():
    """Повне коло r=1 дає в центрі (2, 2) значення 1.0, і це єдиний максимум."""
    acc = hough_circles(CIRCLE_R1_5X5, [1])
    assert acc[0, 2, 2] == pytest.approx(1.0)
    assert np.sum(acc[0] == acc[0].max()) == 1
 
 
def test_accumulator_half_circle_5x5():
    """Лише 2 точки з 4 дають у центрі значення 0.5."""
    half = CIRCLE_R1_5X5.copy()
    half[2, 1] = half[2, 3] = False
    acc = hough_circles(half, [1])
    assert acc[0, 2, 2] == pytest.approx(0.5)
 
 
def test_s1_single_ring():
    """S1: кільце з центром (32, 32), r = 20 знаходиться з похибкою <= 1 px."""
    circles = detect_circles(pipeline1_edges(make_sign(64, [(32, 32, 20)])),
                             10, 30)
    center_y, center_x, radius, _ = circles[0]
    assert abs(center_y - 32) <= 1 and abs(center_x - 32) <= 1
    assert abs(radius - 20) <= 1
 
 
def test_s2_noisy_ring():
    """S2: те саме кільце з шумом знаходиться з похибкою <= 2 px."""
    image = make_sign(64, [(32, 32, 20)], noise_sigma=15)
    edges = pipeline1_edges(image)
    center_y, center_x, radius, _ = detect_circles(edges, 10, 30)[0]
    assert abs(center_y - 32) <= 2 and abs(center_x - 32) <= 2
    assert abs(radius - 20) <= 2
 
 
def test_s3_two_rings():
    """S3: два кільця різного радіуса знаходяться обидва."""
    edges = pipeline1_edges(make_sign(100, [(30, 30, 12), (65, 65, 25)]))
    circles = detect_circles(edges, 8, 30, max_circles=2)
    found = sorted((c[0], c[1], c[2]) for c in circles)
    assert len(found) == 2
    for (y_f, x_f, r_f), (y_e, x_e, r_e) in zip(found, [(30, 30, 12),
                                                         (65, 65, 25)]):
        assert abs(y_f - y_e) <= 1 and abs(x_f - x_e) <= 1
        assert abs(r_f - r_e) <= 1
 
 
def test_s4_no_edges():
    """S4: порожній контур (немає червоного) - жодного кола, без помилки."""
    assert not detect_circles(np.zeros((32, 32), dtype=bool), 5, 15)
 
 
def test_circle_to_box_clipped():
    """Прямокутник кола обрізається межами зображення."""
    assert circle_to_box(2, 3, 5, (10, 10)) == (0, 0, 8, 7)
 
 
def test_wrong_radius():
    """Радіус 0 викликає ValueError."""
    with pytest.raises(ValueError):
        hough_circles(CIRCLE_R1_5X5, [0, 1])
 
