import numpy as np

from methods.region_growing import (border_seeds, connected_component,
                                    mask_to_box, region_grow, segment_sign)

# Ліві 3 стовпці - яскравість 10, праві 2 - яскравість 200
TWO_REGIONS_5X5 = np.array([[10, 10, 10, 200, 200]] * 5, dtype=np.uint8)


def test_region_grow_stops_at_edge():
    """Від (0, 0) з порогом 20 область заповнює лише ліві 3 стовпці."""
    expected = np.zeros((5, 5), dtype=bool)
    expected[:, :3] = True
    result = region_grow(TWO_REGIONS_5X5, [(0, 0)], threshold=20)
    np.testing.assert_array_equal(result, expected)


def test_region_grow_large_threshold_takes_all():
    """З порогом 300 (більшим за перепад 190) область займає все зображення."""
    assert region_grow(TWO_REGIONS_5X5, [(0, 0)], threshold=300).all()


def test_border_seeds_3x3():
    """Крайові пікселі 3x3 - усі, крім центру (8 різних точок)."""
    assert set(border_seeds((3, 3))) == {(y, x) for y in range(3)
                                         for x in range(3)} - {(1, 1)}


def test_connected_component_and_box():
    """Дві окремі плями: беремо ту, що містить точку, і її рамку."""
    mask = np.zeros((5, 5), dtype=bool)
    mask[0:2, 0:2] = True
    mask[3:5, 2:5] = True
    part = connected_component(mask, (4, 4))
    assert part.sum() == 6
    assert mask_to_box(part) == (2, 3, 4, 4)
    assert mask_to_box(np.zeros((3, 3), dtype=bool)) is None


def test_segment_sign_synthetic():
    """Червоний диск на сірому фоні виділяється з точною рамкою."""
    y_coords, x_coords = np.mgrid[:40, :40]
    image = np.full((40, 40, 3), [90, 110, 90], dtype=np.uint8)
    image[np.hypot(y_coords - 20, x_coords - 20) <= 14] = [200, 30, 40]
    mask, _ = segment_sign(image, range(5, 61, 5), min_area=0.35)
    assert mask_to_box(mask) == (6, 6, 34, 34)
