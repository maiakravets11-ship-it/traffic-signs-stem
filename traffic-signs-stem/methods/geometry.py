import numpy as np


def bilinear_sample(image, y_coords, x_coords):
    """Бере значення зображення в дробових координатах (білінійна інтерполяція)."""
    image = np.asarray(image, dtype=np.float64)
    height, width = image.shape[:2]
    y_coords = np.clip(y_coords, 0, height - 1)
    x_coords = np.clip(x_coords, 0, width - 1)

    y_0 = np.floor(y_coords).astype(int)
    x_0 = np.floor(x_coords).astype(int)
    y_1 = np.minimum(y_0 + 1, height - 1)
    x_1 = np.minimum(x_0 + 1, width - 1)
    frac_y = y_coords - y_0
    frac_x = x_coords - x_0
    if image.ndim == 3:
        frac_y = frac_y[..., None]
        frac_x = frac_x[..., None]

    top = image[y_0, x_0] * (1 - frac_x) + image[y_0, x_1] * frac_x
    bottom = image[y_1, x_0] * (1 - frac_x) + image[y_1, x_1] * frac_x
    return top * (1 - frac_y) + bottom * frac_y


def affine_transform(image, matrix, out_shape):
    """Застосовує афінне перетворення (зворотне відображення)."""
    matrix = np.asarray(matrix, dtype=np.float64)
    out_h, out_w = out_shape
    y_out, x_out = np.indices((out_h, out_w), dtype=np.float64)
    x_src = matrix[0, 0] * x_out + matrix[0, 1] * y_out + matrix[0, 2]
    y_src = matrix[1, 0] * x_out + matrix[1, 1] * y_out + matrix[1, 2]
    result = bilinear_sample(image, y_src, x_src)
    return np.clip(np.round(result), 0, 255).astype(np.uint8)


def box_to_matrix(box, out_size):
    """Будує матрицю, що переводить квадрат out_size x out_size у прямокутник box.

    Масштаб s = (розмір рамки) / out_size, зсув - лівий верхній кут рамки.
    Центри пікселів узгоджуються зсувом на 0.5:
        x_src = x1 + (x_out + 0.5) * s_x - 0.5. """
    x_1, y_1, x_2, y_2 = box
    scale_x = (x_2 - x_1 + 1) / out_size
    scale_y = (y_2 - y_1 + 1) / out_size
    return np.array([[scale_x, 0.0, x_1 + 0.5 * scale_x - 0.5],
                     [0.0, scale_y, y_1 + 0.5 * scale_y - 0.5]])


def normalize_sign(image, box, out_size=48):
    """Вирізає рамку box і масштабує її до квадрата out_size x out_size."""
    return affine_transform(image, box_to_matrix(box, out_size),
                            (out_size, out_size))
