import numpy as np


def disk_kernel(radius):
    """Створює дисковий структурний елемент."""
    if radius < 0:
        raise ValueError("Радіус має бути невід'ємним")
    y_coords, x_coords = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    return x_coords ** 2 + y_coords ** 2 <= radius ** 2


def _check(mask, kernel):
    """Перевіряє вхідні дані й повертає їх як масиви bool."""
    mask = np.asarray(mask, dtype=bool)
    kernel = np.asarray(kernel, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"Маска має бути двовимірною, отримано {mask.shape}")
    if kernel.ndim != 2 or kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("Структурний елемент має бути 2D з непарними розмірами")
    return mask, kernel


def _windows(mask, kernel, pad_value):
    """Повертає зсунуті копії маски для кожної одиниці структурного елемента.

    Маска доповнюється по краях значенням pad_value, після чого
    для кожної позиції (i, j), де kernel[i, j] = True, береться
    зріз розміру маски. Зріз k-ї позиції містить для кожного пікселя
    значення його сусіда з відповідним зсувом.
    """
    pad_y, pad_x = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded = np.pad(mask, ((pad_y, pad_y), (pad_x, pad_x)),
                    constant_values=pad_value)
    height, width = mask.shape
    return [padded[i:i + height, j:j + width]
            for i, j in zip(*np.nonzero(kernel))]


def erode(mask, kernel):
    """Ерозія: піксель лишається 1, якщо всі сусіди під елементом дорівнюють 1.

    За межами зображення вважаємо пікселі одиничними, щоб ерозія
    не «з'їдала» об'єкти лише через те, що вони торкаються краю."""
    mask, kernel = _check(mask, kernel)
    result = np.ones_like(mask)
    for window in _windows(mask, kernel, pad_value=True):
        result &= window
    return result


def dilate(mask, kernel):
    """Дилатація: піксель стає 1, якщо хоча б один сусід під елементом дорівнює 1.    """
    mask, kernel = _check(mask, kernel)
    result = np.zeros_like(mask)
    for window in _windows(mask, kernel[::-1, ::-1], pad_value=False):
        result |= window
    return result


def opening(mask, kernel):
    """Відкриття = ерозія, потім дилатація. Прибирає дрібні плями й виступи."""
    return dilate(erode(mask, kernel), kernel)


def closing(mask, kernel):
    """Закриття = дилатація, потім ерозія. Заповнює дірки й розриви."""
    return erode(dilate(mask, kernel), kernel)


def gradient(mask, kernel):
    """Морфологічний градієнт = дилатація мінус ерозія (контур завтовшки ~2 px)."""
    return dilate(mask, kernel) & ~erode(mask, kernel)


def boundary(mask, kernel=None):
    """Внутрішня межа = маска мінус її ерозія (контур завтовшки 1 px)."""
    if kernel is None:
        kernel = disk_kernel(1)
    mask = np.asarray(mask, dtype=bool)
    return mask & ~erode(mask, kernel)

def fill_holes(mask):
    """Заповнює внутрішні порожнини маски (морфологічна реконструкція)."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"Маска має бути двовимірною, отримано {mask.shape}")
    background = np.zeros_like(mask)
    background[0, :] = background[-1, :] = True
    background[:, 0] = background[:, -1] = True
    background &= ~mask
    cross = disk_kernel(1)
    while True:
        grown = dilate(background, cross) & ~mask
        if np.array_equal(grown, background):
            return ~background
        background = grown