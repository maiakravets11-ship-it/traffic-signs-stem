import argparse
import csv
import random
import shutil
from pathlib import Path
 
# Класи GTSRB за групами
CIRCLE_RED = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16, 17]
TRIANGLE_RED = [11, 13] + list(range(18, 32))
CIRCLE_BLUE = list(range(33, 41))
 
GROUPS = {
    "circle_red": (CIRCLE_RED, 70),
    "triangle_red": (TRIANGLE_RED, 15),
    "circle_blue": (CIRCLE_BLUE, 15),
}
 
SEED = 42
MIN_SIDE = 30  # дуже малі зображення відкидаємо
 
 
def find_test_csv(source):
    """Повертає шлях до Test.csv усередині розпакованого архіву."""
    matches = list(Path(source).rglob("Test.csv"))
    if not matches:
        raise FileNotFoundError(f"Test.csv не знайдено в {source}")
    return matches[0]
 
 
def read_rows(csv_path):
    """Читає рядки розмітки, залишаючи лише зображення достатнього розміру."""
    with open(csv_path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return [r for r in rows
            if min(int(r["Width"]), int(r["Height"])) >= MIN_SIDE]
 
 
def select(rows):
    """Відбирає зображення для кожної групи з фіксованим seed."""
    rng = random.Random(SEED)
    selected = []
    for group, (classes, count) in GROUPS.items():
        candidates = [r for r in rows if int(r["ClassId"]) in classes]
        for row in rng.sample(candidates, count):
            selected.append((group, row))
    return selected
 
 
def main():
    """Точка входу: відбір, копіювання файлів і збереження розмітки."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="папка з розпакованим GTSRB")
    parser.add_argument("target", help="папка data проєкту")
    args = parser.parse_args()
 
    csv_path = find_test_csv(args.source)
    root = csv_path.parent
    images_dir = Path(args.target) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
 
    selected = select(read_rows(csv_path))
    out_csv = Path(args.target) / "annotations.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "group", "class_id", "width", "height",
                         "x1", "y1", "x2", "y2"])
        for group, row in selected:
            src = root / row["Path"]
            name = Path(row["Path"]).name
            shutil.copy(src, images_dir / name)
            writer.writerow([name, group, row["ClassId"], row["Width"],
                             row["Height"], row["Roi.X1"], row["Roi.Y1"],
                             row["Roi.X2"], row["Roi.Y2"]])
 
    print(f"Скопійовано {len(selected)} зображень у {images_dir}")
    print(f"Розмітку збережено в {out_csv}")
 
 
if __name__ == "__main__":
    main()
 
