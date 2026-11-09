import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from methods.metrics import iou
from pipelines import PIPELINES

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".ppm"}


def parse_args():
    """Розбирає аргументи командного рядка."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=sorted(PIPELINES),
                        default="1", help="номер пайплайна")
    parser.add_argument("--input", default="data/images",
                        help="зображення або папка із зображеннями")
    parser.add_argument("--annotations", default="data/annotations.csv",
                        help="CSV з розміткою (для обчислення IoU)")
    parser.add_argument("--config", default="config.json",
                        help="файл параметрів")
    parser.add_argument("--output", default="results",
                        help="папка для результатів")
    return parser.parse_args()


def list_images(path):
    """Повертає список файлів-зображень: один файл або вміст папки."""
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(p for p in path.iterdir()
                  if p.suffix.lower() in IMAGE_EXTENSIONS)


def load_annotations(path):
    """Читає розмітку: {ім'я файлу: (group, (x1, y1, x2, y2))}."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as file:
        return {row["filename"]: (row["group"],
                                  tuple(int(row[k]) for k in
                                        ("x1", "y1", "x2", "y2")))
                for row in csv.DictReader(file)}


def save_stages(stages, folder):
    """Зберігає кожну стадію як PNG у вказану папку."""
    folder.mkdir(parents=True, exist_ok=True)
    for name, image in stages.items():
        Image.fromarray(image).save(folder / f"{name}.png")


def process_image(path, pipeline, params, annotations, output):
    """Обробляє одне зображення і повертає рядок метрик (або None при помилці)."""
    try:
        image = np.array(Image.open(path).convert("RGB"))
    except (OSError, UnidentifiedImageError) as error:
        print(f"Пропущено {path.name}: не вдалося відкрити ({error})")
        return None

    start = time.perf_counter()
    stages, box, score = pipeline(image, params)
    elapsed = time.perf_counter() - start

    save_stages(stages, output / path.stem)
    group, truth = annotations.get(path.name, ("unknown", None))
    return {
        "filename": path.name,
        "group": group,
        "found": box is not None,
        "box": list(box) if box else None,
        "truth": list(truth) if truth else None,
        "score": round(score, 4),
        "iou": round(iou(box, truth), 4) if truth else None,
        "time_s": round(elapsed, 4),
    }


def save_metrics(rows, output):
    """Записує метрики в metrics.csv і metrics.json."""
    with open(output / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)
    with open(output / "metrics.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    """Виводить підсумок метрик загалом і по групах зображень. """
    groups = sorted({row["group"] for row in rows})
    print(f"{'група':>13} | {'n':>3} | {'found':>5} | {'IoU сер.':>8} | "
          f"{'>=0.5':>5} | {'>=0.75':>6}")
    for group in ["усі"] + groups:
        selected = rows if group == "усі" else [
            row for row in rows if row["group"] == group]
        values = np.array([row["iou"] for row in selected
                           if row["iou"] is not None])
        found = np.mean([row["found"] for row in selected])
        if values.size:
            print(f"{group:>13} | {len(selected):3d} | {found:5.0%} | "
                  f"{values.mean():8.3f} | {np.mean(values >= 0.5):5.0%} | "
                  f"{np.mean(values >= 0.75):6.0%}")


def main():
    """Точка входу програми."""
    args = parse_args()
    with open(args.config, encoding="utf-8") as file:
        params = json.load(file)[f"pipeline{args.pipeline}"]

    output = Path(args.output) / f"pipeline{args.pipeline}"
    output.mkdir(parents=True, exist_ok=True)
    annotations = load_annotations(args.annotations)
    pipeline = PIPELINES[args.pipeline]

    rows = []
    for path in list_images(args.input):
        row = process_image(path, pipeline, params, annotations, output)
        if row is not None:
            rows.append(row)

    if not rows:
        print("Не оброблено жодного зображення")
        return
    save_metrics(rows, output)
    print(f"Оброблено {len(rows)} зображень, результати в {output}")
    print_summary(rows)


if __name__ == "__main__":
    main()