import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from methods.bilateral import bilateral_filter
from methods.metrics import iou, mse, psnr, ssim
from pipelines import pipeline1, pipeline2
from traffic_signs_stem import list_images, load_annotations

plt.switch_backend("Agg")  # малювати у файли, без вікон

NOISE_LEVELS = (0, 10, 25)
SEED = 42
GROUPS = ("circle_red", "triangle_red", "circle_blue")
GROUP_NAMES = {"circle_red": "круглі червоні", "triangle_red": "трикутні",
               "circle_blue": "сині"}
METHOD_NAMES = {"frame": "рамка «весь кадр»", "pipeline1": "пайплайн 1",
                "pipeline2": "пайплайн 2"}
COLORS = {"pipeline1": "#2a78d6", "pipeline2": "#eb6834", "frame": "#9a9a94"}


def parse_args():
    """Розбирає аргументи командного рядка."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/images")
    parser.add_argument("--annotations", default="data/annotations.csv")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output", default="results/experiments")
    return parser.parse_args()


def add_noise(image, sigma, rng):
    """Додає гаусівський шум із середнім 0 і стандартним відхиленням sigma."""
    if sigma == 0:
        return image.copy()
    noisy = image.astype(np.float64) + rng.normal(0.0, sigma, image.shape)
    return np.clip(np.round(noisy), 0, 255).astype(np.uint8)


def brightness_groups(images):
    """Ділить зображення на три рівні групи за середньою яскравістю V.

    Межі - 33-й і 67-й перцентилі середньої яскравості по вибірці.
    """
    means = {name: image.max(axis=2).mean() for name, image in images.items()}
    low, high = np.percentile(list(means.values()), [100 / 3, 200 / 3])
    return {name: ("темні" if value < low else
                   "середні" if value < high else "світлі")
            for name, value in means.items()}


def detect_all(image, config):
    """Запускає обидва пайплайни й базову рамку на одному зображенні.

    Returns:
        Словник {метод: (рамка або None, час у секундах)}.
    """
    height, width = image.shape[:2]
    results = {"frame": ((0, 0, width - 1, height - 1), 0.0)}
    for method, pipeline in (("pipeline1", pipeline1), ("pipeline2", pipeline2)):
        start = time.perf_counter()
        _, box, _ = pipeline(image, config[method])
        results[method] = (box, time.perf_counter() - start)
    return results


def run_detection(images, annotations, config):
    """Запускає обидва пайплайни й базову рамку для всіх рівнів шуму.

    Returns:
        Список рядків: зображення, група, шум, метод, IoU, знайдено, час.
    """
    rows = []
    for sigma in NOISE_LEVELS:
        rng = np.random.default_rng(SEED)
        for name, image in images.items():
            group, truth = annotations[name]
            results = detect_all(add_noise(image, sigma, rng), config)
            for method, (box, elapsed) in results.items():
                rows.append({"filename": name, "group": group,
                             "sigma": sigma, "method": method,
                             "found": box is not None,
                             "iou": round(iou(box, truth), 4),
                             "time_s": round(elapsed, 4)})
        print(f"  шум sigma = {sigma}: готово")
    return rows


def run_filter_quality(images, params):
    """Оцінює білатеральний фільтр: метрики до і після фільтрації шуму."""
    rows = []
    for sigma in NOISE_LEVELS[1:]:
        rng = np.random.default_rng(SEED)
        for name, image in images.items():
            noisy = add_noise(image, sigma, rng)
            filtered = bilateral_filter(noisy, **params)
            rows.append({"filename": name, "sigma": sigma,
                         "mse_noisy": round(mse(image, noisy), 2),
                         "mse_filtered": round(mse(image, filtered), 2),
                         "psnr_noisy": round(psnr(image, noisy), 2),
                         "psnr_filtered": round(psnr(image, filtered), 2),
                         "ssim_noisy": round(ssim(image, noisy), 4),
                         "ssim_filtered": round(ssim(image, filtered), 4)})
    return rows


def summarize(rows, keys):
    """Групує рядки за keys і рахує середній IoU, частки й середній час."""
    table = {}
    for row in rows:
        table.setdefault(tuple(row[k] for k in keys), []).append(row)
    summary = []
    for key, items in sorted(table.items(), key=lambda kv: str(kv[0])):
        values = np.array([item["iou"] for item in items])
        summary.append({**dict(zip(keys, key)), "n": len(items),
                        "found": round(np.mean([i["found"] for i in items]), 3),
                        "iou_mean": round(values.mean(), 3),
                        "iou_std": round(values.std(), 3),
                        "iou_ge_0_5": round(np.mean(values >= 0.5), 3),
                        "iou_ge_0_75": round(np.mean(values >= 0.75), 3),
                        "time_mean_s": round(np.mean([i["time_s"]
                                                      for i in items]), 4)})
    return summary


def save_csv(rows, path):
    """Зберігає список словників у CSV."""
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def lookup(summary, **conditions):
    """Повертає перший рядок зведення, що відповідає всім умовам."""
    for row in summary:
        if all(row[k] == v for k, v in conditions.items()):
            return row
    return None


def style(axis, title, ylabel):
    """Спільне оформлення графіків: заголовок, підпис осі, легка сітка."""
    axis.set_title(title, fontsize=12)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="#e4e3dd", linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def plot_groups(summary, path):
    """Графік 1: частка IoU >= 0.75 за групами знаків (без шуму)."""
    fig, axis = plt.subplots(figsize=(8, 4.5))
    methods = ("pipeline1", "pipeline2", "frame")
    positions = np.arange(len(GROUPS))
    width = 0.26
    for k, method in enumerate(methods):
        values = [100 * lookup(summary, sigma=0, method=method,
                               group=g)["iou_ge_0_75"] for g in GROUPS]
        bars = axis.bar(positions + (k - 1) * width, values, width * 0.92,
                        color=COLORS[method], label=METHOD_NAMES[method])
        axis.bar_label(bars, fmt="%.0f%%", fontsize=9, padding=2)
    axis.set_xticks(positions, [GROUP_NAMES[g] for g in GROUPS])
    axis.set_ylim(0, 100)
    style(axis, "Точні виявлення (IoU ≥ 0,75) за групами знаків, без шуму",
          "частка зображень, %")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_noise(summary, path):
    """Графік 2: як частка IoU >= 0.75 змінюється зі зростанням шуму."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for axis, group, title in ((axes[0], "circle_red", "Круглі червоні знаки"),
                               (axes[1], None, "Усі знаки")):
        for method in ("pipeline1", "pipeline2"):
            values = [100 * (lookup(summary, sigma=s, method=method, group=group)
                             if group else
                             lookup(summary, sigma=s, method=method,
                                    group="усі"))["iou_ge_0_75"]
                      for s in NOISE_LEVELS]
            axis.plot(NOISE_LEVELS, values, marker="o", linewidth=2,
                      markersize=7, color=COLORS[method],
                      label=METHOD_NAMES[method])
            axis.annotate(f"{values[-1]:.0f}%", (NOISE_LEVELS[-1], values[-1]),
                          textcoords="offset points", xytext=(6, 0),
                          va="center", fontsize=9)
        axis.set_xticks(NOISE_LEVELS)
        axis.set_xlabel("рівень шуму sigma")
        style(axis, title, "частка IoU ≥ 0,75, %")
    axes[0].set_ylim(0, 100)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_filter(filter_rows, path):
    """Графік 3: PSNR і SSIM до і після білатерального фільтра."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    sigmas = NOISE_LEVELS[1:]
    positions = np.arange(len(sigmas))
    for axis, metric, label in ((axes[0], "psnr", "PSNR, дБ"),
                                (axes[1], "ssim", "SSIM")):
        for k, (suffix, name, color) in enumerate(
                (("noisy", "з шумом", "#9a9a94"),
                 ("filtered", "після фільтра", COLORS["pipeline1"]))):
            values = [np.mean([r[f"{metric}_{suffix}"] for r in filter_rows
                               if r["sigma"] == s]) for s in sigmas]
            bars = axis.bar(positions + (k - 0.5) * 0.36, values, 0.33,
                            color=color, label=name)
            axis.bar_label(bars, fmt="%.2f" if metric == "ssim" else "%.1f",
                           fontsize=9, padding=2)
        axis.set_xticks(positions, [f"sigma = {s}" for s in sigmas])
        style(axis, f"{label}: більше - ближче до оригіналу", label)
    axes[1].set_ylim(0, 1)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_brightness(summary, path):
    """Графік 4: круглі червоні знаки за освітленням (без шуму)."""
    fig, axis = plt.subplots(figsize=(8, 4.5))
    levels = ("темні", "середні", "світлі")
    positions = np.arange(len(levels))
    for k, method in enumerate(("pipeline1", "pipeline2")):
        values = [100 * lookup(summary, method=method, brightness=b)
                  ["iou_ge_0_75"] for b in levels]
        bars = axis.bar(positions + (k - 0.5) * 0.36, values, 0.33,
                        color=COLORS[method], label=METHOD_NAMES[method])
        axis.bar_label(bars, fmt="%.0f%%", fontsize=9, padding=2)
    axis.set_xticks(positions, levels)
    axis.set_ylim(0, 100)
    style(axis, "Круглі червоні знаки за освітленням (IoU ≥ 0,75), без шуму",
          "частка зображень, %")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_example(image, params, path):
    """Графік 5: приклад зображення до і після фільтрації шуму."""
    rng = np.random.default_rng(SEED)
    noisy = add_noise(image, 25, rng)
    filtered = bilateral_filter(noisy, **params)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4))
    for axis, picture, title in (
            (axes[0], image, "оригінал"),
            (axes[1], noisy, f"шум sigma = 25\nPSNR {psnr(image, noisy):.1f} дБ"),
            (axes[2], filtered,
             f"після фільтра\nPSNR {psnr(image, filtered):.1f} дБ")):
        axis.imshow(picture, interpolation="nearest")
        axis.set_title(title, fontsize=11)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    """Точка входу: усі експерименти, таблиці й графіки."""
    args = parse_args()
    with open(args.config, encoding="utf-8") as file:
        config = json.load(file)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    annotations = load_annotations(args.annotations)
    images = {path.name: np.array(Image.open(path).convert("RGB"))
              for path in list_images(args.input) if path.name in annotations}
    print(f"Зображень: {len(images)}")

    print("1. Виявлення знаків на різних рівнях шуму...")
    detection = run_detection(images, annotations, config)
    brightness = brightness_groups(images)
    for row in detection:
        row["brightness"] = brightness[row["filename"]]
    save_csv(detection, output / "detection.csv")

    by_group = summarize(detection, ("sigma", "method", "group"))
    overall = summarize([{**r, "group": "усі"} for r in detection],
                        ("sigma", "method", "group"))
    save_csv(by_group + overall, output / "summary_by_group.csv")
    by_light = summarize([r for r in detection if r["sigma"] == 0
                          and r["group"] == "circle_red"],
                         ("method", "brightness"))
    save_csv(by_light, output / "summary_by_brightness.csv")

    print("2. Якість білатерального фільтра...")
    filter_rows = run_filter_quality(images, config["pipeline2"]["bilateral"])
    save_csv(filter_rows, output / "filter_quality.csv")

    print("3. Графіки...")
    plot_groups(by_group, output / "1_groups.png")
    plot_noise(by_group + overall, output / "2_noise.png")
    plot_filter(filter_rows, output / "3_filter.png")
    plot_brightness(by_light, output / "4_brightness.png")
    plot_example(images[sorted(images)[0]], config["pipeline2"]["bilateral"],
                 output / "5_filter_example.png")

    print(f"\nГотово, результати в {output}\n")
    print("Точні виявлення (IoU >= 0.75), усі знаки:")
    for sigma in NOISE_LEVELS:
        cells = [f"{METHOD_NAMES[m]}: "
                 f"{lookup(overall, sigma=sigma, method=m)['iou_ge_0_75']:.0%}"
                 for m in ("pipeline1", "pipeline2", "frame")]
        print(f"  sigma = {sigma:2d} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
