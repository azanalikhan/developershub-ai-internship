"""
Generates a small synthetic tabular + image housing dataset so the
multimodal pipeline (dataset.py / model.py / train.py / evaluate.py) is
runnable end-to-end without needing to download a real image dataset.

Each synthetic "house photo" is a procedurally rendered colored image whose
tone/pattern is correlated with price (loosely standing in for real signals
like curb appeal, finish quality, or lot greenery in an actual photo dataset).

Usage:
    python generate_synthetic_data.py --n_samples 600
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

DATA_DIR = Path(__file__).parent / "data"
IMG_DIR = DATA_DIR / "images"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples", type=int, default=600)
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def render_house_image(path, area, condition, location_score, img_size, rng):
    """Procedurally render a simple 'house facade' whose color/quality tracks price."""
    img = Image.new("RGB", (img_size, img_size), color=(135, 206, 235))  # sky
    draw = ImageDraw.Draw(img)

    # Better condition -> brighter, cleaner facade color; bigger area -> bigger house block
    quality = np.clip(condition / 10 + location_score / 10, 0, 1)
    base_color = (
        int(120 + 100 * quality + rng.normal(0, 5)),
        int(90 + 80 * quality + rng.normal(0, 5)),
        int(70 + 40 * (1 - quality) + rng.normal(0, 5)),
    )
    base_color = tuple(int(np.clip(c, 0, 255)) for c in base_color)

    house_w = int(np.clip(img_size * (0.4 + 0.4 * (area / 4000)), img_size * 0.3, img_size * 0.9))
    house_h = int(house_w * 0.6)
    x0 = (img_size - house_w) // 2
    y0 = img_size - house_h - 8
    draw.rectangle([x0, y0, x0 + house_w, y0 + house_h], fill=base_color, outline=(30, 30, 30))

    # Roof
    draw.polygon(
        [(x0 - 4, y0), (x0 + house_w // 2, y0 - house_h // 3), (x0 + house_w + 4, y0)],
        fill=(80, 40, 40) if quality < 0.5 else (120, 60, 60),
    )
    # Door
    door_w, door_h = house_w // 6, house_h // 2
    draw.rectangle(
        [x0 + house_w // 2 - door_w // 2, y0 + house_h - door_h, x0 + house_w // 2 + door_w // 2, y0 + house_h],
        fill=(60, 40, 20),
    )
    # Lawn (greener for better location score)
    lawn_green = int(100 + 100 * (location_score / 10))
    draw.rectangle([0, img_size - 8, img_size, img_size], fill=(40, lawn_green, 40))

    img.save(path)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    DATA_DIR.mkdir(exist_ok=True)
    IMG_DIR.mkdir(exist_ok=True)

    locations = ["Downtown", "Suburb", "Rural", "Waterfront"]
    location_base_price = {"Downtown": 1.25, "Suburb": 1.0, "Rural": 0.75, "Waterfront": 1.5}
    location_score = {"Downtown": 8, "Suburb": 6, "Rural": 3, "Waterfront": 9}

    rows = []
    for i in range(args.n_samples):
        area = rng.normal(1800, 600)
        area = float(np.clip(area, 500, 5000))
        bedrooms = int(np.clip(rng.normal(3, 1), 1, 6))
        bathrooms = int(np.clip(rng.normal(2, 0.8), 1, 5))
        floors = int(rng.choice([1, 1, 2, 2, 3], p=[0.3, 0.2, 0.3, 0.15, 0.05]))
        year_built = int(rng.integers(1950, 2024))
        location = rng.choice(locations)
        condition = int(np.clip(rng.normal(6, 2), 1, 10))

        base = 100_000 + area * 120 * location_base_price[location]
        base += bedrooms * 8_000 + bathrooms * 6_000 + floors * 5_000
        base += (condition - 5) * 7_000
        base += (year_built - 1950) * 150
        noise = rng.normal(0, 15_000)
        price = max(30_000, base + noise)

        img_name = f"house_{i:04d}.png"
        render_house_image(
            IMG_DIR / img_name, area, condition, location_score[location], args.img_size, rng
        )

        rows.append(
            {
                "id": i,
                "area": area,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "floors": floors,
                "year_built": year_built,
                "location": location,
                "condition": condition,
                "image_path": f"images/{img_name}",
                "price": round(price, 2),
            }
        )

    df = pd.DataFrame(rows)
    out_csv = DATA_DIR / "houses.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} synthetic rows to {out_csv}")
    print(f"Wrote {len(df)} synthetic images to {IMG_DIR}")
    print(df.describe(include="all").T)


if __name__ == "__main__":
    main()
