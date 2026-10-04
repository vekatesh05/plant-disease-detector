"""
Download PlantVillage through TensorFlow Datasets and create a crop-focused
train/validation/test folder dataset.

Default crops:
Apple, Corn, Grape, Peach, Pepper, Potato, Soybean, Strawberry, Tomato

Run:
    pip install tensorflow tensorflow-datasets pillow
    python ml/prepare_plantvillage.py

The full PlantVillage TFDS dataset is ~54k images. This script keeps only
the selected crop classes and creates an 80/10/10 split per class.
"""

from pathlib import Path
import hashlib
import shutil
import tensorflow_datasets as tfds
from PIL import Image
import numpy as np

OUT = Path("dataset")
SEED = 42

# Crop-focused first version. Add/remove crops after the first evaluation.
CROPS = {
    "Apple", "Corn_(maize)", "Grape", "Peach",
    "Pepper,_bell", "Potato", "Soybean",
    "Strawberry", "Tomato"
}

def safe_name(label):
    return label.replace("___", "___").replace("/", "_")

def save_example(image, label, filename):
    cls = safe_name(label)
    # 80/10/10 deterministic split using filename hash.
    h = int(hashlib.md5(filename.encode("utf-8")).hexdigest(), 16) % 100
    split = "train" if h < 80 else ("val" if h < 90 else "test")
    folder = OUT / split / cls
    folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(folder / filename)

def main():
    print("Downloading/loading PlantVillage through TensorFlow Datasets...")
    ds, info = tfds.load(
        "plant_village",
        split="train",
        as_supervised=True,
        with_info=True,
        shuffle_files=False
    )

    names = info.features["label"].names
    selected = 0

    for image, label_id in tfds.as_numpy(ds):
        label = names[int(label_id)]
        plant = label.split("___", 1)[0]

        if plant not in CROPS:
            continue

        filename = f"{selected:07d}.jpg"
        save_example(image, label, filename)
        selected += 1

        if selected % 1000 == 0:
            print(f"Prepared {selected} images...")

    print(f"Finished. Prepared {selected} crop images.")
    print(f"Dataset folders created under: {OUT.resolve()}")

if __name__ == "__main__":
    main()
