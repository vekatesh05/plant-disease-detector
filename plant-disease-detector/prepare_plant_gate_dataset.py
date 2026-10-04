from __future__ import annotations

import hashlib
from pathlib import Path

import tensorflow_datasets as tfds
from PIL import Image


OUTPUT = Path("plant_gate_dataset")


def split_for_image(image) -> str:
    digest = hashlib.sha256(image.tobytes()).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def save_image(image, split: str, category: str, filename: str) -> None:
    folder = OUTPUT / split / category
    folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).convert("RGB").save(folder / filename, quality=92)


def prepare_plantvillage() -> int:
    print("Loading PlantVillage leaf photos...")
    dataset = tfds.load(
        "plant_village",
        split="train",
        as_supervised=True,
        shuffle_files=False,
    )
    count = 0

    for index, (image, _) in enumerate(tfds.as_numpy(dataset)):
        split = split_for_image(image)
        save_image(image, split, "plant", f"plantvillage_{index:07d}.jpg")
        count += 1
        if count % 5000 == 0:
            print(f"Prepared {count} PlantVillage images...")

    return count


def prepare_flowers() -> int:
    print("Loading TensorFlow Flowers plant photos...")
    dataset = tfds.load(
        "tf_flowers",
        split="train",
        as_supervised=True,
        shuffle_files=False,
    )
    count = 0

    for index, (image, _) in enumerate(tfds.as_numpy(dataset)):
        split = split_for_image(image)
        save_image(image, split, "plant", f"flower_{index:05d}.jpg")
        count += 1

    return count


def prepare_cifar10() -> tuple[int, int]:
    print("Loading CIFAR-10 non-plant photos...")
    train_dataset = tfds.load(
        "cifar10",
        split="train",
        as_supervised=True,
        shuffle_files=False,
    )
    test_dataset = tfds.load(
        "cifar10",
        split="test",
        as_supervised=True,
        shuffle_files=False,
    )
    train_count = 0
    test_count = 0

    for index, (image, _) in enumerate(tfds.as_numpy(train_dataset)):
        split = split_for_image(image)
        save_image(image, split, "non_plant", f"cifar_train_{index:05d}.jpg")
        train_count += 1

    for index, (image, _) in enumerate(tfds.as_numpy(test_dataset)):
        save_image(image, "test", "non_plant", f"cifar_test_{index:05d}.jpg")
        test_count += 1

    return train_count, test_count


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plant_village_count = prepare_plantvillage()
    flower_count = prepare_flowers()
    cifar_train_count, cifar_test_count = prepare_cifar10()

    print("Plant gate dataset is ready.")
    print(f"PlantVillage leaf images: {plant_village_count}")
    print(f"Flower photos: {flower_count}")
    print(f"CIFAR-10 train images: {cifar_train_count}")
    print(f"CIFAR-10 test images: {cifar_test_count}")
    print(f"Dataset folders: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
