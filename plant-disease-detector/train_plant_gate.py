from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tf2onnx
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_NAMES = ["non_plant", "plant"]
TARGET_NON_PLANT_ACCEPT_RATE = 0.01


def load_dataset(folder: Path, shuffle: bool):
    return tf.keras.utils.image_dataset_from_directory(
        folder,
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=shuffle,
        seed=42,
    ).prefetch(tf.data.AUTOTUNE)


def collect_plant_probabilities(model, dataset) -> tuple[np.ndarray, np.ndarray]:
    labels = []
    probabilities = []

    for images, batch_labels in dataset:
        batch_probabilities = model.predict(images, verbose=0)
        labels.extend(batch_labels.numpy().tolist())
        probabilities.extend(batch_probabilities[:, 1].tolist())

    return np.asarray(labels, dtype=np.int64), np.asarray(probabilities)


def choose_plant_threshold(labels: np.ndarray, plant_probabilities: np.ndarray) -> float:
    non_plant_probabilities = plant_probabilities[labels == 0]
    if not len(non_plant_probabilities):
        raise ValueError("Validation data must include non-plant images.")

    threshold = np.quantile(
        non_plant_probabilities,
        1 - TARGET_NON_PLANT_ACCEPT_RATE,
        method="higher",
    )
    return float(min(1.0, np.nextafter(threshold, np.inf)))


def build_model() -> tf.keras.Model:
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.12),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomContrast(0.12),
        ],
        name="augmentation",
    )

    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
    )
    backbone.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="images")
    x = augmentation(inputs)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="class_probabilities")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    data_dir = Path("plant_gate_dataset")
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)

    train = load_dataset(data_dir / "train", shuffle=True)
    validation = load_dataset(data_dir / "val", shuffle=False)
    test = load_dataset(data_dir / "test", shuffle=False)

    best_model_path = model_dir / "plant_gate_best.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            best_model_path,
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6,
        ),
    ]

    model = build_model()
    model.fit(train, validation_data=validation, epochs=10, callbacks=callbacks)

    backbone = model.get_layer("efficientnetb0")
    backbone.trainable = True
    for layer in backbone.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train, validation_data=validation, epochs=5, callbacks=callbacks)

    model = tf.keras.models.load_model(best_model_path)
    validation_labels, validation_plant_probabilities = collect_plant_probabilities(
        model, validation
    )
    threshold = choose_plant_threshold(
        validation_labels,
        validation_plant_probabilities,
    )

    test_labels, test_plant_probabilities = collect_plant_probabilities(model, test)
    test_predictions = (test_plant_probabilities >= threshold).astype(np.int64)
    test_report = {
        "threshold": threshold,
        "test_accuracy": float(accuracy_score(test_labels, test_predictions)),
        "plant_recall": float(recall_score(test_labels, test_predictions, pos_label=1)),
        "non_plant_rejection_rate": float(
            recall_score(test_labels, test_predictions, pos_label=0)
        ),
        "confusion_matrix_labels": CLASS_NAMES,
        "confusion_matrix": confusion_matrix(
            test_labels,
            test_predictions,
            labels=[0, 1],
        ).tolist(),
    }

    final_model_path = model_dir / "plant_gate.keras"
    onnx_model_path = model_dir / "plant_gate.onnx"
    model.save(final_model_path)
    tf2onnx.convert.from_keras(
        model,
        input_signature=[
            tf.TensorSpec((None, *IMAGE_SIZE, 3), tf.float32, name="images")
        ],
        opset=17,
        output_path=str(onnx_model_path),
    )

    config = {
        "class_names": CLASS_NAMES,
        "plant_threshold": threshold,
        "input_size": list(IMAGE_SIZE),
        "input_format": "RGB float32 pixels in the 0-255 range",
        "threshold_policy": (
            "Calibrated on validation data to reject approximately 99% "
            "of its non-plant examples."
        ),
    }
    (model_dir / "plant_gate_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    (model_dir / "plant_gate_test_report.json").write_text(
        json.dumps(test_report, indent=2),
        encoding="utf-8",
    )

    print(f"Saved browser model: {onnx_model_path}")
    print(f"Saved browser config: {model_dir / 'plant_gate_config.json'}")
    print(json.dumps(test_report, indent=2))


if __name__ == "__main__":
    main()
