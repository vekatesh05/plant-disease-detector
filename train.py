import argparse, json
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = (224, 224)
BATCH = 32
SEED = 42

def make_ds(folder, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        folder,
        image_size=IMG_SIZE,
        batch_size=BATCH,
        label_mode="int",
        shuffle=shuffle,
        seed=SEED
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()

    data = Path(args.data)
    train = make_ds(data / "train", True)
    val = make_ds(data / "val", False)
    test = make_ds(data / "test", False)

    class_names = train.class_names
    (Path("model")).mkdir(exist_ok=True)
    Path("model/class_names.json").write_text(json.dumps(class_names, indent=2))

    # Compute class weights from training labels.
    labels = np.concatenate([y.numpy() for _, y in train], axis=0)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(class_names)),
        y=labels
    )
    class_weights = dict(enumerate(weights.astype(float)))

    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.12),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.12),
    ], name="augmentation")

    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3)
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = aug(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "model/plantcare_best.keras", monitor="val_accuracy",
            save_best_only=True
        )
    ]

    model.fit(
        train, validation_data=val, epochs=args.epochs,
        class_weight=class_weights, callbacks=callbacks
    )

    # Unfreeze the upper part for transfer-learning fine tuning.
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train, validation_data=val, epochs=max(5, args.epochs // 2),
        class_weight=class_weights, callbacks=callbacks
    )

    model.save("model/plantcare_final.keras")

    y_true, y_pred = [], []
    for x, y in test:
        p = model.predict(x, verbose=0)
        y_true.extend(y.numpy().tolist())
        y_pred.extend(np.argmax(p, axis=1).tolist())

    report = classification_report(
        y_true, y_pred, target_names=class_names,
        output_dict=True, zero_division=0
    )
    Path("model/test_report.json").write_text(json.dumps(report, indent=2))
    np.save("model/confusion_matrix.npy", confusion_matrix(y_true, y_pred))

    print("Classes:", len(class_names))
    print("Test accuracy:", report["accuracy"])
    print("Model saved to model/plantcare_final.keras")

if __name__ == "__main__":
    main()
