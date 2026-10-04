# PlantCare AI — Crop Disease Model

## Recommended dataset strategy

Use a crop-focused subset of PlantVillage for the first training run. PlantVillage contains about 54,305 leaf images across 38 crop-disease combinations. It is useful for initial training, but its controlled backgrounds do not represent all real farm photos.

For stronger real-world validation, keep a separate field-condition test set such as PlantDoc/FieldPlant. Do NOT mix the validation/test images into training.

## Folder format

Place the training data like this:

dataset/
  train/
    Tomato___Early_blight/
    Tomato___Late_blight/
    Tomato___healthy/
    Potato___Early_blight/
    Potato___Late_blight/
    Potato___healthy/
    ...
  val/
    ...
  test/
    ...

You can start with the crop classes available in your selected dataset and expand later.

## Training

Install:
    pip install -r requirements-ml.txt

Then:
    python ml/train.py --data dataset --epochs 15

The script uses EfficientNet-B0 transfer learning, augmentation, class-weight support, early stopping and saves the best model.

## Important

The website should display "confidence" rather than calling confidence an accuracy percentage. Overall accuracy, macro F1, per-class recall and a confusion matrix should be calculated on a held-out test set.

## Plant/non-plant image screening

The hosted disease classifier always chooses among its known disease labels, so it cannot reject unrelated photos by itself. `v4.html` now first runs a general ImageNet image classifier in the browser and only continues when one of its top five labels matches a plant-related category. This is an untrained, best-effort filter: it can reject unusual plant photos and may still accept an unrelated image that resembles a plant category. The screening model needs internet access the first time it is loaded.

An optional custom binary gate training workflow is also available:

1. Install the ML dependencies:
   `pip install -r requirements-ml.txt`
2. From this folder, prepare public PlantVillage leaf photos and TensorFlow Flowers photos as plant examples, with CIFAR-10 images as non-plant examples:
   `python prepare_plant_gate_dataset.py`
3. Train the gate and export the browser ONNX model:
   `python train_plant_gate.py`

The scripts write `model/plant_gate.onnx` and `model/plant_gate_config.json`; `v4.html` does not currently load that custom gate.

The public datasets are a starter set, not a guarantee that every non-plant photo will be rejected. CIFAR-10 images are small, and the plant examples do not cover every kind of plant photo or camera condition. Review the held-out `model/plant_gate_test_report.json` and test the gate with representative photos before relying on it. Check each dataset's license and terms before redistributing images or trained weights.
