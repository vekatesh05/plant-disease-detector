# PlantCare AI — Training Guide

## Easiest option: Google Colab

A GPU is strongly recommended for training.

1. Open Google Colab.
2. Upload the PlantCare AI project ZIP.
3. Extract it.
4. Install dependencies:

```bash
pip install -r requirements-ml.txt
pip install tensorflow-datasets
```

5. Prepare the crop-focused PlantVillage dataset:

```bash
python ml/prepare_plantvillage.py
```

6. Train:

```bash
python ml/train.py --data dataset --epochs 15
```

7. The best/final model will be saved in:

```text
model/plantcare_final.keras
```

8. Copy the generated `model/` folder into the Flask project before running the website.

## Crop focus

The first version selects:
- Apple
- Corn
- Grape
- Peach
- Bell Pepper
- Potato
- Soybean
- Strawberry
- Tomato

This is intentionally crop-focused rather than trying to support every possible plant.

## Important evaluation rule

Do not report training accuracy as the project's final accuracy. Use the held-out test set and report:
- Test accuracy
- Macro precision
- Macro recall
- Macro F1
- Per-class performance
- Confusion matrix

Also test a small set of real photographs taken outside the dataset. A model can perform very well on controlled images and still struggle with real-world lighting, backgrounds, camera quality, or multiple leaves.

## Model interpretation

The website should call the output a prediction confidence, not "accuracy". A confidence score such as 92% is not the same thing as 92% model accuracy.
