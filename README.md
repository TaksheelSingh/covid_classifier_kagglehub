### ✅ `README.md`

```markdown
# 🦠 COVID-19 Chest X-ray Classification with ResNet18

This project classifies chest X-ray images into **COVID-19**, **Normal**, or **Viral Pneumonia** using a deep learning model (ResNet18). It is trained and evaluated using the **COVID-19 Radiography Dataset**.

---

## 📁 Dataset Structure

The dataset must follow this structure:

```

datasets/COVID-19\_Radiography\_Dataset/
├── train/
│   ├── COVID/
│   │   └── images/
│   ├── Normal/
│   │   └── images/
│   └── Viral Pneumonia/
│       └── images/
├── val/
│   ├── COVID/
│   │   └── images/
│   ├── Normal/
│   │   └── images/
│   └── Viral Pneumonia/
│       └── images/

````

> ✅ Use `images/` inside each class folder in both `train/` and `val/`.

---

## 🧠 Model

- Backbone: **ResNet-18**
- Modified final layer to support 3 classes.
- Trained on resized images (224x224) with normalization.

---

## 🚀 Usage

### 1. 🔧 Install Dependencies
```bash
pip install torch torchvision matplotlib scikit-learn seaborn tqdm
````

### 2. 🏋️‍♂️ Training (optional)

Training script not included in this repo snapshot. You can integrate your own training loop using `torchvision.models.resnet18`.

### 3. 🔍 Inference

Run on validation set and save predictions:

```bash
python inference.py
```

This will output:

* `y_true.npy` – true labels
* `y_pred.npy` – predicted class labels
* `y_prob.npy` – class probabilities from softmax

### 4. 📊 Evaluation

Run ROC curve, confusion matrix, classification report:

```bash
python evaluate.py
```

---

## 📈 Visualizations

* ✅ ROC Curve for each class with AUC
* ✅ Confusion Matrix
* ✅ Classification Report
* ✅ Sample Predictions with Confidence
* ✅ (Optional) Misclassified image viewer

---

## 🧪 Sample Prediction

Run a visual prediction on one image per class:

```bash
python predict_sample_images.py
```

Displays true and predicted labels with confidence scores.

---

## 🔍 Sample Output

<img src="sample_roc_curve.png" width="500"/>
<img src="sample_confusion_matrix.png" width="400"/>

---

## 🗂️ Output Files

| File                       | Description                     |
| -------------------------- | ------------------------------- |
| `y_true.npy`               | Ground truth labels             |
| `y_pred.npy`               | Predicted class indices         |
| `y_prob.npy`               | Predicted probabilities (N x 3) |
| `evaluate.py`              | Evaluation script               |
| `inference.py`             | Inference over validation set   |
| `predict_sample_images.py` | Image-wise prediction viewer    |

---

## 📌 Notes

* Requires proper folder structure (train/val with nested `images/`).
* Model must be saved as `covid_classifier.pth` in project root.
* Evaluation only works if `inference.py` is run first.

---

## 🙌 Acknowledgements

* Dataset: [COVID-19 Radiography Database – Kaggle](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)
* Model: PyTorch ResNet18


