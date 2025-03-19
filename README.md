# COVID-19 Image Classification

This project is a deep learning-based image classification model that classifies chest X-ray images into three categories: **COVID-19, Normal, and Viral Pneumonia**. The model is trained using **PyTorch** and leverages a **Convolutional Neural Network (CNN)** for classification.

## 📂 Dataset
The dataset used is the **COVID-19 Radiography Database**, available on Kaggle:
[Kaggle Dataset](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

### 🗂 Folder Structure After Downloading
```
COVID-19_Radiography_Dataset/
│-- COVID/
│-- Lung_Opacity/
│-- Normal/
│-- Viral Pneumonia/
│-- train/
│   │-- COVID/
│   │-- Normal/
│   │-- Viral Pneumonia/
│-- val/
│   │-- COVID/
│   │-- Normal/
│   │-- Viral Pneumonia/
│-- README.md.txt
```

## 🚀 Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/TaksheelSingh/covid_classifier_kagglehub.git
cd covid_classifier_kagglehub
```
### 2️⃣ Install Dependencies
Make sure you have Python 3.8+ installed, then run:
```bash
pip install -r requirements.txt
```

### 3️⃣ Download and Extract Dataset
Download the dataset using Kaggle API:
```bash
kaggle datasets download -d tawsifurrahman/covid19-radiography-database -p datasets/ --unzip
```

### 4️⃣ Split Dataset into Train/Validation
Since the dataset does not have predefined train/validation folders, run the script to split the dataset:
```bash
python scripts/split_dataset.py
```

## 🎯 Training the Model
To start training the CNN model, run:
```bash
python scripts/train.py
```
This will:
- Load and preprocess the dataset
- Train the CNN model using PyTorch
- Save the trained model

## 🏆 Model Evaluation
To evaluate the trained model:
```bash
python scripts/evaluate.py
```

## 🔮 Predictions
To make predictions on new images:
```bash
python scripts/predict.py --image_path <path_to_image>
```

## 📜 Troubleshooting
- **FileNotFoundError:** If you see an error regarding missing directories, ensure that your dataset path is correctly specified.
- **Import Errors:** If you get an import error, try reinstalling dependencies using `pip install -r requirements.txt`.
- **Incorrect Labels:** Ensure that `train` and `val` folders contain correctly labeled images inside the respective subdirectories.

## 💡 Future Improvements
- Enhance model accuracy with data augmentation
- Implement a web-based interface for real-time predictions
- Train on a larger, more diverse dataset

## 👨‍💻 Author
**Taksheel Singh Rawat**  
[GitHub](https://github.com/TaksheelSingh) | [LinkedIn](https://www.linkedin.com/in/taksheelrawat)