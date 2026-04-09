# 🫁 Lung Cancer Detection System

## 📌 Overview

This project implements a deep learning-based system for detecting lung cancer from CT scan images using a Convolutional Neural Network (CNN). The model classifies images into multiple lung conditions and provides a confidence score for each prediction. To improve interpretability, Grad-CAM is used to generate heatmaps that highlight the regions influencing the model’s decision.

---

## 🎯 Objective

To develop an efficient and reliable CNN-based model for lung cancer classification and enhance prediction transparency through visual explanations.

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Flask

---

## 📂 Project Files

```bash
app.py              # Flask web application
model.py            # CNN model training
predict.py          # Prediction and inference
preprocess.py       # Image preprocessing
heatmap.py          # Grad-CAM heatmap generation
test_heatmap.py     # Heatmap validation/testing
```

---

## 📊 Output

* Predicted lung condition
* Confidence score
* Grad-CAM heatmap visualization

---

## ⚠️ Note

This project is developed for educational purposes and is not intended for clinical or medical use.

---

## 👨‍💻 Author

Amal Mohammed
