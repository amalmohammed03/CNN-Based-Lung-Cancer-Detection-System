# 🫁 CNN-Based Lung Cancer Detection System

## 📌 Overview

This project is a deep learning-based web application that detects lung cancer from CT scan images using a Convolutional Neural Network (CNN). The system classifies CT images into four categories and provides visual explanations using Grad-CAM.

---

## 🎯 Features

* Multi-class classification:

  * Adenocarcinoma
  * Large Cell Carcinoma
  * Squamous Cell Carcinoma
  * Normal
* Confidence score for predictions
* Grad-CAM heatmap visualization
* Flask-based web interface

---

## 🧠 Model Details

* CNN with 3 convolution layers
* Input size: 224 × 224
* Activation: ReLU
* Output: Softmax (4 classes)

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Flask

---

## 📂 Project Structure

```bash
lung-cancer-detection/
│
├── app.py
├── model.py
├── predict.py
├── preprocess.py
├── heatmap.py
├── test_heatmap.py
│
├── templates/
├── static/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ▶️ How to Run

### 1. Clone Repository

```bash
git clone https://github.com/your-username/lung-cancer-detection.git
cd lung-cancer-detection
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Run App

```bash
python app.py
```

### 4. Open in Browser

```
http://127.0.0.1:5000/
```

---

## 📊 Output

* Predicted class
* Confidence score
* Heatmap visualization


## ⚠️ Disclaimer

This project is for educational purposes only and not for medical use.

---

## 👨‍💻 Author
* K R Amal Mohammed

---
