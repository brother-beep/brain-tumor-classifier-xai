# 🧠 Brain Tumor Classifier with XAI (Grad-CAM)

This repository hosts a deep learning-based **brain tumor classification** app trained on the **BRISC2025** dataset. The app predicts the type of brain tumor and explains the prediction using **Grad-CAM**, a powerful Explainable AI (XAI) technique.

---

## 🚀 Features

- 🧠 Classifies MRI images into:
  - Glioma
  - Meningioma
  - Pituitary tumor
  - No tumor
- 🔍 Explains model predictions using Grad-CAM heatmaps.
- 🖼️ Clean and interactive **Streamlit** web interface.
- ✅ Lightweight and easy to deploy locally or on the web.

---

## 📂 Dataset

- **Name**: BRISC2025 Brain Tumor Dataset
- **Content**: Labeled MRI images of 4 classes: Glioma, Meningioma, Pituitary, and No Tumor.

---

## 🧠 Model Architecture

A custom Convolutional Neural Network (CNN) was trained with:

- 3 convolutional blocks
- Dropout for regularization
- Fully connected layers
- Output: 4-class softmax

The trained model is saved as: `model/hybrid_model_weights.pth`.

---

## 💡 Explainability with Grad-CAM

Grad-CAM highlights the image regions responsible for the prediction. It enhances trust and provides interpretability for black-box models.

---

## 🧪 How to Run the App

### ✅ 1. Clone the repository

```bash
git clone https://github.com/brother-beep/brain-tumor-classifier-xai.git
cd brain-tumor-classifier-xai

### ✅ 2. Install dependencies

```bash
pip install -r requirements.txt

### ✅ 2. Run streamlit app

```bash
streamlit run app.py

