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

## 📂 Dataset Used

- **BRaTS BRISC2025 Brain Tumor Dataset**
- [📁 Download Dataset](https://www.kaggle.com/datasets/briscdataset/brisc2025/data)

---

## 🧠 Pretrained Model

- Download the trained model file (`hybrid_model_weights.pth`) from Google Drive:  
[🔗 Download Model (Google Drive)]([https://drive.google.com/file/d/YOUR_MODEL_ID/view?usp=sharing](https://drive.google.com/drive/u/0/folders/1rj-c3FG69cGCK0jYoEidhaMALz4cAVfR?q=sharedwith:public%20parent:1rj-c3FG69cGCK0jYoEidhaMALz4cAVfR%20sharedwith:CgJtZSgH)


---

## 🧠 Model Architecture

A custom Convolutional Neural Network (CNN) was trained with:

- 3 convolutional blocks
- Dropout for regularization
- Fully connected layers
- Output: 4-class softmax

The trained model achieved 98.20% testing accuracy and saved as: `hybrid_model_weights.pth`.

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

