# 🩺 Disease Classification from Chest X-Rays  
**(Python | TensorFlow/Keras | OpenCV | Scikit-learn)**  

## 📘 Overview  
This project implements a deep learning model for **automated disease detection** from chest X-ray images.  
Using **transfer learning (MobileNetV2)**, the model classifies X-rays as **Normal** or **Pneumonia** with high accuracy and reliability.  
Additionally, **Integrated Gradients** are applied to visualize and interpret the model’s decision-making process, ensuring explainability in a medical context.

---

## 🎯 Objectives  
- Build a deep learning model to detect pneumonia from chest X-ray images.  
- Apply **transfer learning** and **data augmentation** to improve generalization.  
- Evaluate model performance using **accuracy, precision, recall, F1-score, and ROC-AUC**.  
- Use **explainable AI** techniques to highlight lung regions influencing predictions.  

---

## 🧩 Dataset  
- **Source:** [Kaggle – Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
- **Classes:**  
  - `NORMAL`  
  - `PNEUMONIA`  
- **Size:** 5,863 X-ray images divided into training, validation, and test sets.  
- Images were resized to **224×224 px** and normalized before training.

---

## ⚙️ Model Architecture  
- **Base Model:** `MobileNetV2` pretrained on ImageNet  
- **Custom Layers:**  
  - Global Average Pooling  
  - Dense (128, ReLU)  
  - Dropout (0.3)  
  - Dense (1, Sigmoid)  
- **Loss Function:** Binary Crossentropy  
- **Optimizer:** Adam (lr = 1e-4)  
- **Callbacks:** EarlyStopping, ModelCheckpoint  

---

## 📈 Results  
| Metric | Score |
|---------|-------|
| **Accuracy** | 82.9 % |
| **Precision** | 0.80 |
| **Recall** | 0.97 |
| **F1-Score** | 0.88 |
| **ROC-AUC** | 0.95 |

**Confusion Matrix:**  
[[137 97]
[ 10 380]]


✅ High recall (0.97) indicates strong sensitivity — the model rarely misses pneumonia cases.  

## 💻 Tech Stack  
- **Languages:** Python  
- **Frameworks:** TensorFlow / Keras  
- **Libraries:** NumPy, Pandas, Scikit-learn, Matplotlib, OpenCV  
- **Environment:** Google Colab GPU  
