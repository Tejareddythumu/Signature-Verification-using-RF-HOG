# ✍️ Signature Verification using HOG + Random Forest

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Model-orange?style=flat-square&logo=scikit-learn)
![Flask](https://img.shields.io/badge/Flask-WebApp-lightblue?style=flat-square&logo=flask)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

> 🧠 An intelligent offline handwritten signature verification system using **HOG feature extraction** and a **Random Forest classifier**. This system can accurately distinguish between **genuine** and **forged** signatures.

---

## 📌 Table of Contents
- [🔍 Overview](#-overview)
- [📁 Folder Structure](#-folder-structure)
- [⚙️ Technologies Used](#️-technologies-used)
- [🚀 How to Run the Project](#-how-to-run-the-project)
- [📊 Results & Evaluation](#-results--evaluation)
- [📂 Dataset Info](#-dataset-info)
- [🙋‍♂️ Author](#-author)
- [📄 License](#-license)

---

## 🔍 Overview

This project was built as part of my academic final year work. It focuses on detecting signature forgery using machine learning techniques. We extract features from signature images using **Histogram of Oriented Gradients (HOG)** and classify them using a **Random Forest** model. A simple **Flask-based web app** is also provided to test the model in real-time.

---

## 📁 Folder Structure
```
📁 signature-verification
├── Code/
│   ├── app.py                      # Flask web app
│   ├── rf_cedar.py                 # CEDAR dataset experiment
│   ├── rf_bengali.py               # BHSig-B dataset experiment
│   ├── rf_hindi.py                 # BHSig-H dataset experiment
│   ├── rf_mysign.py                # My personal signature demo
│   ├── train_model.py              # Training the Random Forest model
│   └── utils.py                    # HOG feature extraction & utilities
├── Dataset/
│   └── link.txt                   # Download links for datasets
├── Model/
│   ├── signature_verification_model_test.pkl
│   └── scaler_test.pkl
├── results/
│   ├── results_bengali_1.txt       
│   ├── results_cedar.txt
│   ├── results_hindi.txt               
│   ├── results_mine.txt                 
├── Templates/
│   └── index.html                 # HTML frontend for Flask app
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Technologies Used

- Python 3.10
- Scikit-learn
- OpenCV
- Flask
- NumPy & Pandas
- PIL (Pillow)
- Matplotlib

---

## 🚀 How to Run the Project

### 🖥️ 1. Clone the Repository
```bash
git clone https://github.com/Tejareddythumu/Signature-Verification-using-RF-HOG.git
cd Signature-Verification-using-RF-HOG
```
### 🧱 2. Install Dependencies
```
pip install -r requirements.txt
```
### 🌐 3. Run the Web App
```
cd Code
python app.py
```
Open your browser and go to:
http://127.0.0.1:5000

Upload a test signature image and check the result!

## 📊 Results & Evaluation

- 📁 [`results/`](./results): Contains `.txt` files showing detailed classification results, evaluation metrics, and summaries.
- 📁 [`confusion matrix images/`](./confusion%20matrix%20images): Includes confusion matrices visualized as images for both training and testing phases.
## 📂 Dataset Info
This project uses offline signature datasets:

- **BHSig-B** (Bengali)  
- **BHSig-H** (Hindi)  
- **CEDAR**

🔗 [Click here to access dataset links](./Dataset/link.txt)
