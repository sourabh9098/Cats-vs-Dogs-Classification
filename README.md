# AI-Powered Cats vs Dogs Classification using CNN

<p align="center">

<img src="assets/banner.png" width="100%"/>

</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-3-red?style=for-the-badge&logo=keras)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit)
![CNN](https://img.shields.io/badge/DeepLearning-CNN-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</p>

---

# Project Overview

This project is an end-to-end **Deep Learning Image Classification System** that identifies whether an uploaded image belongs to a **Cat** or a **Dog** using a Convolutional Neural Network (CNN).

The model has been trained using TensorFlow/Keras and deployed with an interactive Streamlit web application that allows users to upload images and receive instant predictions with confidence scores.

This project demonstrates the complete deep learning workflow including:

- Data preprocessing
- Image normalization
- CNN model development
- Model training
- Performance evaluation
- Model deployment
- Interactive web application

---

# Live Demo

**Live Application**

> Add your deployed Streamlit/Render URL here

Example

```
https://your-app.onrender.com
```

---

# Project Preview

## Home Page

<img src="assets/home.png" width="100%">

---

## Prediction Result

<img src="assets/prediction.png" width="100%">

---

# Features

- Upload any Cat or Dog image
- Real-time prediction
- Confidence score
- Image preview
- Fast inference
- Responsive Streamlit UI
- CNN-based Deep Learning model
- Easy to use interface

---

# Dataset

Dataset Used

**Microsoft Cats vs Dogs Dataset**

Classes

- Cat
- Dog

Images are resized to

```
256 × 256
```

---

# Deep Learning Pipeline

```
Dataset

      │

      ▼

Image Preprocessing

      │

      ▼

Normalization

      │

      ▼

CNN Model

      │

      ▼

Training

      │

      ▼

Evaluation

      │

      ▼

Saved Model (.keras)

      │

      ▼

Streamlit Deployment

```

---

# CNN Architecture

The model consists of multiple Convolutional and Pooling layers followed by Dense layers.

Example Architecture

```
Input Image

↓

Conv2D

↓

MaxPooling

↓

Conv2D

↓

MaxPooling

↓

Conv2D

↓

MaxPooling

↓

Flatten

↓

Dense

↓

Dropout

↓

Dense

↓

Output Layer

```

---

# Technologies Used

| Technology | Purpose |
|------------|----------|
| Python | Programming Language |
| TensorFlow | Deep Learning |
| Keras | CNN Model |
| NumPy | Numerical Computing |
| Pillow | Image Processing |
| Streamlit | Web Application |
| Joblib | Saving Files |

---

# Folder Structure

```
Cats-vs-Dogs-CNN/

│

├── app.py

├── cnn_model.keras

├── class_names.pkl

├── requirements.txt

├── assets/

│     ├── banner.png

│     ├── home.png

│     └── prediction.png

│

└── README.md

```

---

# Installation

Clone Repository

```bash
git clone https://github.com/yourusername/Cats-vs-Dogs-CNN.git
```

Move into project

```bash
cd Cats-vs-Dogs-CNN
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run Streamlit

```bash
streamlit run app.py
```

---

# Prediction Workflow

```
Upload Image

↓

Resize Image

↓

Normalize Pixels

↓

CNN Model Prediction

↓

Softmax / Sigmoid Output

↓

Display Class

↓

Display Confidence

```

---

# Future Improvements

- MobileNetV2 Transfer Learning
- EfficientNet
- Multi-Class Animal Classification
- Grad-CAM Visualization
- Model Explainability
- Batch Image Prediction
- Webcam Prediction
- Docker Deployment
- FastAPI Backend

---

# Repository

```
Model

+

Deployment

+

Deep Learning

+

CNN

+

Interactive UI

=

Production Ready Project
```

---

# Author

**Sourabh Vishwakarma**

Aspiring AI Engineer | Machine Learning Engineer | Deep Learning Enthusiast

GitHub

```
https://github.com/sourabh9098
```

LinkedIn

```
https://linkedin.com/in/sourabh9098
```

---

## If you found this project useful

Please consider giving this repository a ⭐.

It motivates me to build more Machine Learning and Deep Learning projects.
