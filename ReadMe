# 🩻 Chest X-Ray Classification using Deep Learning

A deep learning-based web application that classifies chest X-ray images into two categories:

- **NORMAL**
- **PNEUMONIA**

The project uses **EfficientNet-B0 with PyTorch** for image classification and provides a **FastAPI backend** and **Streamlit frontend** for making predictions through a web interface.

> ⚠️ **Disclaimer:** This project is developed for educational and research purposes. It is not intended to replace professional medical diagnosis.

---

## 🚀 Features

- 🩻 Upload chest X-ray images
- 🤖 Deep learning-based classification
- 🧠 EfficientNet-B0 model
- ⚡ PyTorch inference
- 🔌 FastAPI REST API
- 🌐 Streamlit web interface
- 📊 Prediction confidence
- 📈 Probability for each class
- 🖥️ GPU support with CUDA when available

---

## 🏗️ Project Architecture

```text
                 ┌──────────────────────┐
                 │      Streamlit       │
                 │      Frontend        │
                 │                      │
                 │   Upload X-Ray       │
                 │        ↓             │
                 │     Predict          │
                 └──────────┬───────────┘
                            │
                            │ HTTP POST
                            ▼
                 ┌──────────────────────┐
                 │       FastAPI        │
                 │       Backend        │
                 │                      │
                 │   Receive Image      │
                 │        ↓             │
                 │   Preprocessing      │
                 │        ↓             │
                 │   EfficientNet-B0    │
                 │        ↓             │
                 │    Prediction        │
                 └──────────┬───────────┘
                            │
                            ▼
                  NORMAL / PNEUMONIA
                     + Confidence