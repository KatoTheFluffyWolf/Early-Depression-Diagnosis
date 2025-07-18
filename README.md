# 🧠 Early Depression Diagnosis using Deep Learning

A web-based tool powered by a Multi-Layer Perceptron (MLP) model that predicts the risk of depression in students using lifestyle, academic, and psychological inputs. This system utilizes embedded categorical features and numerical normalization, all integrated into an intuitive interface.

---

## 🚀 Features

- ⚙️ MLP model with embedded categorical features
- 📊 Uses academic, lifestyle, and mental health data
- 🌐 Web interface with visualized results and recommendations
- 🔌 REST API via FastAPI for model predictions
- 📈 Dashboard with confidence score and risk visualization
- 🏫 Designed for research at FPT University of Greenwich (Can Tho campus)

---

## 🧠 Model Overview

- Inputs: 
  - Numerical: Age, CGPA, Academic Pressure, Study Satisfaction, etc.
  - Categorical: Sleep Duration, Dietary Habits, Degree, etc.
- Architecture:
  - Embedding layers for categorical inputs
  - Dense layers with ReLU activations
  - Sigmoid output for binary classification  
- Optimizer: Adam  
- Loss: Binary Crossentropy  
- EarlyStopping for regularization

![Model Architecture](model_architecture.png)

---

## 📁 Project Structure

    .
    ├── index.html                     # Frontend UI
    ├── model_api.py                  # FastAPI backend for model serving
    ├── DepressionDetection_code.py   # Model training script
    ├── Depression_detection.h5      # Saved trained model
    ├── Depression_detection.h5      # The same model, but saved as a .keras file
    ├── model_architecture.png        # Neural network diagram
    ├── requirements.txt              # Backend dependencies
    ├── student_depression_dataset.csv # Source dataset




---

## 🧪 Dataset

- Source: Kaggle — [Student Depression Dataset](https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset)
- Preprocessing:
  - Missing value handling
  - Standardization for numerical features
  - Label encoding for categorical features

---

## 🖼 Web Application Interface Overview

![Screenshot](model_architecture.png)

---

## 🔍 How It Works

1. Fill out the form (e.g., GPA, sleep, stress level)
2. Click **Analyze Risk Factors**
3. View your depression risk score and recommendations

---

## 🧰 Tech Stack

- Frontend: HTML, CSS, JS
- Backend: FastAPI + TensorFlow
- Model: MLP with categorical embeddings
- Hosting: Render (or your platform)

---

## 👥 Team

- Duy Anh Nguyen. GitHub: https://github.com/KatoTheFluffyWolf  
- Trung Hau Nguyen. GitHub: https://github.com/Guerfu  
- Hoang Khoa Trinh. GitHub: https://github.com/trinhkhoa  
- Supervisor 1: Kim Khanh Nguyen (FPT University of Greenwich)  
- Supervisor 2: Nguyen Thanh Hai (Can Tho University)   

---

## 📄 License

This project is intended for educational and research purposes only.
