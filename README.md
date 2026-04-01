# 🎤 Speech Emotion Recognition using Machine Learning and Deep Learning

This project implements a Speech Emotion Recognition (SER) system using spectral feature extraction and multiple machine learning models.

## 📊 Dataset

The dataset consists of audio recordings representing four emotions:

* Angry
* Calm
* Happy
* Sad

## 🧠 Approach

### Feature Extraction

Mel-spectrograms were generated from audio signals and divided into frequency bands. The following features were extracted:

* Spectral Centroid (SC)
* Spectral Bandwidth (SBW)
* Spectral Band Energy (SBE)

### Models Used

* Support Vector Machine (SVM)
* 1D Convolutional Neural Network (1D CNN)
* 2D Convolutional Neural Network (2D CNN)

## 📈 Results

| Model            | Performance                       |
| ---------------- | --------------------------------- |
| SVM (SC feature) | Best (~55% accuracy)              |
| 1D CNN           | Moderate (~42% accuracy)          |
| 2D CNN           | Poor performance (model collapse) |

## 🔍 Key Insights

* Spectral Centroid was the most effective feature
* Traditional ML (SVM) outperformed deep learning on small datasets
* 2D CNN failed due to insufficient data and lack of augmentation

## ⚠️ Challenges

* Limited dataset size
* Overfitting in deep learning models
* Need for data augmentation techniques

## 🚀 How to Run

```bash
pip install -r requirements.txt
python ser_model.py
```

## 📎 Files

* `ser_model.py` – implementation
* `report.pdf` – detailed analysis

## 👤 Author

Rithik Vishal Nair
Master of Applied Artificial Intelligence, Deakin University
