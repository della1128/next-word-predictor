# 🧠 Next Word Predictor using LSTM in PyTorch

A deep learning model built with PyTorch to predict the next word in a given sentence using Long Short-Term Memory (LSTM) neural networks. This project demonstrates the fundamentals of sequence modeling in NLP using a simple and interpretable architecture.

---

## 🚀 Demo Preview

Enter a starting phrase like:
the sun is


And the model may predict:
shining



---

## 📌 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [How to Run](#how-to-run)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)

---

## 📖 Overview

Language modeling is a foundational task in natural language processing. This project implements an LSTM-based model using PyTorch to predict the next word in a sequence. It walks through preprocessing, tokenization, embedding, training, and evaluation.

---

## 🗃 Dataset

The model is trained on a small custom text. You can replace this dataset with any plain text corpus for experimentation.

---

## 🧠 Model Architecture

- **Embedding Layer**: Converts word indices to dense vectors
- **LSTM Layer**: Captures temporal dependencies
- **Linear Layer**: Predicts next-word logits
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam

---

## ⚙️ How to Run

### 🟢 Option 1: Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/della1128/next-word-lstm/blob/main/pytorch_lstm_next_word_predictor.ipynb)

Just open the notebook and run the cells from top to bottom.

---

### 🔵 Option 2: Local Setup

1. Clone the repo:

git clone https://github.com/della1128/next-word-lstm.git
cd next-word-lstm
Install requirements:


pip install -r requirements.txt
Run the notebook:


jupyter notebook pytorch_lstm_next_word_predictor.ipynb
✅ Results
Training Accuracy: ~95% (on small dataset)

Next Word Examples:

Input: the world is → Prediction: beautiful

Input: she walked into → Prediction: the

Note: Results may vary based on the dataset used.

🧰 Technologies Used
Python

PyTorch

NumPy

Jupyter Notebook / Google Colab

💡 Future Improvements
Use pretrained embeddings (GloVe or FastText)

Add beam search or top-k sampling

Train on larger corpora (e.g., Wikipedia or news data)

Export model using TorchScript or ONNX for deployment

👩‍💻 Author
Della Cherian
AI/ML Enthusiast | ECE Graduate | Aspiring Data Scientist


📄 License
This project is open source under the MIT License.
