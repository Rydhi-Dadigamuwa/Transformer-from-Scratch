# 🧠 Build Transformer from Scratch

In this project, I built a **Transformer network** — an **encoder-only architecture** with **two encoder blocks**, each containing **two multi-head attention heads**, specifically designed for **sentiment analysis**.  
This implementation is built **entirely from scratch**, without using any prebuilt Transformer layers, to provide a clear understanding of the architecture’s inner mechanics.

---

## 📘 Overview

Transformers have redefined modern NLP by relying purely on **attention mechanisms** instead of recurrence or convolution.  
This project demonstrates how to **build an encoder-only Transformer** for **sentiment classification** from the ground up using TensorFlow and NumPy.

The notebook explains and implements:

1. Token and Positional Embeddings  
2. Scaled Dot-Product Attention  
3. Multi-Head Attention (2 heads per encoder)  
4. Feed-Forward Networks  
5. Layer Normalization and Residual Connections  
6. Encoder Stack (2 layers)  
7. Classification Head for Sentiment Prediction  
8. Training Loop and Evaluation

---

## 🏗️ Project Structure

build-transformer-from-scratch

├── Build_Transformer_from_Scratch.ipynb             # Main Jupyter notebook implementing the Transformer model  
├── README.md                                        # Project documentation  
├── IMDB_Dataset.csv                                 # Kaggle dataset (https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  
├── Transformer_Sentiment_Model.h5                   # Saved encoder-only Transformer model (H5 format)  
├── Transformer_Sentiment_Model.keras                # Saved encoder-only Transformer model (Keras format)  
└── model_architecture.png                           # Model architecture visualization  



---

## ⚙️ Features

- ✅ **Encoder-only Transformer** designed for sentiment analysis  
- 🧩 **Two encoder blocks**, each with **two multi-head attention heads**  
- 🧠 Built **completely from scratch** using TensorFlow/Keras low-level APIs  
- 📊 Includes training and evaluation for classification  
- 💬 Easy to extend to other NLP tasks  

---

## 🧩 Key Components

| Component | Description |
|------------|-------------|
| **Embedding Layer** | Converts tokens into dense vector representations. |
| **Positional Encoding** | Adds position information to embeddings using sine and cosine functions. |
| **Scaled Dot-Product Attention** | Calculates the attention scores between queries, keys, and values. |
| **Multi-Head Attention** | Allows the model to focus on different parts of the sequence simultaneously. |
| **Feed-Forward Network** | Applies non-linear transformations to enrich learned features. |
| **Encoder Block** | Combines attention and feed-forward sublayers with residual connections. |
| **Classifier Head** | Outputs sentiment class probabilities. |

---
