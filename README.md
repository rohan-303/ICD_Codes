# 🧠 ICD Code to Clinical Description Mapping using Transformer and Seq2Seq Models

This repository contains a deep learning-based system for translating **ICD-10 codes** into their corresponding **clinical descriptions**, and vice versa. We implemented and compared traditional **Seq2Seq models** (RNN, LSTM, GRU and their deep/bidirectional forms) with a custom-built **Transformer architecture**, achieving state-of-the-art results in both scenarios.

---

## 📂 Project Structure

├── data/ │ ├── Tokens.csv │ └── icd10-codes.csv ├── models/ │ ├── Model_rnn.py │ ├── transformer.py ├── training/ │ ├── transformer_train.py │ ├── rnn_first.py │ ├── rnn_middle.py │ ├── rnn_last.py ├── utils/ │ └── helper.py └── results/ ├── checkpoints/ └── metrics/



---

## 🔍 Problem Statement

Translate structured ICD codes to plain clinical text and vice versa using neural architectures. Two main tasks were considered:

- **Scenario A**: ICD Code → Clinical Description  
- **Scenario B**: Clinical Description → ICD Code

---

## 🧹 Data Preprocessing

- Public ICD dataset (71,704 rows)
- Applied:
  - Lowercasing, special character removal
  - Byte Pair Encoding (BPE) tokenization
  - [SOS], [EOS], [PAD] token injection
  - Sequence padding to 128 tokens
- Final format: `ICD_Code`, `Description`
- Custom `TransformerDataset` used for structured input generation and masking

---

## ⚙️ Models Implemented

| Model Type       | Variants                                |
|------------------|------------------------------------------|
| Seq2Seq Models   | RNN, LSTM, GRU, DeepRNN, DeepLSTM, DeepGRU |
| Bidirectional    | Bi-RNN, Bi-LSTM, Bi-GRU                  |
| Transformer      | Custom from scratch                      |

---

## 📈 Training Strategy

- Optimizer: **Adam** with weight decay
- Loss: **Cross Entropy** (ignoring `[PAD]` tokens)
- **Gradient Clipping** used to stabilize training
- **Validation loss monitored every 5 epochs**
- **Optuna** for hyperparameter tuning:
  - `lr`, `hidden_size`, `d_model`, `num_heads`, `layers`, etc.
- Best checkpoints saved and restored based on validation performance

---

## 📊 Evaluation Metrics

- **Accuracy**
- **Precision**, **Recall**, **F1-Score**
- **BLEU Score**
- **Word Error Rate (WER)**
- **Character Error Rate (CER)**

---

## ✅ Results — Scenario A (ICD → Description)

| Model       | Accuracy | F1-Score |
|-------------|----------|----------|
| Transformer | 99.82%   | 0.87     |
| Bi-GRU      | 91%      | 0.50     |
| DeepGRU     | 82%      | 0.41     |
| RNN         | 65%      | 0.36     |

BLEU: `0.97` | WER: `0.012` | CER: `0.012`

---

## 🔁 Results — Scenario B (Description → ICD)

- Description padded to 60 tokens
- ICD code output fixed at 5 tokens
- RNN/LSTM/GRU: tested on **first**, **middle**, and **last** 5 tokens
- Deep/Bi-directional models: used best-performing segment
- Transformer: trained on full sequence

| Model        | Accuracy | F1-Score |
|--------------|----------|----------|
| Transformer  | 98%      | 0.78     |
| Bi-GRU       | 98%      | 0.97     |
| LSTM_last    | 98%      | 0.86     |
| GRU_last     | 95%      | 0.69     |

---

## 🧪 Optuna Hyperparameter Tuning

- Used to optimize learning rate, model depth, attention heads, dropout, etc.
- Trials = 5–10 per model
- Objective: minimize validation loss
- Best model retrained and evaluated on test set

---

## 🖥️ Hardware Used

- 3× NVIDIA RTX 3090 GPUs
- CUDA 12.4
- Transformer training took ~1 week due to multiple Optuna trials

---

## 🚧 Limitations & Future Work

- Long ICD codes and rare cases can still be challenging
- Future improvements:
  - Clinical domain-specific pretrained Transformers
  - Model interpretability (e.g., attention visualization)
  - Ensembling and real-world deployment

---

## ✅ Conclusion

This project demonstrates a custom Transformer model's ability to outperform traditional architectures in mapping ICD codes to and from clinical descriptions. The combination of structured preprocessing, attention mechanisms, and detailed evaluation provides a robust framework for medical code-to-text applications.

---

## 📎 Citation

