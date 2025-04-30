# Mapping the Lexicon of Healthcare: Connecting ICD codes to Clinical narratives


This repository contains a deep learning-based system for translating **ICD-10 codes** into their corresponding **clinical descriptions**, and vice versa. We implemented and compared traditional **Seq2Seq models** (RNN, LSTM, GRU and their deep/bidirectional forms) with a custom-built **Transformer architecture**, achieving state-of-the-art results in both scenarios.

---

## Project Structure


```bash
Survival_Model/
│── Dataset/                         
│   ├── icd10-codes-and-descriptions/         # This folder contains all the datasets required for this project
│
│── Notebook/            
│   ├── data_cleaning.ipynb                   # This notebook has cleaning code for ICD raw dataset
│   ├── data_collection&Analysis.ipynb        # This notebook has the code for Dataset Extraction and EDA on Cleaned Dataset.
│   ├── Evaluation_Desc_ICD.ipynb             # This notebook has the graphs of both training and tetsing data for all models used in Description to ICD Code conversion
│   ├── Evaluation_ICD_Desc.ipynb             # This notebook has the graphs of both training and tetsing data for all models used in ICD Code to Description conversion
│   ├── Testing_Desc_ICD.ipynb                # This notebook has the code for using the best model to generate ICD Code from Description from every models used.
│   ├── Testing_ICD_Desc.ipynb                # This notebook has the code for using the best model to generate Description from Desction from every models used.
│   ├── Tokenization.ipynb                    # This notebook has the code for using BPE tokenizer and adding padding before using Seq2Seq models.
│
│── Results/                      
│   ├── Desc_to_ICD/
│   │   │   ├── BiGRU/                        # This folder has all the saved metrics of testing data on best model and also has the best hyperparameters for the Bidirectional GRU model
│   │   │   ├── BiLSTM/                       # This folder has all the saved metrics of testing data on best model and also has the best hyperparameters for the Bidirectional LSTM model
│   │   │   ├── BiRNN/                        # This folder has all the saved metrics of testing data on best model and also has the best hyperparameters for the Bidirectional RNN model
│   │   │   ├── DeepGRU/                      # This folder has all the saved metrics of testing data on best model and also has the best hyperparameters for the Deep GRU model
│   │   │   ├── DeepLSTM/                     # This folder has all the saved metrics of testing data on best model and also has the best hyperparameters for the Deep LSTM model
│   │   │   ├── DeepRNN/                      # This folder has all the saved metrics of testing data on best model and also has the best hyperparameters for the Deep RNN Model
│   │   │   ├── GRU_first/                    # This folder has all the saved metrics of testing data on best model and also has the best hyperparameters for the GRU model with using first 5 tokens
│   │   │   ├── GRU_last/                     # This folder has all the saved metrics of testing data on best model and also has the best hyperparameters for the GRU model with using last 5 tokens
│   │   │   ├── GRU_middle/                   # This folder has all the saved metrics of testing data on best model and also has the best hyperparameters for the GRU model with using middle 5 tokens
│   │   │   ├── LSTM_last/                    # This folder has all the saved metrics of testing data on best model and also has the best hyperparameters for the LSTM model with using last 5 tokens
│   │   │   ├── LSTM_first/                   # This folder has all the saved metrics of testing data on best model and also has the best hyperparameters for the LSTM model with using first 5 tokens
│   │   │   ├── LSTM_middle/                  # This folder has all the saved metrics of testing data on best model and also has the best hyperparameters for the LSTM model with using middle 5 tokens
│   │   │   ├── RNN_first/                    # This folder has all the saved metrics of testing data on best model and also has the best hyperparameters for the RNN model with using first 5 tokens
│   │   │   ├── RNN_last/                     # This folder has all the saved metrics of testing data on best model and also has the best hyperparameters for the RNN model with using last 5 tokens
│   │   │   ├── RNN_middle/                   # This folder has all the saved metrics of testing data on best model and also has the best hyperparameters for the RNN model with using middle 5 tokens
  │   │   │   ├── Transfomer/                 # This folder has all the saved metrics of testing data on best model and also has the best hyperparameters for the Transformers model
│   ├── ICD_to_Desc/
│   │   │   ├── BiGRU/                        # This folder has all the saved metrics of testing data on best model and also has the best hyperparameters for the Bidirectional GRU model
│   │   │   ├── BiLSTM/                       # This folder has all the saved metrics of testing data on best model and also has the best hyperparameters for the Bidirectional LSTM model
│   │   │   ├── BiRNN/                        # This folder has all the saved metrics of testing data on best model and also has the best hyperparameters for the Bidirectional RNN model
│   │   │   ├── DeepGRU/                      # This folder has all the saved metrics of testing data on best model and also has the best hyperparameters for the Deep GRU model
│   │   │   ├── DeepLSTM/                     # This folder has all the saved metrics of testing data on best model and also has the best hyperparameters for the Deep LSTM model
│   │   │   ├── DeepRNN/                      # This folder has all the saved metrics of testing data on best model and also has the best hyperparameters for the Deep RNN model
│   │   │   ├── GRU/                          # This folder has all the saved metrics of testing data on best model and also has the best hyperparameters for the GRU model
│   │   │   ├── LSTM/                         # This folder has all the saved metrics of testing data on best model and also has the best hyperparameters for the LSTM model
│   │   │   ├── RNN/                          # This folder has all the saved metrics of testing data on best model and also has the best hyperparameters for the RNN model
│   │   │   ├── Transfomer/                   # This folder has all the saved metrics of testing data on best model and also has the best hyperparameters for the Transformers model
│
│── src/                            
│   ├── main/                      
│   │   ├── Desc_to_ICD/               
│   │   │   ├── BiGRU.py                      # This python file has the code of implementing Bidirectional GRU model along with optuna hyperparameter tuning.
│   │   │   ├── BiLSTM.py                     # This python file has the code of implementing Bidirectional LSTM model along with optuna hyperparameter tuning.
│   │   │   ├── BiRNN.py                      # This python file has the code of implementing Bidirectional RNN model along with optuna hyperparameter tuning.
│   │   │   ├── DeepGRU.py                    # This python file has the code of implementing Deep GRU model along with optuna hyperparameter tuning.
│   │   │   ├── DeepLSTM.py                   # This python file has the code of implementing Deep LSTM model along with optuna hyperparameter tuning.
│   │   │   ├── DeepRNN.py                    # This python file has the code of implementing Deep RNN model along with optuna hyperparameter tuning.
│   │   │   ├── GRU_first.py                  # This python file has the code of implementing GRU model with first 5 tokens as output(ICD Codes tokens) along with optuna hyperparameter tuning.
│   │   │   ├── GRU_last.py                   # This python file has the code of implementing GRU model with last 5 tokens as output(ICD Codes tokens) along with optuna hyperparameter tuning.
│   │   │   ├── GRU_middle.py                 # This python file has the code of implementing GRU model with middle 5 tokens as output(ICD Codes tokens) along with optuna hyperparameter tuning.
│   │   │   ├── LSTM_last.py                  # This python file has the code of implementing LSTM model with last 5 tokens as output(ICD Codes tokens) along with optuna hyperparameter tuning.
│   │   │   ├── LSTM_first.py                 # This python file has the code of implementing LSTM model with first 5 tokens as output(ICD Codes tokens) along with optuna hyperparameter tuning.
│   │   │   ├── LSTM_middle.py                # This python file has the code of implementing LSTM model with middle 5 tokens as output(ICD Codes tokens) along with optuna hyperparameter tuning.
│   │   │   ├── RNN_first.py                  # This python file has the code of implementing RNN model with first 5 tokens as output(ICD Codes tokens) along with optuna hyperparameter tuning.
│   │   │   ├── RNN_last.py                   # This python file has the code of implementing RNN model with last 5 tokens as output(ICD Codes tokens) along with optuna hyperparameter tuning.
│   │   │   ├── RNN_middle.py                 # This python file has the code of implementing RNN model with middle 5 tokens as output(ICD Codes tokens) along with optuna hyperparameter tuning.
│   │   │   ├── Transfomer.py                 # This python file has the code of implementing Transformers model along with optuna hyperparameter tuning.
│   │   ├── ICD_to_Desc/
│   │   │   ├── BiGRU.py                      # This python file has the code of implementing Bidirectional GRU model along with optuna hyperparameter tuning.
│   │   │   ├── BiLSTM.py                     # This python file has the code of implementing Bidirectional LSTM model along with optuna hyperparameter tuning.
│   │   │   ├── BiRNN.py                      # This python file has the code of implementing Bidirectional RNN model along with optuna hyperparameter tuning.
│   │   │   ├── DeepGRU.py                    # This python file has the code of implementing Deep GRU model along with optuna hyperparameter tuning.
│   │   │   ├── DeepLSTM.py                   # This python file has the code of implementing Deep LSTM model along with optuna hyperparameter tuning.
│   │   │   ├── DeepRNN.py                    # This python file has the code of implementing Deep RNN model along with optuna hyperparameter tuning.
│   │   │   ├── GRU.py                        # This python file has the code of implementing GRU model along with optuna hyperparameter tuning.
│   │   │   ├── LSTM.py                       # This python file has the code of implementing LSTM model along with optuna hyperparameter tuning.
│   │   │   ├── RNN.py                        # This python file has the code of implementing RNN model along with optuna hyperparameter tuning.
│   │   │   ├── Transfomer.py                 # This python file has the code of implementing Transformers model along with optuna hyperparameter tuning.
│   │
├── Model_rnn.py                              # This python file has the code of all seq2seq models used along with the train function
├── Transformer_model.py                      # This python file has the code of all modules in transformer architecture used along with the train function
├── utils.py                                  # This python file has the supported dataset function which was used before passing into the model
│── requirements.txt                          # List of required Python libraries
│── README.md                                 # Project documentation

```




---

## Problem Statement

Translate structured ICD codes to plain clinical text and vice versa using neural architectures. Two main tasks were considered:

- **Scenario A**: ICD Code → Clinical Description  
- **Scenario B**: Clinical Description → ICD Code

---

##  Data Preprocessing

- Public ICD dataset (71,704 rows)
- Applied:
  - Lowercasing, special character removal
  - Byte Pair Encoding (BPE) tokenization
  - [SOS], [EOS], [PAD] token injection
  - Sequence padding to 128 tokens
- Final format: `ICD_Code`, `Description`
- Custom `TransformerDataset` used for structured input generation and masking

---

## Models Implemented

| Model Type       | Variants                                |
|------------------|------------------------------------------|
| Seq2Seq Models   | RNN, LSTM, GRU, DeepRNN, DeepLSTM, DeepGRU |
| Bidirectional    | Bi-RNN, Bi-LSTM, Bi-GRU                  |
| Transformer      | Custom from scratch                      |

---

## Training Strategy

- Optimizer: **Adam** with weight decay
- Loss: **Cross Entropy** (ignoring `[PAD]` tokens)
- **Gradient Clipping** used to stabilize training
- **Validation loss monitored every 5 epochs**
- **Optuna** for hyperparameter tuning:
  - `lr`, `hidden_size`, `d_model`, `num_heads`, `layers`, etc.
- Best checkpoints saved and restored based on validation performance

---

## Evaluation Metrics

- **Accuracy**
- **Precision**, **Recall**, **F1-Score**
- **BLEU Score**
- **Word Error Rate (WER)**
- **Character Error Rate (CER)**

---

## Results — Scenario A (ICD → Description)

| Model       | Accuracy | F1-Score |
|-------------|----------|----------|
| Transformer | 99.82%   | 0.87     |
| Bi-GRU      | 91%      | 0.50     |
| DeepGRU     | 82%      | 0.41     |
| RNN         | 65%      | 0.36     |

BLEU: `0.97` | WER: `0.012` | CER: `0.012`

---

## Results — Scenario B (Description → ICD)

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

## Optuna Hyperparameter Tuning

- Used to optimize learning rate, model depth, attention heads, dropout, etc.
- Trials = 5–10 per model
- Objective: minimize validation loss
- Best model retrained and evaluated on test set

---



## Conclusion

This project demonstrates a custom Transformer model's ability to outperform traditional architectures in mapping ICD codes to and from clinical descriptions. The combination of structured preprocessing, attention mechanisms, and detailed evaluation provides a robust framework for medical code-to-text applications.

---


