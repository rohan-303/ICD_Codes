# ICD Code to Clinical Description Mapping using Transformer and Seq2Seq Models

This repository contains a deep learning-based system for translating **ICD-10 codes** into their corresponding **clinical descriptions**, and vice versa. We implemented and compared traditional **Seq2Seq models** (RNN, LSTM, GRU and their deep/bidirectional forms) with a custom-built **Transformer architecture**, achieving state-of-the-art results in both scenarios.

---

## Project Structure


```bash
Survival_Model/
│── data/                          # Contains datasets (before and after preprocessing)
│   ├── raw/                       # Raw datasets before preprocessing
│   ├── processed/                  # Processed datasets after preprocessing
│
│── data_preprocessing/             # Scripts and notebooks for dataset preprocessing
│   ├── Metabric.ipynb                   # Preprocessing for Metabric dataset
│   ├── Support.ipynb                    # Preprocessing for Support dataset
│   ├── PBC2.ipynb                       # Preprocessing for PBC2 dataset
│   ├── Syn_TimeVar.ipynb                # This python notebook has the code for cox data generation along with including and excluding units.
│
│── examples/                       # Example usage of the Survival Model
│   ├── using_baseline_obs/         # Baseline observation examples (Metabric, PBC2, Support)
│   ├── using_cox_synthetic_data/   # Cox synthetic data generation examples
│   │   │   ├── EncoderCox_exp2.ipynb.ipynb/            # This python notebook has the insights for EnocderCox(After including units) and Deep Recurrent Survival Machine(After Including Units)
│   │   │   ├── EncoderCox_experiments.ipynb/            # This python notebook is to visualize the c-index from the output of EncoderCox_experiments.py
│   │   │   ├── Syn_Linear_cox.ipynb/             # This pyton Notebook has the comparision across all the model for Linear Data (Linear Time-invariant) 
│   │   │   ├── Syn_Non-Linear_Cox.ipynb/         # This pyton Notebook has the comparision across all the model for Linear Data (Non-Linear Time-invariant) 
│   │   │   ├── Syn_TimeVar.ipynb/            # This python Notebook has the visualization of all the time-varying models like Cox, EncoderCox, RNN, LSTM, GRU.
│   ├── using_LOCF/                 # LOCF (Last Observation Carried Forward) method examples
│
│── results/                        # Results from different experiments
│   ├── baseline_obs/               # Results from baseline observation models (Metabric, PBC2, Support)
│   ├── cox_synthetic_data/         # Results from Cox synthetic models (Linear, Non-Linear, Time-Varying)
│   │   │   ├── Linear/             # Cox synthetic Linear data models (Linear, Non-Linear, Time-Varying)
│   │   │   ├── Non-Linear/         # Cox synthetic Non-Linear data models (Linear, Non-Linear, Time-Varying)
│   │   │   ├── TimeVar/            # Cox synthetic data models (Time-Varying)
│   │   │   │   ├── Cox_Model/      # This file contains a results.txt, all_simulations_results.pth(After using torch.load, contains the Dictionary of all the parameters used in the simulation), test_cindex.pth(Has the list of 0.25,0.50,0.75 test c-index)
│   │   │   │   ├── DRSM/             # This file contains a results.txt, all_simulations_results_{number}.pth(After using torch.load, contains the Dictionary of all the parameters used in the simulation), test_cindex_{number}.pth(Has the list of 0.25,0.50,0.75 test c-index), val_cindex_{number}.pth (The number indicates the placeholder of interval_list which was used in DRSM.py)
│   │   │   │   ├── EncoderCox_exp2/                # This file contains a results.txt, all_simulations_results_{number}.pth(After using torch.load, contains the Dictionary of all the parameters used in the simulation), test_cindex_{number}.pth(Has the list of 0.25,0.50,0.75 test c-index), val_cindex_{number}.pth (The number indicates the placeholder of interval_list which was used in DRSM.py)
│   │   │   │   ├── EncoderCox_experiments/            # This file has the individual test and val c-index's along with the results to the simulation
│   │   │   │   ├── EncoderCox/            # This file contains a results.txt, all_simulations_results.pth(After using torch.load, contains the Dictionary of all the parameters used in the simulation), test_cindex.pth(after using torch.load it contains the list of 0.25,0.50,0.75 test c-index), val_cindex.pth(Ater using torch.load it contains the list of 0.25,0.50,0.75 test c-index)
│   │   │   │   ├── RDSM/            # EncoderCox model For TimeVarying (Has previous cox data generation, Excluding Units ) (Here units indicates Different Interval Lengths) 
│   │   │   │   │   ├── GRU/              # This file contains a results.txt, all_simulations_results.pth(After using torch.load, contains the Dictionary of all the parameters used in the simulation), test_cindex.pth(after using torch.load it contains the list of 0.25,0.50,0.75 test c-index)
│   │   │   │   │   ├── LSTM/             # This file contains a results.txt, all_simulations_results.pth(After using torch.load, contains the Dictionary of all the parameters used in the simulation), test_cindex.pth(after using torch.load it contains the list of 0.25,0.50,0.75 test c-index)
│   │   │   │   │   ├── RNN/              # This file contains a results.txt, all_simulations_results.pth(After using torch.load, contains the Dictionary of all the parameters used in the simulation), test_cindex.pth(after using torch.load it contains the list of 0.25,0.50,0.75 test c-index)
│   │   │   │   ├── Transformer/      # This file contains a results.txt, all_simulations_results.pth(After using torch.load, contains the Dictionary of all the parameters used in the simulation), test_cindex.pth(after using torch.load it contains the list of 0.25,0.50,0.75 test c-index)
│   ├── LOCF/                       # Results from LOCF (Last Observation Carried Forward) method
│
│── src/                            # Source code for the project
│   ├── main/                       # Main scripts
│   │   ├── baseline/               # Baseline data processing and models
│   │   ├── cox_synthetic_data/     # Cox synthetic data generation models 
│   │   │   ├── Linear/             # Cox synthetic Linear data models
│   │   │   ├── Non-Linear/         # Cox synthetic Non-Linear data models
│   │   │   ├── TimeVar/            # Cox synthetic data models (Time-Varying)
│   │   │   │   ├── Cox.py          # Time Varying cox model along with Cox Data generation
│   │   │   │   ├── DRSM.py             # Deep Recurrent Survival Machine Code for Time Varying (Cox Data generation with including units) (Here units indicates Different Interval Lengths) 
│   │   │   │   ├── EncoderCox_exp2.py               # EncoderCox model with different interval_lengths to see if there is any change in c-index 
│   │   │   │   ├── EncoderCox_experiments.py            # Dummy EncoderCox model without optuna and fixed parameters to get the fast results
│   │   │   │   ├── EncoderCox.py            # EncoderCox model For TimeVarying (Has previous cox data generation, Excluding Units ) (Here units indicates Different Interval Lengths) 
│   │   │   │   ├── GRU.py              # Gated Recurrent Unit Code from Deep Recurrent Survival Machine (Cox Data generation excluding units) (Here units indicates Different Interval Lengths) 
│   │   │   │   ├── LSTM.py             # Long Short term Memory from Deep Recurrent Survival Machine (Cox Data generation excluding units) (Here units indicates Different Interval Lengths) 
│   │   │   │   ├── RNN.py              # Recurrent Neural Network from Deep Recurrent Survival Machine (Cox Data generation excluding units) (Here units indicates Different Interval Lengths) 
│   │   │   │   ├── Transformer.py      # Original Transformer Model (Cox Data generation excluding unit)  (Here units indicates Different Interval Lengths) 
│   │   ├── LOCF/                   # Implementation of Time-Varying survival models
│   │
│   ├── concordance.py              # Concordance calculation for Time-invariant Covariates (Encoder)
│   ├── Transformer_utils.py        # This file has the neural network code 
│   ├── other_models_utils.py       # Contains the functions for Cox Model, DeepSurv, RDSM(Deep Recurrent Survial Machine) for our loss, DRSM(Deep Recurrent Survival Machine Exluding our loss) using their loss which is commented out
│   ├── My_model_utils.py           # Contains the functions for Time Varying Cox Model, Original Trnasformer for time Vraying , EncoderCox (Our Model)
│   ├── DRSM_init.py                # Contains all the Deep Recurrent Survival Model initialization code including Compute_Risk function and Compute_Loss function while evaluation.
│   ├── DRSM_loss.py                # This file has the loss functions used in Deep Recurrent Survival Machines(Not used any for our task) 
│   ├── DRSM_torch.py               # Contains the code for model(RNN,LSTM,GRU)
│   ├── DRSM_utils.py               # This file contains the train function for Deep Recurrent Survival Model

│
│── requirements.txt                # List of required Python libraries
│── README.md                       # Project documentation

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


