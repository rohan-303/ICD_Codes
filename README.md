🧠 ICD Code to Clinical Description Mapping using Transformer and Seq2Seq Models
This repository presents a deep learning-based solution for translating ICD-10 codes into their corresponding clinical descriptions. Built using custom Transformer architecture and traditional Seq2Seq models (RNN, LSTM, GRU and their deep/bidirectional versions), this project demonstrates state-of-the-art accuracy in generating meaningful clinical narratives from structured medical codes.

📂 Project Structure
bash
Copy
Edit
├── data/
│   ├── Tokens.csv            # Cleaned dataset with tokenized sequences
│   └── icd10-codes.csv       # Raw ICD code to description mapping
├── models/
│   ├── Model_rnn.py          # Seq2Seq model definitions (RNN, LSTM, GRU)
│   ├── transformer.py        # Transformer model architecture
├── training/
│   ├── transformer_train.py  # Training loop for Transformer
│   ├── rnn_first.py          # RNN training (first 5 tokens)
│   ├── rnn_middle.py         # RNN training (middle 5 tokens)
│   ├── rnn_last.py           # RNN training (last 5 tokens)
│   └── ...
├── utils/
│   └── helper.py             # Token processing, attention masks etc.
└── results/
    ├── checkpoints/
    └── metrics/
🔍 Problem Statement
Goal: Bridge the gap between structured ICD codes and human-readable clinical descriptions using neural sequence-to-sequence modeling.

We tackle two scenarios:

Scenario A: ICD code → Clinical Description

Scenario B: Clinical Description → ICD code

🧹 Data Preprocessing
Used a public ICD-10 dataset with 71,704 pairs

Applied:

Lowercasing, special character removal

Byte Pair Encoding (BPE) tokenization

Sequence padding to 128 tokens

Injected [SOS], [EOS], [PAD] tokens

Created custom TransformerDataset for structured batching and attention mask generation

⚙️ Models Implemented

Model Type	Variants
Seq2Seq	RNN, LSTM, GRU, DeepRNN, DeepLSTM, DeepGRU
Bidirectional	Bi-RNN, Bi-LSTM, Bi-GRU
Transformer	Custom implementation from scratch
📈 Training Strategy
Trained using Adam Optimizer with weight decay and gradient clipping

Loss: Cross Entropy, ignoring [PAD] tokens

Optuna for hyperparameter tuning (learning rate, hidden size, batch size, layers, etc.)

Validation loss monitored every 5 epochs

Model Checkpointing at best validation

📊 Evaluation Metrics
Accuracy

Precision / Recall / F1-Score

BLEU Score

Word Error Rate (WER)

Character Error Rate (CER)

✅ Results (Scenario A: ICD → Description)

Model	Accuracy	F1-Score
Transformer	99.82%	0.87
Bi-GRU	91%	0.50
DeepGRU	82%	0.41
RNN	65%	0.36
🔹 BLEU: 0.97
🔹 WER: 0.012
🔹 CER: 0.012

🔁 Scenario B: Description → ICD
Description padded to 60 tokens, ICD target remains 5

For RNN/LSTM/GRU: trained using first, middle, and last 5 tokens

For BiRNN, Deep variants: used best performing segment

Transformer model trained on full input sequence (no segmenting)


Model	Accuracy	F1-Score
Transformer	98%	0.78
Bi-GRU	98%	0.97
LSTM_last	98%	0.86
🧪 Optuna Hyperparameter Tuning
Search spaces:

learning_rate: [0.0001, 0.001]

d_model, d_ff: [128, 256, 2048]

num_heads: [2, 4, 8]

num_layers: [2, 4, 6]

Validation loss used for objective minimization

💻 Hardware Used
3× NVIDIA RTX 3090 GPUs

CUDA 12.4

Training simulations ran for several days due to computational load

🚧 Limitations & Future Work
Rare ICD codes with limited examples remain challenging

Future improvements:

Medical domain-specific pretraining

Model interpretability techniques

Generalization on unseen codes or lengthy descriptions

✅ Conclusion
This project demonstrated a high-performing Transformer architecture tailored for mapping ICD codes to natural descriptions. It significantly outperformed all other traditional sequence models in both forward and reverse tasks. By using effective data preparation, attention mechanisms, and structured training, we deliver a robust solution for medical text understanding and coding assistance systems.
