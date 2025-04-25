import torch
import pandas as pd
import os
import sys
import random
import numpy as np
import optuna
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm


device = 'cuda' if torch.cuda.is_available() else 'cpu'

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..')))
from utils import *
from Transformer_Model import *


data_dir = '/home/careinfolab/Dr_Luo/Rohan/ICD_Codes/Results/Desc_to_ICD/Transformer'
checkpoint_dir = '/home/careinfolab/Dr_Luo/Rohan/ICD_Codes/Results/Desc_to_ICD/Transformer/checkpoints'
results_file = '/home/careinfolab/Dr_Luo/Rohan/ICD_Codes/Results/Desc_to_ICD/Transformer/results.txt'

os.makedirs(checkpoint_dir, exist_ok=True)
with open(results_file, "w") as file:
    file.write("")

tokenizer = Tokenizer.from_file("/home/careinfolab/Dr_Luo/Rohan/ICD_Codes/Notebook/bpe_tokenizer.json")


df = pd.read_csv("/home/careinfolab/Dr_Luo/Rohan/ICD_Codes/Dataset/icd10-codes-and-descriptions/Codes&Desc_cleaned.csv")

seq_len = 128
df_full = df.copy()

test_val = df_full.sample(frac=0.3, random_state=42)
test_data = test_val.sample(frac=2/3, random_state=42)
val_data = test_val.drop(test_data.index)
train_data = df_full

train_dataset = TranslationDataset(train_data, tokenizer,'Description', 'ICD_Code',seq_len)
val_dataset = TranslationDataset(val_data, tokenizer, 'Description', 'ICD_Code', seq_len)
test_dataset = TranslationDataset(test_data, tokenizer,'Description', 'ICD_Code', seq_len)

def objective(trial):
    lr = trial.suggest_float('lr', 0.0001, 0.001)
    d_model = trial.suggest_categorical('d_model', [64, 128, 256, 512])
    d_ff = trial.suggest_categorical('d_ff', [256, 512, 1024, 2048])
    N = trial.suggest_categorical('N', [2, 4, 6, 8])
    h = trial.suggest_categorical('h', [2, 4, 8])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [8,16, 32, 64])
    num_epochs = trial.suggest_categorical('num_epochs', [100,250,300])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,num_workers=4,pin_memory=True,shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1,num_workers=2,pin_memory=True,shuffle=False)  

    model = build_transformer(
        src_vocab_size=tokenizer.get_vocab_size(),
        tgt_vocab_size=tokenizer.get_vocab_size(),
        src_seq_len=seq_len,
        tgt_seq_len=seq_len,
        d_model=d_model,
        N=N,
        h=h,
        dropout=dropout,
        d_ff=d_ff,
    )
    trainer = TransformerBuild(
        num_epochs=num_epochs,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        lr=lr,
        data_dir=data_dir,
        save_ckpt_dir=checkpoint_dir,
        vocab_size=tokenizer.get_vocab_size(),
        seq_len=seq_len,
        tokenizer_src=tokenizer,
        tokenizer_tgt=tokenizer,
    )

    results = trainer.train(model)
    best_val_loss = results[-1]

    return best_val_loss


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=5)

print("Best parameters:", study.best_params)
print("Best C-index:", study.best_value)

best_params = study.best_params
train_dataloader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True,num_workers=4,pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=1,num_workers=2,pin_memory=True,shuffle=False)

model = build_transformer(
    src_vocab_size=tokenizer.get_vocab_size(),
    tgt_vocab_size=tokenizer.get_vocab_size(),
    src_seq_len=seq_len,
    tgt_seq_len=seq_len,
    d_model=best_params['d_model'],
    N=best_params['N'],
    h=best_params['h'],
    dropout=best_params['dropout'],
    d_ff=best_params['d_ff'],
)

trainer = TransformerBuild(
    num_epochs=best_params['num_epochs'],
    train_loader=train_dataloader,
    val_loader=test_dataloader,
    lr=best_params['lr'],
    data_dir=data_dir,
    save_ckpt_dir=checkpoint_dir,
    vocab_size=tokenizer.get_vocab_size(),
    seq_len=seq_len,
    tokenizer_src=tokenizer,
    tokenizer_tgt=tokenizer
)

train_loss, val_loss, train_accuracy, val_accuracy, train_precision, val_precision, train_recall, val_recall, train_f1score, val_f1score, bleu_scores, word_errors, char_errors, best_val_loss = trainer.train(model)

results = {
    'best params': best_params,
    'train_loss': train_loss,
    'test_loss': val_loss,
    'train_accuracy': train_accuracy,
    'train_precision': train_precision,
    'train_recall': train_recall,
    'train_f1': train_f1score,
    'test_accuracy': val_accuracy,
    'test_precision': val_precision,
    'test_recall': val_recall,
    'test_f1': val_f1score,
    'bleu_scores': bleu_scores,
    'word_errors': word_errors,
    'char_errors': char_errors,
    'best_val_loss': best_val_loss
}

torch.save(train_loss, f"{data_dir}/train_loss.pth")
torch.save(val_loss, f"{data_dir}/test_loss.pth")
torch.save(train_accuracy, f"{data_dir}/train_accuracy.pth")
torch.save(train_precision, f"{data_dir}/train_precision.pth")
torch.save(train_recall, f"{data_dir}/train_recall.pth")
torch.save(train_f1score, f"{data_dir}/train_f1score.pth")
torch.save(val_accuracy, f"{data_dir}/test_accuracy.pth")
torch.save(val_precision, f"{data_dir}/test_precision.pth")
torch.save(val_recall, f"{data_dir}/test_recall.pth")
torch.save(val_f1score, f"{data_dir}/test_f1score.pth")
torch.save(bleu_scores, f"{data_dir}/bleu_scores.pth")
torch.save(word_errors, f"{data_dir}/word_errors.pth")
torch.save(char_errors, f"{data_dir}/char_errors.pth")
torch.save(model.state_dict(), f"{checkpoint_dir}/best_model.pth")

with open(results_file, "a") as file:
    for key, value in results.items():
        file.write(f"{key}: {value}\n")
    file.write("\n" + "-" * 50 + "\n\n")
