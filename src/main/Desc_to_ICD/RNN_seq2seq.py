import optuna
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from utils import *
from Model_rnn_seq2seq import *
import pandas as pd


device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_dir = '/home/careinfolab/Dr_Luo/Rohan/ICD_Codes/Results/RNN_seq2seq'
checkpoint_dir = '/home/careinfolab/Dr_Luo/Rohan/ICD_Codes/Results/RNN_seq2seq/checkpoints'
results_file = "/home/careinfolab/Dr_Luo/Rohan/ICD_Codes/Results/RNN_seq2seq/results.txt"

with open(results_file, "w") as file:
    file.write("")

df = pd.read_csv("/home/careinfolab/Dr_Luo/Rohan/ICD_Codes/Dataset/icd10-codes-and-descriptions/Tokens.csv", index_col=False)

df["desc_padded"] = df["desc_padded"].apply(lambda x: torch.tensor(extract_tensor_ids(x)))
df["code_padded"] = df["code_padded"].apply(lambda x: torch.tensor(extract_tensor_ids(x)))


X = df["desc_padded"].tolist()
y = df["code_padded"].tolist()

X_train, X_dummy, y_train, y_dummy = train_test_split(X, y, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_dummy, y_dummy, test_size=0.6667, random_state=42)

train_dataset = TokenDataset(X_train, y_train)
test_dataset = TokenDataset(X_test, y_test)
val_dataset = TokenDataset(X_val,y_val)

vocab_size = max([seq.max().item() for seq in X_train]) + 1
output_size = max([seq.max().item() for seq in y_train]) + 1
print(f"The vocab_size is {vocab_size}")
print(f"The  output_size is {output_size}")

def objective(trial):
    lr = trial.suggest_float('lr', 0.0001, 0.001)
    hidden = trial.suggest_int('hidden', 50, 200)
    epochs = trial.suggest_int('epochs', 100,200)
    batch_size = trial.suggest_categorical('batch_size',[16,32,64,128])
    embed_size = trial.suggest_int('embed_size', 50, 256)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,pin_memory=True,num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,pin_memory=True,num_workers=4)
    
    model = Seq2Seq(vocab_size=vocab_size,embed_size=embed_size,hidden_size=hidden,output_size=output_size,batch_size=batch_size,act='relu',model_type='rnn') 
    rnn_build = RNN_Build(num_epochs=epochs, train_loader=train_loader, test_loader=val_loader, lr=lr, data_dir=data_dir,save_ckpt_dir=checkpoint_dir,vocab_size=output_size)
    train_loss,val_loss,train_accuracy,train_precision,train_recall,train_f1score,test_accuracy,test_precision,test_recall,test_f1score,val_avg_loss = rnn_build.train(model)
    
    return val_avg_loss

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

print("Best parameters:", study.best_params)
print("Best C-index:", study.best_value)

best_params = study.best_params
lr = best_params['lr']
hidden = best_params['hidden']
epochs = best_params['epochs']
batch_size = best_params['batch_size']
embed_size = best_params['embed_size']

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,pin_memory=True)

model = SeqModel(vocab_size=vocab_size,embed_size=embed_size,hidden_size=hidden,output_size=output_size,batch_size=batch_size,act='relu',model_type='rnn')
 
rnn_build = RNN_Build(num_epochs=epochs, train_loader=train_loader, test_loader=test_loader, lr=lr, data_dir=data_dir,save_ckpt_dir=checkpoint_dir,vocab_size=vocab_size)

train_loss,test_loss,train_accuracy,train_precision,train_recall,train_f1score,test_accuracy,test_precision,test_recall,test_f1score,val_avg_loss = rnn_build.train(model)

results = {
        'train_loss': train_loss,
        'test_loss': test_loss,
        'train_accuracy': train_accuracy,
        'train_precision': train_precision,
        'train_recall': train_recall,
        'train_f1': train_f1score,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1score
            }

torch.save(train_loss, f"{data_dir}/train_loss.pth")
torch.save(test_loss, f"{data_dir}/test_loss.pth")
torch.save(train_accuracy, f"{data_dir}/train_accuracy.pth")
torch.save(train_precision, f"{data_dir}/train_precision.pth")
torch.save(train_recall, f"{data_dir}/train_recall.pth")
torch.save(train_f1score, f"{data_dir}/train_f1score.pth")

torch.save(test_accuracy, f"{data_dir}/test_accuracy.pth")
torch.save(test_precision, f"{data_dir}/test_precision.pth")
torch.save(test_recall, f"{data_dir}/test_recall.pth")
torch.save(test_f1score, f"{data_dir}/test_f1score.pth")

with open(results_file, "a") as file:
        for key, value in results.items():
            file.write(f"{key}: {value}\n")
        file.write("\n" + "-" * 50 + "\n\n")
        
        