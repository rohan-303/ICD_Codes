import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
from typing import Tuple
import os

torch.cuda.empty_cache()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,batch_size, act):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state):
        output_seq, hidden_state = self.lstm(x, hidden_state)  
        out = self.h2o(output_seq)  
        return out, hidden_state

    def init_zero_hidden(self, batch_size=1):
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return (h0, c0)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size, act) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.act = act
    
    def forward(self, x, hidden_state) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.i2h(x)
        hidden_state = self.h2h(hidden_state)
        if self.act.lower() == 'tanh':
            hidden_state = torch.tanh(x + hidden_state)
        elif self.act.lower() == 'relu':
            hidden_state = torch.relu(x + hidden_state)
        
        out = self.h2o(hidden_state)
        return out, hidden_state
        
    def init_zero_hidden(self, batch_size=1) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, requires_grad=False,device=self.h2h.weight.device)
    
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size, act):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state):
        output_seq, hidden_state = self.gru(x, hidden_state)
        out = self.h2o(output_seq)
        return out, hidden_state

    def init_zero_hidden(self, batch_size=1):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)
    
class DeepRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, act):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.act = act.lower()

        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_states):
        x = self.i2h(x)
        new_hidden_states = []

        for i, h2h in enumerate(self.h2h_layers):
            h = hidden_states[i]
            h = h2h(h)
            if self.act == 'tanh':
                h = torch.tanh(x + h)
            elif self.act == 'relu':
                h = F.relu(x + h)
            else:
                raise ValueError(f"Unsupported activation: {self.act}")
            new_hidden_states.append(h)
            x = h  

        out = self.h2o(x)
        return out, new_hidden_states

    def init_zero_hidden(self, batch_size=1):
        device = next(self.parameters()).device
        return [
            torch.zeros(batch_size, self.hidden_size, device=device)
            for _ in range(self.num_layers)
        ]
    
class DeepLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state):
        output_seq, hidden_state = self.lstm(x, hidden_state)
        out = self.h2o(output_seq)
        return out, hidden_state

    def init_zero_hidden(self, batch_size=1):
        device = next(self.parameters()).device
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h0, c0)
    
class DeepGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state):
        output_seq, hidden_state = self.gru(x, hidden_state)
        out = self.h2o(output_seq)
        return out, hidden_state

    def init_zero_hidden(self, batch_size=1):
        device = next(self.parameters()).device
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h0

class BidirectionalRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, act):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.act = act

        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.h2o = nn.Linear(hidden_size * 2, output_size)  # *2 for bidirectional

    def forward(self, x, hidden_state):
        output_seq, hidden_state = self.rnn(x, hidden_state)
        out = self.h2o(output_seq)
        return out, hidden_state

    def init_zero_hidden(self, batch_size=1):
        device = next(self.parameters()).device
        num_directions = 2  # forward + backward
        h0 = torch.zeros(num_directions, batch_size, self.hidden_size, device=device)
        return h0

class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.h2o = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x, hidden_state):
        output_seq, hidden_state = self.lstm(x, hidden_state)
        out = self.h2o(output_seq)
        return out, hidden_state

    def init_zero_hidden(self, batch_size=1):
        device = next(self.parameters()).device
        num_directions = 2  
        h0 = torch.zeros(num_directions, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(num_directions, batch_size, self.hidden_size, device=device)
        return (h0, c0)

class BidirectionalGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.h2o = nn.Linear(hidden_size * 2, output_size)  # *2 for bidirectional

    def forward(self, x, hidden_state):
        output_seq, hidden_state = self.gru(x, hidden_state)
        out = self.h2o(output_seq)
        return out, hidden_state

    def init_zero_hidden(self, batch_size=1):
        device = next(self.parameters()).device
        num_directions = 2  # forward + backward
        h0 = torch.zeros(num_directions, batch_size, self.hidden_size, device=device)
        return h0


class SeqModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, batch_size, act, model_type, num_layers=1):
        super().__init__()
        self.model_type = model_type.lower()
        self.embedding = nn.Embedding(vocab_size, embed_size)

        if self.model_type == 'rnn':
            self.rnn = RNN(embed_size, hidden_size, output_size, batch_size, act)
        elif self.model_type == 'lstm':
            self.rnn = LSTM(embed_size, hidden_size, output_size, batch_size, act)
        elif self.model_type == 'gru':
            self.rnn = GRU(embed_size, hidden_size, output_size, batch_size, act)
        elif self.model_type == 'deeprnn':
            self.rnn = DeepRNN(embed_size, hidden_size, output_size, num_layers, act)
        elif self.model_type == 'deeplstm':
            self.rnn = DeepLSTM(embed_size, hidden_size, output_size, num_layers)
        elif self.model_type == 'deepgru':
            self.rnn = DeepGRU(embed_size, hidden_size, output_size, num_layers)
        elif self.model_type == 'bilstm':
            self.rnn = BidirectionalLSTM(embed_size, hidden_size, output_size)
        elif self.model_type == 'birnn':
            self.rnn = BidirectionalRNN(embed_size, hidden_size, output_size, act)
        elif self.model_type == 'bigru':
            self.rnn = BidirectionalGRU(embed_size, hidden_size, output_size)
        else:
            raise ValueError("model_type must be one of: rnn, lstm, gru, deeprnn, deeplstm, deepgru, bilstm, birnn, bigru")

    def forward(self, x, hidden):
        x_embed = self.embedding(x)

        if self.model_type in ['rnn', 'deeprnn']:
            outputs = []
            for t in range(x.size(1)):
                out_t, hidden = self.rnn(x_embed[:, t], hidden)
                outputs.append(out_t)
            out = torch.stack(outputs, dim=1)
        else:
            out, hidden = self.rnn(x_embed, hidden)
        return out


    
class RNN_Build:
    def __init__(self, num_epochs, train_loader, test_loader, lr, data_dir, save_ckpt_dir, vocab_size):
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
        self.data_dir = data_dir
        self.save_ckpt_dir = save_ckpt_dir
        self.vocab_size = vocab_size

    def train(self, model,pred):
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr,weight_decay=1e-5)
        best_val_loss, best_epoch = float('inf'), 0

        train_loss, val_loss = [], []
        train_accuracy, train_precision, train_recall, train_f1score = [], [], [], []
        test_accuracy, test_precision, test_recall, test_f1score = [], [], [], []

        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            model.train()
            total_loss = 0
            all_preds = []
            all_targets = []

            for X_batch, Y_batch in self.train_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                hidden = model.rnn.init_zero_hidden(X_batch.size(0))
                if isinstance(hidden, (list,tuple)):
                    hidden = tuple(h.to(device) for h in hidden)
                else:
                    hidden = hidden.to(device)


                optimizer.zero_grad()
                outputs = model(X_batch, hidden)

                min_len = Y_batch.shape[1] 
                
                if pred.lower() == 'first':
                    pred_logits = outputs[:, :min_len, :]
                    target_tokens = Y_batch[:, :min_len]

                elif pred.lower() == 'middle':
                    mid = min_len // 2
                    start = mid - 1
                    end = mid + 2
                    pred_logits = outputs[:, start:end, :]
                    target_tokens = Y_batch[:, start:end]

                elif pred.lower() == 'last':
                    pred_logits = outputs[:, -min_len:, :]
                    target_tokens = Y_batch[:, -min_len:]
                    
                elif pred.lower() == 'full':
                    pred_logits = outputs
                    target_tokens = Y_batch
                    
                else:
                    print('Please mention a valid Prediction Type (first, middle, last)')
                    return


                loss = self.loss_fn(
                    pred_logits.reshape(-1, pred_logits.shape[2]),
                    target_tokens.reshape(-1)
                )

                preds = pred_logits.argmax(dim=2)
                mask = target_tokens != 0
                preds = preds[mask]
                target_tokens = target_tokens[mask]
                all_preds.append(preds)
                all_targets.append(target_tokens)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
                optimizer.step()
                total_loss += loss.item()

            all_preds = torch.cat(all_preds).cpu().tolist()
            all_targets = torch.cat(all_targets).cpu().tolist()
            avg_loss = total_loss / len(self.train_loader)
            train_loss.append(avg_loss)

            precision, recall, f1, _ = precision_recall_fscore_support(
                all_targets, all_preds, average='macro', zero_division=0
            )
            accuracy = accuracy_score(all_targets, all_preds)

            train_accuracy.append(accuracy)
            train_precision.append(precision)
            train_recall.append(recall)
            train_f1score.append(f1)

            print(f"Train Loss: {avg_loss:.4f} | Acc: {accuracy:.4f} | P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f}")


            if epoch % 5 == 0 or epoch == self.num_epochs:
                val_avg_loss, val_acc, val_prec, val_rec, val_f1 = self.evaluate(model,pred=pred)

                val_loss.append(val_avg_loss)
                test_accuracy.append(val_acc)
                test_precision.append(val_prec)
                test_recall.append(val_rec)
                test_f1score.append(val_f1)

                print(f"Val Loss: {val_avg_loss:.4f} | Acc: {val_acc:.4f} | P: {val_prec:.4f} | R: {val_rec:.4f} | F1: {val_f1:.4f}")

                if val_avg_loss < best_val_loss:
                    best_val_loss = val_avg_loss
                    best_epoch = epoch
                    self.checkpoint(model)

        print(f"\n Best model saved from epoch {best_epoch} with val loss {best_val_loss:.4f}")

        return train_loss, val_loss, train_accuracy, train_precision, train_recall, train_f1score, test_accuracy, test_precision, test_recall, test_f1score, val_avg_loss

    def evaluate(self, model, pred):
        model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, Y_batch in self.test_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                hidden = model.rnn.init_zero_hidden(X_batch.size(0))
                if isinstance(hidden, (list,tuple)):
                    hidden = tuple(h.to(device) for h in hidden)
                else:
                    hidden = hidden.to(device)

                min_len = Y_batch.shape[1]

                outputs = model(X_batch, hidden)
                if pred.lower() == 'first':
                    pred_logits = outputs[:, :min_len, :]
                    target_tokens = Y_batch[:, :min_len]
                elif pred.lower() == 'middle':
                    mid = min_len // 2
                    start = mid - 1
                    end = mid + 2
                    pred_logits = outputs[:, start:end, :]
                    target_tokens = Y_batch[:, start:end]
                elif pred.lower() == 'last':
                    pred_logits = outputs[:, -min_len:, :]
                    target_tokens = Y_batch[:, -min_len:]
                elif pred.lower() == 'full':
                    pred_logits = outputs
                    target_tokens = Y_batch
                else:
                    print('Please mention a valid Prediction Type (first, middle, last)')
                    return

                loss = self.loss_fn(
                    pred_logits.reshape(-1, pred_logits.shape[2]),
                    target_tokens.reshape(-1).long()
                )

                preds = pred_logits.argmax(dim=2)
                mask = target_tokens != 0
                preds = preds[mask]
                target_tokens = target_tokens[mask]

                if preds.numel() == 0 or target_tokens.numel() == 0:
                    continue  

                all_preds.append(preds)
                all_targets.append(target_tokens)
                total_loss += loss.item()

        all_preds = torch.cat(all_preds).cpu().tolist()
        all_targets = torch.cat(all_targets).cpu().tolist()
        avg_loss = total_loss / len(self.test_loader)

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='macro', zero_division=0
        )
        accuracy = accuracy_score(all_targets, all_preds)

        return avg_loss, accuracy, precision, recall, f1


    def checkpoint(self, model):
        datadir = self.data_dir.replace('/', '.')
        model_out_path = f"{self.save_ckpt_dir}/best_model_trainset_{datadir}.pth"
        torch.save(model, f"{self.save_ckpt_dir}/best_model.pt")
        torch.save(model.state_dict(), model_out_path)
        print(f" Checkpoint saved to {model_out_path}")