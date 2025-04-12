import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import os


class EncoderRNN(nn.Module):
    def __init__(self, input_vocab_size, embed_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.embedding = nn.Embedding(input_vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        
    def forward(self, src):
        embedded = self.embedding(src)  # [batch, src_len] → [batch, src_len, embed_size]
        outputs, hidden = self.rnn(embedded)  # outputs: all hidden states, hidden: last hidden
        return hidden
class DecoderRNN(nn.Module):
    def __init__(self, output_vocab_size, embed_size, hidden_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_vocab_size)
        
    def forward(self, input_token, hidden):
        embedded = self.embedding(input_token).unsqueeze(1)  # [batch] → [batch, 1, embed_size]
        output, hidden = self.rnn(embedded, hidden)  # output: [batch, 1, hidden_size]
        prediction = self.fc(output.squeeze(1))     # [batch, output_vocab_size]
        return prediction, hidden
    
class Seq2Seq(nn.Module):
    def __init__(self,vocab_size,embed_size,hidden_size,output_size, sos_token, eos_token, max_len=50):
        super().__init__()
        self.encoder = EncoderRNN(input_vocab_size=vocab_size,embed_size=embed_size,hidden_size=hidden_size)
        self.decoder = DecoderRNN(output_vocab_size=output_size,embed_size=embed_size,hidden_size=hidden_size)
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.max_len = max_len

    def forward(self, src, tgt=None, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        tgt_len = tgt.size(1) if tgt is not None else self.max_len
        tgt_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(src.device)
        hidden = self.encoder(src)

        input_token = torch.full((batch_size,), self.sos_token, dtype=torch.long).to(src.device)

        for t in range(tgt_len):
            output, hidden = self.decoder(input_token, hidden)
            outputs[:, t] = output

            if tgt is not None and torch.rand(1).item() < teacher_forcing_ratio:
                input_token = tgt[:, t]  # Use actual next token (teacher forcing)
            else:
                input_token = output.argmax(1)  # Use predicted token

        return outputs

class RNN_Build:
    def __init__(self, num_epochs, train_loader, test_loader, lr, save_ckpt_dir, pad_token, sos_token, eos_token):
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr = lr
        self.save_ckpt_dir = save_ckpt_dir
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token)
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token

    def train(self, model):
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        best_val_loss = float("inf")
        best_epoch = 0

        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            model.train()
            total_loss = 0

            all_preds, all_targets = [], []

            for src, tgt in self.train_loader:
                src, tgt = src.to(device), tgt.to(device)

                optimizer.zero_grad()
                output = model(src, tgt, teacher_forcing_ratio=0.5)

                output_dim = output.shape[-1]
                output_flat = output[:, :-1, :].reshape(-1, output_dim)
                tgt_flat = tgt[:, 1:].reshape(-1)

                loss = self.loss_fn(output_flat, tgt_flat)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                preds = output.argmax(2)
                mask = tgt[:, 1:] != self.pad_token
                all_preds.extend(preds[:, :-1][mask].cpu().tolist())
                all_targets.extend(tgt[:, 1:][mask].cpu().tolist())

            acc, p, r, f1 = self._metrics(all_targets, all_preds)
            print(f"Train Loss: {total_loss:.4f} | Acc: {acc:.4f} | P: {p:.4f} | R: {r:.4f} | F1: {f1:.4f}")

            if epoch % 5 == 0 or epoch == self.num_epochs:
                val_loss, val_acc, val_p, val_r, val_f1 = self.evaluate(model)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    self.checkpoint(model)

                print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | P: {val_p:.4f} | R: {val_r:.4f} | F1: {val_f1:.4f}")

        print(f"\nBest model saved from epoch {best_epoch} with val loss {best_val_loss:.4f}")
        return best_val_loss

    def evaluate(self, model):
        model.eval()
        total_loss = 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for src, tgt in self.test_loader:
                src, tgt = src.to(device), tgt.to(device)
                output = model(src, tgt=None, teacher_forcing_ratio=0.0)  # inference mode

                output_dim = output.shape[-1]
                preds = output.argmax(2)
                mask = tgt[:, 1:] != self.pad_token

                output_flat = output[:, :-1, :].reshape(-1, output_dim)
                tgt_flat = tgt[:, 1:].reshape(-1)
                loss = self.loss_fn(output_flat, tgt_flat)
                total_loss += loss.item()

                all_preds.extend(preds[:, :-1][mask].cpu().tolist())
                all_targets.extend(tgt[:, 1:][mask].cpu().tolist())

        acc, p, r, f1 = self._metrics(all_targets, all_preds)
        avg_loss = total_loss / len(self.test_loader)
        return avg_loss, acc, p, r, f1

    def checkpoint(self, model):
        os.makedirs(self.save_ckpt_dir, exist_ok=True)
        path = os.path.join(self.save_ckpt_dir, "best_model.pt")
        torch.save(model.state_dict(), path)
        print(f"Checkpoint saved to {path}")

    def _metrics(self, targets, preds):
        acc = accuracy_score(targets, preds)
        p, r, f1, _ = precision_recall_fscore_support(targets, preds, average='macro', zero_division=0)
        return acc, p, r, f1
