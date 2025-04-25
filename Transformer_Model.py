import torch
import torch.nn as nn
import math
from tqdm import tqdm
from torch.utils.data import Dataset
from nltk.translate.bleu_score import corpus_bleu,SmoothingFunction
import random
import jiwer
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False)
        return self.dropout(x)

class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        assert d_model % h == 0
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)  
        attn_weights = torch.softmax(scores, dim=-1)
        if dropout is not None:
            attn_weights = dropout(attn_weights)
        return torch.matmul(attn_weights, value), attn_weights

    def forward(self, q, k, v, mask, return_attention=False):
        B = q.size(0)
        L_q = q.size(1)
        L_k = k.size(1)

        query = self.w_q(q).view(B, L_q, self.h, self.d_k).transpose(1, 2)
        key = self.w_k(k).view(B, L_k, self.h, self.d_k).transpose(1, 2)
        value = self.w_v(v).view(B, L_k, self.h, self.d_k).transpose(1, 2)

        output, attn_weights = self.attention(query, key, value, mask, self.dropout)
        output = output.transpose(1, 2).contiguous().view(B, L_q, self.d_model)
        output = self.w_o(output)

        if return_attention:
            return output, attn_weights
        return output

class EncoderBlock(nn.Module):
    def __init__(self, features, self_attn, ff, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.ff = ff
        self.res_conns = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, mask):
        x = self.res_conns[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.res_conns[1](x, self.ff)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, features, self_attn, cross_attn, ff, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.ff = ff
        self.res_conns = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask, return_attention=False):
        x = self.res_conns[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        if return_attention:
            x2, attn = self.cross_attn(x, encoder_output, encoder_output, src_mask, return_attention=True)
            x = self.res_conns[1](x, lambda x: x2)
        else:
            x = self.res_conns[1](x, lambda x: self.cross_attn(x, encoder_output, encoder_output, src_mask))
        x = self.res_conns[2](x, self.ff)
        return (x, attn) if return_attention else x

class Encoder(nn.Module):
    def __init__(self, layers, norm):
        super().__init__()
        self.layers = layers
        self.norm = norm

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, layers, norm):
        super().__init__()
        self.layers = layers
        self.norm = norm

    def forward(self, x, encoder_output, src_mask, tgt_mask, return_attention=False):
        attn_weights = []
        for layer in self.layers:
            if return_attention:
                x, attn = layer(x, encoder_output, src_mask, tgt_mask, return_attention=True)
                attn_weights.append(attn)
            else:
                x = layer(x, encoder_output, src_mask, tgt_mask)
        return (self.norm(x), attn_weights) if return_attention else self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.proj(x)

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, proj):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.proj = proj

    def encode(self, src, mask):
        return self.encoder(self.src_pos(self.src_embed(src)), mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask, return_attention=False):
        x = self.tgt_pos(self.tgt_embed(tgt))
        return self.decoder(x, encoder_output, src_mask, tgt_mask, return_attention=return_attention)

    def project(self, x):
        return self.proj(x)

def build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model, N, h, dropout, d_ff):
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    encoder_blocks = nn.ModuleList([
        EncoderBlock(d_model, MultiHeadAttentionBlock(d_model, h, dropout), FeedForwardBlock(d_model, d_ff, dropout), dropout) for _ in range(N)
    ])

    decoder_blocks = nn.ModuleList([
        DecoderBlock(d_model,
                     MultiHeadAttentionBlock(d_model, h, dropout),
                     MultiHeadAttentionBlock(d_model, h, dropout),
                     FeedForwardBlock(d_model, d_ff, dropout),
                     dropout) for _ in range(N)
    ])

    encoder = Encoder(encoder_blocks, LayerNormalization(d_model))
    decoder = Decoder(decoder_blocks, LayerNormalization(d_model))
    projection = ProjectionLayer(d_model, tgt_vocab_size)

    model = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class TransformerBuild:
    def __init__(self, num_epochs, train_loader, val_loader, lr, data_dir, save_ckpt_dir,
                 vocab_size, seq_len, tokenizer_src, tokenizer_tgt):
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.test_loader = val_loader
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
        self.data_dir = data_dir
        self.save_ckpt_dir = save_ckpt_dir
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt

    def train(self, model):
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-5)
        best_val_loss, best_epoch = float('inf'), 0

        train_loss, val_loss = [], []
        train_accuracy, train_precision, train_recall, train_f1score = [], [], [], []
        val_accuracy, val_precision, val_recall, val_f1score = [], [], [], []
        bleu_scores, word_errors, char_errors = [], [], []

        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            model.train()
            total_loss = 0
            all_preds, all_targets = [], []

            for batch in self.train_loader:
                encoder_input = batch['encoder_input'].to(device)
                decoder_input = batch['decoder_input'].to(device)
                encoder_mask = batch['encoder_mask'].to(device)
                decoder_mask = batch['decoder_mask'].to(device)
                label = batch['label'].to(device)

                optimizer.zero_grad()

                encoder_output = model.encode(encoder_input, encoder_mask)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                proj_output = model.project(decoder_output)

                loss = self.loss_fn(proj_output.reshape(-1, self.vocab_size), label.reshape(-1))

                preds = proj_output.argmax(dim=-1)
                mask = label != 0
                preds = preds[mask]
                target_tokens = label[mask]

                all_preds.append(preds.detach().cpu())
                all_targets.append(target_tokens.detach().cpu())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
                optimizer.step()

                total_loss += loss.item()

            all_preds = torch.cat(all_preds)
            all_targets = torch.cat(all_targets)
            avg_loss = total_loss / len(self.train_loader)
            train_loss.append(avg_loss)

            precision, recall, f1, _ = precision_recall_fscore_support(
                all_targets, all_preds, average='macro', zero_division=0)
            accuracy = accuracy_score(all_targets, all_preds)

            train_accuracy.append(accuracy)
            train_precision.append(precision)
            train_recall.append(recall)
            train_f1score.append(f1)

            print(f"Train Loss: {avg_loss:.4f} | Acc: {accuracy:.4f} | P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f}")

            if epoch % 5 == 0 or epoch == self.num_epochs:
                val_avg_loss, val_acc, val_prec, val_rec, val_f1, bleu, wer, cer = self.evaluate(model)
                val_loss.append(val_avg_loss)
                val_accuracy.append(val_acc)
                val_precision.append(val_prec)
                val_recall.append(val_rec)
                val_f1score.append(val_f1)
                bleu_scores.append(bleu)
                word_errors.append(wer)
                char_errors.append(cer)

                print(f"Val Loss: {val_avg_loss:.4f} | Acc: {val_acc:.4f} | P: {val_prec:.4f} | R: {val_rec:.4f} | F1: {val_f1:.4f} | BLEU: {bleu:.4f} | WER: {wer:.4f} | CER: {cer:.4f}")

                print("\nExample Predictions:")
                model.eval()
                with torch.no_grad():
                    sample_batches = random.sample(list(self.test_loader), k=1)
                    batch = sample_batches[0]
                    encoder_input = batch['encoder_input'].to(device)
                    decoder_input = batch['decoder_input'].to(device)
                    encoder_mask = batch['encoder_mask'].to(device)
                    decoder_mask = batch['decoder_mask'].to(device)
                    label = batch['label'].to(device)

                    encoder_output = model.encode(encoder_input, encoder_mask)
                    decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                    proj_output = model.project(decoder_output)
                    preds = proj_output.argmax(dim=-1)

                    for i in range(min(2, encoder_input.size(0))):
                        src_sent = self.tokenizer_src.decode(encoder_input[i].cpu().tolist(), skip_special_tokens=True)
                        tgt_sent = self.tokenizer_tgt.decode(label[i].cpu().tolist(), skip_special_tokens=True)
                        pred_sent = self.tokenizer_tgt.decode(preds[i].cpu().tolist(), skip_special_tokens=True)

                        print(f"Source   : {src_sent}")
                        print(f"Target   : {tgt_sent}")
                        print(f"Predicted: {pred_sent}\n")

                if val_avg_loss < best_val_loss:
                    best_val_loss = val_avg_loss
                    best_epoch = epoch
                    self.checkpoint(model)

        print(f"\n Best model saved from epoch {best_epoch} with val loss {best_val_loss:.4f}")

        return train_loss, val_loss, train_accuracy, val_accuracy, train_precision, val_precision, train_recall, val_recall, train_f1score, val_f1score, bleu_scores, word_errors, char_errors, best_val_loss

    def evaluate(self, model):
        from nltk.translate.bleu_score import corpus_bleu
        import jiwer

        model.eval()
        total_loss = 0
        all_preds, all_targets = [], []
        references, hypotheses = [], []

        with torch.no_grad():
            for batch in self.test_loader:
                encoder_input = batch['encoder_input'].to(device)
                decoder_input = batch['decoder_input'].to(device)
                encoder_mask = batch['encoder_mask'].to(device)
                decoder_mask = batch['decoder_mask'].to(device)
                label = batch['label'].to(device)

                encoder_output = model.encode(encoder_input, encoder_mask)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                proj_output = model.project(decoder_output)

                loss = self.loss_fn(proj_output.reshape(-1, self.vocab_size), label.reshape(-1))
                total_loss += loss.item()

                preds = proj_output.argmax(dim=-1)
                mask = label != 0
                preds = preds[mask]
                target_tokens = label[mask]

                all_preds.append(preds.detach().cpu())
                all_targets.append(target_tokens.detach().cpu())

                pred_text = self.tokenizer_tgt.decode(preds.tolist(), skip_special_tokens=True)
                target_text = self.tokenizer_tgt.decode(target_tokens.tolist(), skip_special_tokens=True)
                hypotheses.append(pred_text.split())
                references.append([target_text.split()])

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        avg_loss = total_loss / len(self.test_loader)

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='macro', zero_division=0)
        accuracy = accuracy_score(all_targets, all_preds)
        smoothing = SmoothingFunction().method1
        bleu = corpus_bleu(references, hypotheses,smoothing_function=smoothing)
        wer = jiwer.wer([" ".join(r[0]) for r in references], [" ".join(h) for h in hypotheses])
        cer = jiwer.cer([" ".join(r[0]) for r in references], [" ".join(h) for h in hypotheses])

        return avg_loss, accuracy, precision, recall, f1, bleu, wer, cer

    def checkpoint(self, model):
        datadir = self.data_dir.replace('/', '.')
        model_out_path = f"{self.save_ckpt_dir}/best_model_trainset_{datadir}.pth"
        torch.save(model, f"{self.save_ckpt_dir}/best_model.pt")
        torch.save(model.state_dict(), model_out_path)
        print(f" Checkpoint saved to {model_out_path}")