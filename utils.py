import torch
from torch.utils.data import Dataset
import re
from tqdm import tqdm


class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = [seq.clone().detach().long() if isinstance(seq, torch.Tensor) else torch.tensor(seq, dtype=torch.long) for seq in X]
        self.Y = [seq.clone().detach().long() if isinstance(seq, torch.Tensor) else torch.tensor(seq, dtype=torch.long) for seq in Y]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def extract_tensor_ids(s):
    if isinstance(s, str):
        return [int(n) for n in re.findall(r"tensor\((\d+)\)", s)]
    elif isinstance(s, list):
        return [int(t.item()) if isinstance(t, torch.Tensor) else int(t) for t in s]
    else:
        raise ValueError(f"Unexpected data type: {type(s)}")
    
    
class TransformerDataset(Dataset):
    def __init__(self, df,tokenizer, src_lang, tgt_lang, seq_len):
        self.df = df
        self.seq_len = seq_len
        self.tokenizer_src = tokenizer
        self.tokenizer_tgt = tokenizer
        self.pad_token = self.tokenizer_tgt.encode("[PAD]").ids[0]
        self.sos_token = self.tokenizer_tgt.encode("[SOS]").ids[0]
        self.eos_token = self.tokenizer_tgt.encode("[EOS]").ids[0]

        self.data = []

        print("Preprocessing and tokenizing dataset...", flush=True)
        for i in tqdm(range(len(df)), desc="Tokenizing"):
            src_text = df[src_lang].iloc[i]
            tgt_text = df[tgt_lang].iloc[i]

            enc_input_tokens = self.tokenizer_src.encode(src_text).ids
            dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

            enc_pad_len = seq_len - len(enc_input_tokens) - 2  
            dec_pad_len = seq_len - len(dec_input_tokens) - 1 

            if enc_pad_len < 0 or dec_pad_len < 0:
                continue  
            encoder_input = [self.sos_token] + enc_input_tokens + [self.eos_token] + [self.pad_token] * enc_pad_len
            decoder_input = [self.sos_token] + dec_input_tokens + [self.pad_token] * dec_pad_len
            label = dec_input_tokens + [self.eos_token] + [self.pad_token] * dec_pad_len
            self.data.append({
                "encoder_input": torch.tensor(encoder_input, dtype=torch.long),
                "decoder_input": torch.tensor(decoder_input, dtype=torch.long),
                "label": torch.tensor(label, dtype=torch.long),
                "src_text": src_text,
                "tgt_text": tgt_text
            })

        if len(self.data) == 0:
            raise ValueError("All sequences were too long after filtering.")

        print(f"Loaded {len(self.data)} valid examples.", flush=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        encoder_input = item["encoder_input"]
        decoder_input = item["decoder_input"]
        label = item["label"]
        encoder_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() 
        decoder_mask = (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(self.seq_len) 

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask,
            "label": label,
            "src_text": item["src_text"],
            "tgt_text": item["tgt_text"]
        }


def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
