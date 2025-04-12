import torch
from torch.utils.data import Dataset
import re


class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = [seq.clone().detach().long() if isinstance(seq, torch.Tensor) else torch.tensor(seq, dtype=torch.long) for seq in X]
        self.Y = [seq.clone().detach().long() if isinstance(seq, torch.Tensor) else torch.tensor(seq, dtype=torch.long) for seq in Y]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Keep tensors on CPU here, let DataLoader pin them
        return self.X[idx], self.Y[idx]



def extract_tensor_ids(s):
    if isinstance(s, str):
        return [int(n) for n in re.findall(r"tensor\((\d+)\)", s)]
    elif isinstance(s, list):
        return [int(t.item()) if isinstance(t, torch.Tensor) else int(t) for t in s]
    else:
        raise ValueError(f"Unexpected data type: {type(s)}")