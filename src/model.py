import torch
print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F

class WordGuesser(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.randn(5, 26))  # 5 letters, 26 logits each

    def forward(self):
        return F.softmax(self.logits, dim=1)