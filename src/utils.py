# utils.py

import torch

def one_hot_word(word):
    word = word.upper()
    indices = []

    for c in word:
        if not ('A' <= c <= 'Z'):
            raise ValueError(f"Invalid character '{c}' in word. Only A-Z letters are allowed.")
        index = ord(c) - ord('A')
        indices.append(index)

    return torch.nn.functional.one_hot(torch.tensor(indices), num_classes=26).float()


def cross_entropy_loss(pred_probs, target_onehot):
    loss = -(target_onehot * torch.log(pred_probs + 1e-9)).sum(dim=1).mean()
    return loss
