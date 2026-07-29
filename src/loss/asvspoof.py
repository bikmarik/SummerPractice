import torch
from torch import nn


class CrossEntropyCM(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, labs: torch.Tensor = None, **batch):
        if labs is None:
            labs = batch["labels"]
        return {"loss": self.loss(logits, labs)}
