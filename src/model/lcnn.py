import torch
from torch import nn
from torch.nn import functional as F


class MFM(nn.Module):
    """
    Max-Feature-Map activation used in LightCNN-style blocks.
    """

    def forward(self, x):
        first, second = torch.chunk(x, 2, dim=1)
        return torch.maximum(first, second)


class ConvMFM(nn.Sequential):
    def __init__(self, inc, outc, ksize=3, pad=1):
        super().__init__(
            nn.Conv2d(inc, outc * 2, ksize, padding=pad),
            MFM(),
            nn.BatchNorm2d(outc),
        )


class LightCNN(nn.Module):
    """
    Compact LCNN countermeasure with STFT/log-magnitude front-end.
    """

    def __init__(
        self,
        nfft=512,
        hop=160,
        win=400,
        ncls=2,
        drop=0.5,
    ):
        super().__init__()
        self.nfft = nfft
        self.hop = hop
        self.win = win
        self.register_buffer("window", torch.hann_window(win), persistent=False)

        self.feat = nn.Sequential(
            ConvMFM(1, 32, ksize=5, pad=2),
            nn.MaxPool2d(2),
            ConvMFM(32, 48),
            nn.MaxPool2d(2),
            ConvMFM(48, 64),
            nn.MaxPool2d(2),
            ConvMFM(64, 64),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.drop = nn.Dropout(drop)
        self.final_bn = nn.BatchNorm1d(64)
        self.head = nn.Linear(64, ncls)

    def _stft_features(self, wave):
        spec = torch.stft(
            wave,
            n_fft=self.nfft,
            hop_length=self.hop,
            win_length=self.win,
            window=self.window,
            return_complex=True,
        )
        spec = torch.log1p(spec.abs())
        mean = spec.mean(dim=(1, 2), keepdim=True)
        std = spec.std(dim=(1, 2), keepdim=True).clamp_min(1e-5)
        spec = (spec - mean) / std
        return spec.unsqueeze(1)

    def forward(self, data=None, **batch):
        if data is None:
            data = batch["data_object"]
        if data.ndim == 3:
            data = data.squeeze(1)
        feats = self._stft_features(data)
        hidden = self.feat(feats).flatten(1)
        hidden = self.drop(hidden)
        if hidden.shape[0] > 1:
            hidden = self.final_bn(hidden)
        logits = self.head(F.relu(hidden))
        return {"logits": logits, "scores": logits[:, 1]}

    def __str__(self):
        allp = sum(p.numel() for p in self.parameters())
        trnp = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        return (
            super().__str__()
            + f"\nAll parameters: {allp}"
            + f"\nTrainable parameters: {trnp}"
        )
