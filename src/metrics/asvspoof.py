import torch

from src.metrics.base_metric import BaseMetric


class AccuracyMetric(BaseMetric):
    def __call__(self, logits: torch.Tensor, labs: torch.Tensor = None, **kwargs):
        if labs is None:
            labs = kwargs["labels"]
        preds = logits.argmax(dim=-1)
        return (preds == labs).float().mean().item()


class EERMetric(BaseMetric):
    full_epoch = True

    def __call__(self, scores: torch.Tensor, labs: torch.Tensor = None, **kwargs):
        if labs is None:
            labs = kwargs["labels"]
        scores = scores.detach().cpu()
        labs = labs.detach().cpu()
        bona = scores[labs == 1]
        spoof = scores[labs == 0]
        if bona.numel() == 0 or spoof.numel() == 0:
            return 1.0

        scrs = torch.cat([bona, spoof])
        labs = torch.cat([torch.ones_like(bona), torch.zeros_like(spoof)])
        order = torch.argsort(scrs)
        slabs = labs[order]

        n_bona = float(bona.numel())
        n_spoof = float(spoof.numel())
        frr = torch.cat([torch.zeros(1), torch.cumsum(slabs, dim=0) / n_bona])
        far = torch.cat(
            [
                torch.ones(1),
                (
                    n_spoof
                    - (
                        torch.arange(1, slabs.numel() + 1)
                        - torch.cumsum(slabs, dim=0)
                    )
                )
                / n_spoof,
            ]
        )
        eidx = torch.argmin(torch.abs(frr - far))
        return torch.mean(torch.stack([frr, far]), dim=0)[eidx].item()
