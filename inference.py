import warnings
import argparse

import hydra
import torch
from hydra.utils import instantiate

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)

if not isinstance(getattr(argparse.ArgumentParser, "_check_help", None), property):
    argparse.ArgumentParser._check_help = lambda self, action: None

try:
    from hydra._internal.utils import LazyCompletionHelp

    if not hasattr(LazyCompletionHelp, "__contains__"):
        LazyCompletionHelp.__contains__ = lambda self, item: False
except Exception:
    pass


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. Runs Inferencer to calculate metrics and (or)
    save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.inferencer.seed)

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dls, btfs = get_dataloaders(config, device)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    print(model)

    # get metrics
    metrics = instantiate(config.metrics)

    # save_path for model predictions
    out = ROOT_PATH / "data" / "saved" / config.inferencer.save_path
    out.mkdir(exist_ok=True, parents=True)

    inferencer = Inferencer(
        model=model,
        config=config,
        device=device,
        dls=dls,
        btfs=btfs,
        out=out,
        metrics=metrics,
        skload=False,
    )

    logs = inferencer.run_inference()

    for part in logs.keys():
        for key, value in logs[part].items():
            fkey = part + "_" + key
            print(f"    {fkey:15s}: {value}")


if __name__ == "__main__":
    main()
