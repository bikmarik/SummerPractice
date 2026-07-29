import warnings
import argparse

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)

if not isinstance(getattr(argparse.ArgumentParser, "_check_help", None), property):
    argparse.ArgumentParser._check_help = lambda self, action: None

try:
    from hydra._internal.utils import LazyCompletionHelp

    if not hasattr(LazyCompletionHelp, "__contains__"):
        LazyCompletionHelp.__contains__ = lambda self, item: False
except Exception:
    pass


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    pcfg = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, pcfg)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dls, btfs = get_dataloaders(config, device)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    logger.info(model)

    # get function handles of loss and metrics
    loss = instantiate(config.loss_function).to(device)
    metrics = instantiate(config.metrics)

    # build optimizer, learning rate scheduler
    params = filter(lambda p: p.requires_grad, model.parameters())
    opt = instantiate(config.optimizer, params=params)
    sched = instantiate(config.lr_scheduler, optimizer=opt)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(
        model=model,
        criterion=loss,
        metrics=metrics,
        optimizer=opt,
        sched=sched,
        config=config,
        device=device,
        dls=dls,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        btfs=btfs,
        sk_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
