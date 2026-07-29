from abc import abstractmethod

import torch
from numpy import inf
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from src.datasets.data_utils import inf_loop
from src.metrics.tracker import MetricTracker
from src.utils.io_utils import ROOT_PATH


class BaseTrainer:
    """
    Base class for all trainers.
    """

    def __init__(
        self,
        model,
        criterion,
        metrics,
        optimizer,
        sched,
        config,
        device,
        dls,
        logger,
        writer,
        epoch_len=None,
        sk_oom=True,
        btfs=None,
    ):
        """
        Args:
            model (nn.Module): PyTorch model.
            criterion (nn.Module): loss function for model training.
            metrics (dict): dict with the definition of metrics for training
                (metrics[train]) and inference (metrics[inference]). Each
                metric is an instance of src.metrics.BaseMetric.
            optimizer (Optimizer): optimizer for the model.
            lr_scheduler (LRScheduler): learning rate scheduler for the
                optimizer.
            config (DictConfig): experiment config containing training config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            logger (Logger): logger that logs output.
            writer (WandBWriter | CometMLWriter): experiment tracker.
            epoch_len (int | None): number of steps in each epoch for
                iteration-based training. If None, use epoch-based
                training (len(dataloader)).
            skip_oom (bool): skip batches with the OutOfMemory error.
            batch_transforms (dict[Callable] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
        """
        self.is_train = True

        self.config = config
        self.cfg = self.config.trainer

        self.device = device
        self.sk_oom = sk_oom

        self.logger = logger
        self.logst = config.trainer.get("log_step", 50)

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.sched = sched
        self.btfs = btfs

        # define dataloaders
        self.trn_dl = dls["train"]
        if epoch_len is None:
            # epoch-based training
            self.epoch_len = len(self.trn_dl)
        else:
            # iteration-based training
            self.trn_dl = inf_loop(self.trn_dl)
            self.epoch_len = epoch_len

        self.ev_dls = {
            k: v for k, v in dls.items() if k != "train"
        }

        # define epochs
        self._last = 0  # required for saving on interruption
        self.start = 1
        self.epochs = self.cfg.n_epochs

        # configuration to monitor model performance and save best

        self.sv_per = self.cfg.save_period  # checkpoint each save_period epochs
        self.mon = self.cfg.get(
            "monitor", "off"
        )  # format: "mnt_mode mnt_metric"

        if self.mon == "off":
            self.mode = "off"
            self.best = 0
        else:
            self.mode, self.metr = self.mon.split()
            assert self.mode in ["min", "max"]

            self.best = inf if self.mode == "min" else -inf
            self.stopat = self.cfg.get("early_stop", inf)
            if self.stopat <= 0:
                self.stopat = inf

        # setup visualization writer instance
        self.writer = writer

        # define metrics
        self.metrics = metrics
        self.tr_met = MetricTracker(
            *self.config.writer.loss_names,
            "grad_norm",
            *[m.name for m in self.metrics["train"]],
            writer=self.writer,
        )
        self.ev_met = MetricTracker(
            *self.config.writer.loss_names,
            *[m.name for m in self.metrics["inference"]],
            writer=self.writer,
        )

        # define checkpoint dir and init everything if required

        self.ck_dir = (
            ROOT_PATH / config.trainer.save_dir / config.writer.run_name
        )

        if config.trainer.get("resume_from") is not None:
            rpath = self.ck_dir / config.trainer.resume_from
            self._resume_checkpoint(rpath)

        if config.trainer.get("from_pretrained") is not None:
            self._from_pretrained(config.trainer.get("from_pretrained"))

    def train(self):
        """
        Wrapper around training process to save model on keyboard interrupt.
        """
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last, best=False)
            raise e

    def _train_process(self):
        """
        Full training logic:

        Training model for an epoch, evaluating it on non-train partitions,
        and monitoring the performance improvement (for early stopping
        and saving the best checkpoint).
        """
        bad = 0
        for epoch in range(self.start, self.epochs + 1):
            self._last = epoch
            result = self._train_epoch(epoch)

            # save logged information into logs dict
            logs = {"epoch": epoch}
            logs.update(result)

            # print logged information to the screen
            for key, value in logs.items():
                self.logger.info(f"    {key:15s}: {value}")

            # evaluate model performance according to configured metric,
            # save best checkpoint as best.pth
            best, stop, bad = self._monitor_performance(
                logs, bad
            )

            if epoch % self.sv_per == 0 or best:
                self._save_checkpoint(epoch, best=best, only=True)

            if stop:  # early_stop
                break

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch, including logging and evaluation on
        non-train partitions.

        Args:
            epoch (int): current training epoch.
        Returns:
            logs (dict): logs that contain the average loss and metric in
                this epoch.
        """
        self.is_train = True
        self.model.train()
        self.tr_met.reset()
        self.writer.set_step((epoch - 1) * self.epoch_len)
        self.writer.add_scalar("epoch", epoch)
        for bi, batch in enumerate(
            tqdm(self.trn_dl, desc="train", total=self.epoch_len)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    metrics=self.tr_met,
                )
            except torch.cuda.OutOfMemoryError as e:
                if self.sk_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    torch.cuda.empty_cache()  # free some memory
                    continue
                else:
                    raise e

            self.tr_met.update("grad_norm", self._get_grad_norm())

            # log current results
            if bi % self.logst == 0:
                self.writer.set_step((epoch - 1) * self.epoch_len + bi)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(bi), batch["loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate", self.sched.get_last_lr()[0]
                )
                self._log_scalars(self.tr_met)
                self._log_batch(bi, batch)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                tr_log = self.tr_met.result()
                self.tr_met.reset()
            if bi + 1 >= self.epoch_len:
                break

        logs = tr_log

        # Run val/test
        for part, dl in self.ev_dls.items():
            v_log = self._evaluation_epoch(epoch, part, dl)
            logs.update(**{f"{part}_{name}": value for name, value in v_log.items()})

        return logs

    def _evaluation_epoch(self, epoch, part, dl):
        """
        Evaluate model on the partition after training for an epoch.

        Args:
            epoch (int): current training epoch.
            part (str): partition to evaluate on
            dataloader (DataLoader): dataloader for the partition.
        Returns:
            logs (dict): logs that contain the information about evaluation.
        """
        self.is_train = False
        self.model.eval()
        self.ev_met.reset()
        f_mets = [
            met
            for met in self.metrics["inference"]
            if getattr(met, "full_epoch", False)
        ]
        f_scrs = []
        f_labs = []
        with torch.no_grad():
            for bi, batch in tqdm(
                enumerate(dl),
                desc=part,
                total=len(dl),
            ):
                batch = self.process_batch(
                    batch,
                    metrics=self.ev_met,
                )
                if f_mets:
                    f_scrs.append(batch["scores"].detach().cpu())
                    f_labs.append(batch["labels"].detach().cpu())

            if f_mets and f_scrs:
                f_bat = {
                    "scores": torch.cat(f_scrs),
                    "labels": torch.cat(f_labs),
                }
                for met in f_mets:
                    self.ev_met.update(met.name, met(**f_bat))

            self.writer.set_step(epoch * self.epoch_len, part)
            self._log_scalars(self.ev_met)
            self._log_batch(
                bi, batch, part
            )  # log only the last batch during inference

        return self.ev_met.result()

    def _monitor_performance(self, logs, bad):
        """
        Check if there is an improvement in the metrics. Used for early
        stopping and saving the best checkpoint.

        Args:
            logs (dict): logs after training and evaluating the model for
                an epoch.
            not_improved_count (int): the current number of epochs without
                improvement.
        Returns:
            best (bool): if True, the monitored metric has improved.
            stop_process (bool): if True, stop the process (early stopping).
                The metric did not improve for too much epochs.
            not_improved_count (int): updated number of epochs without
                improvement.
        """
        best = False
        stop = False
        if self.mode != "off":
            try:
                # check whether model performance improved or not,
                # according to specified metric(mnt_metric)
                if self.mode == "min":
                    ok = logs[self.metr] <= self.best
                elif self.mode == "max":
                    ok = logs[self.metr] >= self.best
                else:
                    ok = False
            except KeyError:
                self.logger.warning(
                    f"Warning: Metric '{self.metr}' is not found. "
                    "Model performance monitoring is disabled."
                )
                self.mode = "off"
                ok = False

            if ok:
                self.best = logs[self.metr]
                bad = 0
                best = True
            else:
                bad += 1

            if bad >= self.stopat:
                self.logger.info(
                    "Validation performance didn't improve for {} epochs. "
                    "Training stops.".format(self.stopat)
                )
                stop = True
        return best, stop, bad

    def move_batch_to_device(self, batch):
        """
        Move all necessary tensors to the device.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader with some of the tensors on the device.
        """
        for key in self.cfg.device_tensors:
            batch[key] = batch[key].to(self.device)
        return batch

    def transform_batch(self, batch):
        """
        Transforms elements in batch. Like instance transform inside the
        BaseDataset class, but for the whole batch. Improves pipeline speed,
        especially if used with a GPU.

        Each tensor in a batch undergoes its own transform defined by the key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform).
        """
        # do batch transforms on device
        ttype = "train" if self.is_train else "inference"
        tfms = self.btfs.get(ttype)
        if tfms is not None:
            for tname in tfms.keys():
                batch[tname] = tfms[tname](
                    batch[tname]
                )
        return batch

    def _clip_grad_norm(self):
        """
        Clips the gradient norm by the value defined in
        config.trainer.max_grad_norm
        """
        if self.config["trainer"].get("max_grad_norm", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["max_grad_norm"]
            )

    @torch.no_grad()
    def _get_grad_norm(self, ntype=2):
        """
        Calculates the gradient norm for logging.

        Args:
            norm_type (float | str | None): the order of the norm.
        Returns:
            total_norm (float): the calculated norm.
        """
        params = self.model.parameters()
        if isinstance(params, torch.Tensor):
            params = [params]
        params = [p for p in params if p.grad is not None]
        norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), ntype) for p in params]),
            ntype,
        )
        return norm.item()

    def _progress(self, bi):
        """
        Calculates the percentage of processed batch within the epoch.

        Args:
            batch_idx (int): the current batch index.
        Returns:
            progress (str): contains current step and percentage
                within the epoch.
        """
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.trn_dl, "n_samples"):
            cur = bi * self.trn_dl.batch_size
            total = self.trn_dl.n_samples
        else:
            cur = bi
            total = self.epoch_len
        return base.format(cur, total, 100.0 * cur / total)

    @abstractmethod
    def _log_batch(self, bi, batch, mode="train"):
        """
        Abstract method. Should be defined in the nested Trainer Class.

        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        return NotImplementedError()

    def _log_scalars(self, trk: MetricTracker):
        """
        Wrapper around the writer 'add_scalar' to log all metrics.

        Args:
            metric_tracker (MetricTracker): calculated metrics.
        """
        if self.writer is None:
            return
        for name in trk.keys():
            self.writer.add_scalar(f"{name}", trk.avg(name))

    def _save_checkpoint(self, epoch, best=False, only=False):
        """
        Save the checkpoints.

        Args:
            epoch (int): current epoch number.
            best (bool): if True, save the best checkpoint to 'best.pth'.
            only_best (bool): if True and the checkpoint is the best, save it only as
                'best.pth'(do not duplicate the checkpoint as
                epochEpochNumber.pth)
        """
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.sched.state_dict(),
            "best": self.best,
            "config": self.config,
        }
        fname = str(self.ck_dir / f"epoch{epoch}.pth")
        if not (only and best):
            torch.save(state, fname)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(fname, str(self.ck_dir.parent))
            self.logger.info(f"Saving checkpoint: {fname} ...")
        if best:
            bpath = str(self.ck_dir / "best.pth")
            torch.save(state, bpath)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(bpath, str(self.ck_dir.parent))
            self.logger.info("Saving current best: best.pth ...")

        lpath = str(self.ck_dir / "last.pth")
        torch.save(state, lpath)

    def _resume_checkpoint(self, rpath):
        """
        Resume from a saved checkpoint (in case of server crash, etc.).
        The function loads state dicts for everything, including model,
        optimizers, etc.

        Notice that the checkpoint should be located in the current experiment
        saved directory (where all checkpoints are saved in '_save_checkpoint').

        Args:
            resume_path (str): Path to the checkpoint to be resumed.
        """
        rpath = str(rpath)
        self.logger.info(f"Loading checkpoint: {rpath} ...")
        ckpt = torch.load(rpath, self.device, weights_only=False)
        self.start = ckpt["epoch"] + 1
        self.best = ckpt.get("best", ckpt.get("monitor_best"))

        # load architecture params from checkpoint.
        if ckpt["config"]["model"] != self.config["model"]:
            self.logger.warning(
                "Warning: Architecture configuration given in the config file is different from that "
                "of the checkpoint. This may yield an exception when state_dict is loaded."
            )
        self.model.load_state_dict(ckpt["state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if (
            ckpt["config"]["optimizer"] != self.config["optimizer"]
            or ckpt["config"]["lr_scheduler"] != self.config["lr_scheduler"]
        ):
            self.logger.warning(
                "Warning: Optimizer or lr_scheduler given in the config file is different "
                "from that of the checkpoint. Optimizer and scheduler parameters "
                "are not resumed."
            )
        else:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.sched.load_state_dict(ckpt["lr_scheduler"])

        self.logger.info(
            f"Checkpoint loaded. Resume training from epoch {self.start}"
        )

    def _from_pretrained(self, ppath):
        """
        Init model with weights from pretrained pth file.

        Notice that 'pretrained_path' can be any path on the disk. It is not
        necessary to locate it in the experiment saved dir. The function
        initializes only the model.

        Args:
            pretrained_path (str): path to the model state dict.
        """
        ppath = str(ppath)
        if hasattr(self, "logger"):  # to support both trainer and inferencer
            self.logger.info(f"Loading model weights from: {ppath} ...")
        else:
            print(f"Loading model weights from: {ppath} ...")
        ckpt = torch.load(ppath, self.device, weights_only=False)

        if ckpt.get("state_dict") is not None:
            self.model.load_state_dict(ckpt["state_dict"])
        else:
            self.model.load_state_dict(ckpt)
