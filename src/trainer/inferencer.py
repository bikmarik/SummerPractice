import csv

import torch
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dls,
        out,
        metrics=None,
        btfs=None,
        skload=False,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            save_path (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        assert (
            skload or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg = self.config.inferencer

        self.device = device

        self.model = model
        self.btfs = btfs

        # define dataloaders
        self.ev_dls = {k: v for k, v in dls.items()}

        # path definition

        self.out = out

        # define metrics
        self.metrics = metrics
        if self.metrics is not None:
            self.ev_met = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.ev_met = None

        if not skload:
            # init model
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        plogs = {}
        for part, dl in self.ev_dls.items():
            logs = self._inference_part(part, dl)
            plogs[part] = logs
        return plogs

    def process_batch(self, bi, batch, metrics, part):
        """
        Run batch through the model, compute metrics, and
        save predictions to disk.

        Save directory is defined by save_path in the inference
        config and current partition.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type
                of the partition (train or inference).
            part (str): name of the partition. Used to define proper saving
                directory.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        outputs = self.model(**batch)
        batch.update(outputs)

        if metrics is not None:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        bs = batch["logits"].shape[0]
        curid = bi * bs

        for i in range(bs):
            # clone because of
            # https://github.com/pytorch/pytorch/issues/1995
            logits = batch["logits"][i].clone()
            label = batch["labels"][i].clone()
            pred = logits.argmax(dim=-1)

            oid = curid + i

            output = {
                "pred_label": pred,
                "label": label,
            }

            if getattr(self, "_csv_writer", None) is not None:
                score = batch.get("scores", batch["logits"][:, 1])[i].item()
                self._csv_writer.writerow([batch["utt_id"][i], score])

            if self.out is not None:
                # you can use safetensors or other lib here
                torch.save(output, self.out / part / f"output_{oid}.pth")

        return batch

    def _inference_part(self, part, dl):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.model.eval()

        self.ev_met.reset()

        # create Save dir
        if self.out is not None:
            (self.out / part).mkdir(exist_ok=True, parents=True)

        csv_f = None
        self._csv_writer = None
        if self.config.inferencer.get("save_scores_csv", False):
            scfile = self.config.inferencer.get("score_filename", "scores.csv")
            csv_f = (self.out / scfile).open("w", newline="")
            self._csv_writer = csv.writer(csv_f)

        with torch.no_grad():
            for bi, batch in tqdm(
                enumerate(dl),
                desc=part,
                total=len(dl),
            ):
                batch = self.process_batch(
                    bi=bi,
                    batch=batch,
                    part=part,
                    metrics=self.ev_met,
                )

        if csv_f is not None:
            csv_f.close()
            self._csv_writer = None

        return self.ev_met.result()
