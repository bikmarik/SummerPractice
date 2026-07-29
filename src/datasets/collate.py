import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}

    objects = [elem["data_object"].float() for elem in dataset_items]
    result_batch["data_object"] = pad_sequence(objects, batch_first=True)
    result_batch["labels"] = torch.tensor([elem["labels"] for elem in dataset_items])
    result_batch["lengths"] = torch.tensor([elem.shape[-1] for elem in objects])

    for key in ["utt_id", "path", "attack"]:
        if key in dataset_items[0]:
            result_batch[key] = [elem[key] for elem in dataset_items]

    return result_batch
