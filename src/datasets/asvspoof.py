from pathlib import Path

import torch

from src.datasets.base_dataset import BaseDataset


class ASVspoofDataset(BaseDataset):
    """
    ASVspoof 2019 Logical Access dataset reader.

    The seminar shows that CM protocols contain utterance ids and bonafide/spoof
    labels. This class reads those protocols and returns raw waveforms for a
    two-class countermeasure model.
    """

    def __init__(
        self,
        root,
        proto,
        split,
        sr=16000,
        maxsec=4.0,
        *args,
        **kwargs,
    ):
        self.root = Path(root)
        self.split = split
        self.sr = sr
        self.maxn = int(sr * maxsec) if maxsec else None
        self.auddir = self._resolve_audio_dir()
        index = self._read_protocol(Path(proto))
        super().__init__(index, *args, **kwargs)

    def _read_protocol(self, proto):
        index = []
        with proto.open("r", encoding="utf-8") as protocol:
            for line in protocol:
                spk, utt_id, _, attack, label = line.strip().split()
                apath = self._find_audio_path(utt_id)
                index.append(
                    {
                        "speaker_id": spk,
                        "utt_id": utt_id,
                        "attack": attack,
                        "path": str(apath),
                        "label": 1 if label == "bonafide" else 0,
                    }
                )
        return index

    def _resolve_audio_dir(self):
        cands = [
            self.root
            / "LA"
            / "LA"
            / f"ASVspoof2019_LA_{self.split}"
            / "flac",
            self.root
            / f"ASVspoof2019_LA_{self.split}"
            / "flac",
            self.root
            / "LA"
            / f"ASVspoof2019_LA_{self.split}"
            / "flac",
        ]
        for path in cands:
            if path.exists():
                return path
        return cands[0]

    def _find_audio_path(self, utt_id):
        return self.auddir / f"{utt_id}.flac"

    def load_object(self, path):
        try:
            import soundfile as sf

            audio, sr = sf.read(path, dtype="float32", always_2d=True)
            wave = torch.from_numpy(audio).transpose(0, 1)
        except ImportError:
            import torchaudio

            wave, sr = torchaudio.load(path)

        wave = wave.mean(dim=0)
        if sr != self.sr:
            import torchaudio

            wave = torchaudio.functional.resample(wave, sr, self.sr)

        if self.maxn is not None:
            if wave.numel() > self.maxn:
                wave = wave[: self.maxn]
            elif wave.numel() < self.maxn:
                wave = torch.nn.functional.pad(
                    wave, (0, self.maxn - wave.numel())
                )
        return wave

    def __getitem__(self, ind):
        item = super().__getitem__(ind)
        meta = self._index[ind]
        item["utt_id"] = meta["utt_id"]
        item["attack"] = meta["attack"]
        item["path"] = meta["path"]
        return item


class DummyASVspoofDataset(BaseDataset):
    """
    Small generated dataset for checking the template pipeline without
    downloading ASVspoof.
    """

    def __init__(
        self,
        dlen=16,
        alen=8000,
        name="dummy",
        *args,
        **kwargs,
    ):
        self.alen = alen
        index = [
            {"path": f"{name}_{i}", "label": i % 2, "utt_id": f"DUMMY_{i:04d}"}
            for i in range(dlen)
        ]
        super().__init__(index, *args, **kwargs)

    def load_object(self, path):
        label = int(path.split("_")[-1]) % 2
        base = torch.randn(self.alen) * 0.05
        if label == 1:
            base[::80] += 0.25
        return base

    def __getitem__(self, ind):
        item = super().__getitem__(ind)
        item["utt_id"] = self._index[ind]["utt_id"]
        return item
