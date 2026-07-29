# SummerPractice: ASVspoof 2019 LCNN Countermeasure

This repository is a PyTorch Project Template solution for the final homework:
voice anti-spoofing on the ASVspoof 2019 Logical Access partition.

Only the course homework, lectures, and seminars from
`Blinorot/deep-learning-research` branch `summer_2026` were used while writing
this code. The implementation follows the homework requirements:

- countermeasure system for bonafide/spoof classification;
- LightCNN-style model with Max-Feature-Map blocks;
- STFT/log-magnitude front-end;
- CrossEntropy loss for two-class classification;
- dropout before the final BatchNorm;
- EER metric and evaluation CSV export.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

Download ASVspoof 2019 LA and set:

```bash
export ASVSPOOF_ROOT=/path/to/asvspoof2019
```

The loader supports the Kaggle-like layout used in the course seminar:

```text
$ASVSPOOF_ROOT/LA/LA/ASVspoof2019_LA_train/flac
$ASVSPOOF_ROOT/LA/LA/ASVspoof2019_LA_dev/flac
$ASVSPOOF_ROOT/LA/LA/ASVspoof2019_LA_eval/flac
$ASVSPOOF_ROOT/LA/LA/ASVspoof2019_LA_cm_protocols
```

The homework eval protocol is included at:

```text
protocols/ASVspoof2019.LA.cm.eval.trl.txt
```

## Training

```bash
python3 train.py -cn=asvspoof
```

The main config logs to WandB under project `asvspoof_lcnn`, as required by the
homework. Tune `trainer.epoch_len`, `trainer.n_epochs`, and
`dataloader.batch_size` depending on GPU memory.

## Inference and submission CSV

After training, run:

```bash
python3 inference.py -cn=inference_asvspoof \
  inferencer.from_pretrained=saved/lcnn_stft_ce/best.pth \
  inferencer.score_filename=m1bik.csv
```

The CSV will be written to:

```text
data/saved/asvspoof_eval/m1bik.csv
```

It has exactly two columns without a header:

```text
utterance_id,bonafide_score
```

This is the format expected by the homework `grading.py` script.
