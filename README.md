# ChordMini

This project explores a two-stage training pipeline for chord recognition with knowledge distillation as a regularizer.
It includes the pure-transformer ChordNet architecture with dual encoders (frequency and temporal) and a smoothing pipeline for prediction.

Web application for testing chord recognition models with music audio (youtube/local) available at: https://github.com/ptnghia-j/ChordMiniApp and cloud-hosted/online version: https://www.chordmini.me

All commands below assume you are already inside this folder:

```bash
cd ChordMini
```

All of the implementation details are in the `src` folder.

## Setup

Create a project-local virtual environment and install only the packages used by ChordMini:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` includes `setuptools==80.9.0` on purpose. That pin keeps
MP3 loading working correctly with the current `librosa` and `audioread` stack.

## Included Checkpoints

- `checkpoints/btc_model_best.pth`: BTC CL (full) checkpoint
- `checkpoints/2e1d_model_best.pth`: ChordNet (2E1D) CL (full) checkpoint
- `checkpoints/btc_model_large_voca.pt`: original BTC teacher (https://github.com/jayg996/BTC-ISMIR19)

## Data Layout

```text
data/
├── labeled/
│   ├── audio/
│   └── chordlab/
└── unlabeled/
```

The stage-1 examples below use `--audio_dir data/unlabeled`. Stage 1 does not
default to that path automatically, so pass `--audio_dir` explicitly when using
online audio mode. If you want to reuse another audio collection, point
`--audio_dir` at that directory; `.lab` files are not read in stage 1.

### Training Data

>  For unlabeled data, the experiments in this repository were run with a mix of public music datasets:

>  - FMA: https://github.com/mdeff/fma
>  - DALI: https://github.com/gabolsgabs/DALI
>  - MAESTRO: official dataset page https://magenta.tensorflow.org/datasets/maestro

>  For the labeled dataset used in our chord-recognition experiments, access is restricted due to copyright constraints. The labeled audio and annotation files are therefore not included in this repository. Please contact the developer if you need access for research or reproduction purposes.

## Stage 1: Pseudo-Label Training

Run BTC pseudo-label training from audio only, using the BTC teacher to
generate pseudo-labels:

```bash
python src/training_scripts/train_pseudo_labeling.py \
  --model_type BTC \
  --audio_dir data/unlabeled \
  --teacher_checkpoint checkpoints/btc_model_large_voca.pt \
  --save_dir checkpoints/pseudo_labeling_btc \
  --batch_size 256 \
  --num_epochs 100 \
  --learning_rate 1e-4 \
  --weight_decay 1e-5 \
  --early_stopping_patience 10 \
  --lr_schedule cosine \
  --use_warmup \
  --warmup_epochs 10 \
  --warmup_start_lr 1e-4 \
  --warmup_end_lr 3e-4 \
  --min_learning_rate 1e-6 \
  --seq_len 108 \
  --stride 108 \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --use_focal_loss \
  --focal_gamma 2.0 \
  --seed 42 \
  --num_workers 0 
```

Expected best checkpoint:

```text
checkpoints/pseudo_labeling_btc/best_model.pth
```

Optional KD regularization in stage 1:

- `--teacher_checkpoint` is used for online pseudo-label generation in audio-only stage 1.
- `--use_kd` is optional and additionally enables KD loss against the teacher logits during student training.
- The stage-1 KD defaults in [`config/ChordMini.yaml`](/Users/nghiaphan/Desktop/ChordMini/config/ChordMini.yaml) are `use_kd: false`, `kd_alpha: 0.5`, and `temperature: 3.0`.

Example stage-1 run with pseudo-label generation plus KD enabled:

```bash
python src/training_scripts/train_pseudo_labeling.py \
  --model_type BTC \
  --audio_dir data/unlabeled \
  --teacher_checkpoint checkpoints/btc_model_large_voca.pt \
  --save_dir checkpoints/pseudo_labeling_btc_kd \
  --batch_size 256 \
  --num_epochs 100 \
  --learning_rate 1e-4 \
  --weight_decay 1e-5 \
  --early_stopping_patience 10 \
  --lr_schedule cosine \
  --use_warmup \
  --warmup_epochs 10 \
  --warmup_start_lr 1e-4 \
  --warmup_end_lr 3e-4 \
  --min_learning_rate 1e-6 \
  --seq_len 108 \
  --stride 108 \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --use_kd \
  --kd_alpha 0.5 \
  --temperature 3.0 \
  --use_focal_loss \
  --focal_gamma 2.0 \
  --seed 42 \
  --num_workers 0
```

Resume a stopped stage-1 run from a trainer checkpoint in the same output directory:

```bash
python src/training_scripts/train_pseudo_labeling.py \
  --model_type BTC \
  --audio_dir data/unlabeled \
  --teacher_checkpoint checkpoints/btc_model_large_voca.pt \
  --save_dir checkpoints/pseudo_labeling_btc \
  --resume_checkpoint checkpoints/pseudo_labeling_btc/checkpoint_epoch_10.pth \
  --batch_size 256 \
  --num_epochs 100 \
  --learning_rate 1e-4 \
  --weight_decay 1e-5 \
  --early_stopping_patience 10 \
  --lr_schedule cosine \
  --use_warmup \
  --warmup_epochs 10 \
  --warmup_start_lr 1e-4 \
  --warmup_end_lr 3e-4 \
  --min_learning_rate 1e-6 \
  --seq_len 108 \
  --stride 108 \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --use_focal_loss \
  --focal_gamma 2.0 \
  --seed 42 \
  --num_workers 0
```

When resuming, `--num_epochs` is the total target epoch count for the run.
For example, resuming from `checkpoint_epoch_10.pth` with `--num_epochs 100`
continues at epoch 11 and trains through epoch 100.

## Stage 2: Continual Learning

Fine-tune the pseudo-labeled student on labeled audio plus chord annotations:

This assumes `checkpoints/pseudo_labeling_btc/best_model.pth` already exists
from a completed stage-1 run.

```bash
python src/training_scripts/train_continual_learning.py \
  --model_type BTC \
  --student_checkpoint checkpoints/pseudo_labeling_btc/best_model.pth \
  --teacher_checkpoint checkpoints/btc_model_large_voca.pt \
  --audio_dir data/labeled/audio \
  --label_dir data/labeled/chordlab \
  --save_dir checkpoints/continual_learning_btc \
  --batch_size 128 \
  --num_epochs 50 \
  --learning_rate 1e-5 \
  --weight_decay 1e-5 \
  --early_stopping_patience 10 \
  --seq_len 108 \
  --stride 54 \
  --kd_alpha 0.3 \
  --temperature 3.0 \
  --selective_kd \
  --kd_confidence_threshold 0.9 \
  --kd_min_confidence_threshold 0.1 \
  --use_focal_loss \
  --focal_gamma 2.0 \
  --train_ratio 0.7 \
  --val_ratio 0.1 \
  --seed 42 \
  --num_workers 0 
```

Important: this script writes the main output inside a split subfolder:

```text
checkpoints/continual_learning_btc/single_split/best_model.pth
```

Resume a stopped stage-2 run from a saved trainer checkpoint:

```bash
python src/training_scripts/train_continual_learning.py \
  --model_type BTC \
  --teacher_checkpoint checkpoints/btc_model_large_voca.pt \
  --audio_dir data/labeled/audio \
  --label_dir data/labeled/chordlab \
  --save_dir checkpoints/continual_learning_btc \
  --resume_checkpoint checkpoints/continual_learning_btc/single_split/checkpoint_epoch_10.pth \
  --batch_size 128 \
  --num_epochs 50 \
  --learning_rate 1e-5 \
  --weight_decay 1e-5 \
  --early_stopping_patience 10 \
  --seq_len 108 \
  --stride 54 \
  --kd_alpha 0.3 \
  --temperature 3.0 \
  --selective_kd \
  --kd_confidence_threshold 0.9 \
  --kd_min_confidence_threshold 0.1 \
  --use_focal_loss \
  --focal_gamma 2.0 \
  --train_ratio 0.7 \
  --val_ratio 0.1 \
  --seed 42 \
  --num_workers 0
```

As with stage 1, `--num_epochs` stays the total target epoch count.

## Supervised/Baseline Training From Scratch

Use `src/training_scripts/train_from_scratch.py` when you want to train directly from the labeled
audio dataset without first running pseudo-label training. This script
initializes a fresh BTC or ChordNet model and trains with ground-truth chord
labels.

Single-split supervised training from scratch:

```bash
python src/training_scripts/train_from_scratch.py \
  --model_type BTC \
  --audio_dir data/labeled/audio \
  --label_dir data/labeled/chordlab \
  --save_dir checkpoints/from_scratch_btc \
  --batch_size 8 \
  --num_epochs 100 \
  --early_stopping_patience 10 \
  --num_workers 0 \
  --max_songs 100 \
  --no_kd
```

Expected best checkpoint:

```text
checkpoints/from_scratch_btc/single_split/best_model.pth
```

Resume a stopped supervised run from a trainer checkpoint:

```bash
python src/training_scripts/train_from_scratch.py \
  --model_type BTC \
  --audio_dir data/labeled/audio \
  --label_dir data/labeled/chordlab \
  --save_dir checkpoints/from_scratch_btc \
  --resume_checkpoint checkpoints/from_scratch_btc/single_split/checkpoint_epoch_10.pth \
  --batch_size 8 \
  --num_epochs 100 \
  --early_stopping_patience 10 \
  --num_workers 0 \
  --max_songs 100 \
  --no_kd
```

As with the other training entry points, `--num_epochs` is the total target
epoch count. Resuming from `checkpoint_epoch_10.pth` with `--num_epochs 100`
continues at epoch 11 and trains through epoch 100.

### Cross Validation

`src/training_scripts/train_from_scratch.py` is also cross-validation compatible.

Run one specific fold:

```bash
python src/training_scripts/train_from_scratch.py \
  --model_type BTC \
  --audio_dir data/labeled/audio \
  --label_dir data/labeled/chordlab \
  --save_dir checkpoints/from_scratch_btc_cv \
  --batch_size 8 \
  --num_epochs 100 \
  --early_stopping_patience 10 \
  --num_workers 0 \
  --max_songs 100 \
  --no_kd \
  --use_cv \
  --n_folds 5 \
  --fold 0
```

Run all folds sequentially:

```bash
python src/training_scripts/train_from_scratch.py \
  --model_type BTC \
  --audio_dir data/labeled/audio \
  --label_dir data/labeled/chordlab \
  --save_dir checkpoints/from_scratch_btc_cv \
  --batch_size 8 \
  --num_epochs 100 \
  --early_stopping_patience 10 \
  --num_workers 0 \
  --max_songs 100 \
  --no_kd \
  --use_cv \
  --n_folds 5 \
  --run_all_folds
```

Resume is supported for a single selected fold, but not with `--run_all_folds`.

When cross validation is enabled, fold outputs are written like:

```text
checkpoints/from_scratch_btc_cv/fold_0/best_model.pth
checkpoints/from_scratch_btc_cv/fold_1/best_model.pth
...
checkpoints/from_scratch_btc_cv/cv_results.json
```

## Test: Generate `.lab` For One Audio File

Example using the included ChordNet checkpoint:

```bash
python src/evaluation/test.py \
  --model_type ChordNet \
  --checkpoint checkpoints/2e1d_model_best.pth \
  --config config/ChordMini.yaml \
  --audio_dir data/labeled/audio/<your-file>.mp3 \
  --save_dir outputs/test_single_chordnet \
  --use_overlap \
  --use_gaussian \
  --kernel_size 9 \
  --vote_aggregation logit \
  --min_segment_duration 0.5 \
  --smooth_predictions
```

This writes:

```text
outputs/test_single_chordnet/<your-file>.lab
```

Example using the included BTC checkpoint with temporal smoothing, overlap, and
logit aggregation enabled:

```bash
python src/evaluation/test.py \
  --model_type BTC \
  --checkpoint checkpoints/btc_model_best.pth \
  --config config/ChordMini.yaml \
  --audio_dir data/labeled/audio/<your-file>.mp3 \
  --save_dir outputs/test_single_btc_smoothed \
  --smooth_logits \
  --use_overlap \
  --use_gaussian \
  --kernel_size 9 \
  --vote_aggregation logit \
  --min_segment_duration 0.5 \
  --smooth_predictions

```

This writes:

```text
outputs/test_single_btc_smoothed/<your-file>.lab
```

`src/evaluation/test.py` accepts either:

- a single audio file path in `--audio_dir`
- a directory of audio files in `--audio_dir`

## Evaluation: Labeled Audio Subset

Evaluate a reproducible random subset of labeled audio:

```bash
python src/evaluation/test_labeled_audio.py \
  --model_type ChordNet \
  --checkpoint checkpoints/2e1d_model_best.pth \
  --audio_dir data/labeled/audio \
  --label_dir data/labeled/chordlab \
  --random_subset \
  --max_songs 200 \
  --seed 42 \
  --use_overlap \
  --use_gaussian \
  --vote_aggregation logit \
  --smooth_predictions \
  --output outputs/eval_subset_chordnet.json
```

This writes:

- `outputs/eval_subset_chordnet.json`
- plots under `evaluation_visualizations/<dataset_id>/`

BTC also supports the same style of temporal smoothing and overlap-aware
aggregation.

```bash
python src/evaluation/test_labeled_audio.py \
  --model_type BTC \
  --checkpoint checkpoints/btc_model_best.pth \
  --audio_dir data/labeled/audio \
  --label_dir data/labeled/chordlab \
  --random_subset \
  --max_songs 50 \
  --seed 42 \
  --smooth_logits \
  --use_overlap \
  --use_gaussian \
  --vote_aggregation logit \
  --smooth_predictions \
  --output outputs/eval_subset_btc.json
```


## Citation
To be updated

## Future Work Directions
- Larger vocabulary size supporting inversions, chord extensions
- Stronger baseline models:
  + Model using decomposition outputs
  + Model with different architectures: Conformer, etc.
  + Model trained with more data (for stronger teacher pseudo-labels)
- Enhance pseudo-labeling:
  + Pseudo-labeling with multiple teachers
  + Pseudo-labeling with filtering
- Enhance continual learning:
  + continual learning with multiple teachers
  + continual learning with replay, memory, or other regularization techniques

## License

This repository is licensed under the MIT License. See `LICENSE` for details.

The MIT license applies to the repository source code, configuration, and documentation unless noted otherwise. It does not grant rights to third-party datasets, and it should not be assumed to cover any dataset content whose redistribution is restricted by copyright or dataset-specific terms.
