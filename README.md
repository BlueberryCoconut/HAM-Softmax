<div align="center">

# HAM-Softmax + DS³

## Robust Speaker Verification with Hyperbolic Margins and Dual-Space Sample Selection

[![HAM-Softmax](https://img.shields.io/badge/HAM--Softmax-ICASSP%202026-00629B?style=for-the-badge)](https://ieeexplore.ieee.org/abstract/document/11463316)
[![DS3](https://img.shields.io/badge/DS%C2%B3-Pattern%20Recognition-4C8BF5?style=for-the-badge)](https://doi.org/10.1016/j.patcog.2026.113595)

Research repository for integrating **HAM-Softmax** with **DS³ (Dual-Space Sample Selection)** to improve speaker representation learning under noisy labels.

</div>

## Research Objective

This project investigates whether the hierarchical hyperbolic decision geometry of HAM-Softmax can be combined with the noisy-label handling strategy of DS³. The intended system will use:

- **HAM-Softmax** as the speaker-classification objective;
- **feature-space selection** with multiple centres per speaker;
- **output-space Top2-Margin** analysis for separating in-distribution and out-of-distribution label noise;
- a **multi-stage training procedure** for sample selection and label correction.

The primary evaluation setting is speaker verification on VoxCeleb under clean, symmetric-noise, and asymmetric-noise conditions. Results must be reported with exact list-file hashes, seeds, checkpoints, EER, and minDCF.

## Current Status

| Component | Status |
| --- | --- |
| Original HAM-Softmax baseline | Available from the upstream implementation |
| Portable VoxCeleb1 training and evaluation | Available |
| Controlled noisy-label list experiments | Available; exact list hashes are recorded |
| ES-GMM noisy-sample baseline | Available in `main_es_gmm.py` |
| DS³ multi-centre feature-space selection | Not yet implemented |
| DS³ Top2-Margin output-space selection | Not yet implemented |
| Full HAM-Softmax + DS³ multi-stage training | Research integration in progress |

See [RESEARCH_PROGRESS.md](RESEARCH_PROGRESS.md) for verified results, known data issues, checkpoint provenance, and the current handoff state. In particular, the observed label-change rates of the existing noisy lists do not match their nominal filenames; do not treat those names alone as experimental ground truth.

## Upstream Project and Attribution

This repository is a research fork of [PunkMale/HAM-Softmax](https://github.com/PunkMale/HAM-Softmax). The original HAM-Softmax implementation and its README were authored by Zhihua Fang and collaborators. The research-fork changes add portability, noisy-label experiments, ES-GMM baselines, reproducibility records, and the planned HAM-Softmax + DS³ integration.

Original HAM-Softmax resources:

- [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/11463316)
- [arXiv](https://arxiv.org/abs/2601.19709)

DS³ reference:

- Zhihua Fang and Liang He, “DS³: Dual-space sample selection for speaker representation learning with noisy labels,” *Pattern Recognition*, 2026, DOI: [10.1016/j.patcog.2026.113595](https://doi.org/10.1016/j.patcog.2026.113595).


## Data Preparation

Prepare VoxCeleb or CN-Celeb style audio directories, then replace the dataset roots in [utils.py](utils.py) with your local paths:

```python
vox1_dev_path = '/path/to/voxceleb1/dev/wav/'
vox2_dev_path = '/path/to/voxceleb2/dev/aac/'
cn_dev_path = '/path/to/cnceleb'
eval_path = '/path/to/voxceleb1'
```

The repository uses list files under [data](data):

```text
data/
  v1_clean.txt
  v2_clean.txt
  cn_clean.txt
  vox_O.txt
  vox_E.txt
  vox_H.txt
  vox_EH_list.txt
  CN.Eval_list.txt
```

Training list format:

```text
speaker_id relative/path/to/audio.wav
```

VoxCeleb-O/E/H evaluation list format:

```text
label enroll_utterance test_utterance
```

For CN-Celeb evaluation, replace the corresponding paths in [utils.py](utils.py):

```python
eval_list1 = '/path/to/cn_veri_test.txt'
eval_list2 = '/path/to/cn_veri_test.txt'
eval_path = '/path/to/cnceleb/cn_1/eval'
```

For augmentation, replace the MUSAN and RIR paths in [datasets.py](datasets.py):

```python
self.musan_path = "/path/to/musan"
self.rir_path = "/path/to/RIRS_NOISES/simulated_rirs"
```

## Training

General training command:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --dataset v2 \
  --lambda_1 0 \
  --lambda_2 1 \
  --h_C 3 \
  --h_s 30 \
  --h_m 0.2 \
  --batch_size 256 \
  --augment
```

Important arguments:

| Argument | Description |
| --- | --- |
| `--dataset` | Dataset key: `v1`, `v2`, or `cn`. |
| `--loss_type` | Euclidean auxiliary loss: `ce`, `ces`, `am`, `aam`, or `ram`. |
| `--lambda_1` | Weight of the Euclidean loss branch. |
| `--lambda_2` | Weight of the hyperbolic loss branch. |
| `--h_C` | Hyperbolic curvature parameter. |
| `--h_m` | Hyperbolic additive margin. Use `0` for H-Softmax. |
| `--h_s` | Hyperbolic scale factor. |
| `--augment` | Enable waveform augmentation. |

## Example Experiments

The provided [run.sh](run.sh) contains three example VoxCeleb1 runs:

```bash
./run.sh
```

They correspond to:

| Experiment | Key setting |
| --- | --- |
| H-Softmax | `h_C=5`, `h_m=0`, `h_s=30`, `lambda_1=0`, `lambda_2=1` |
| HAM-Softmax | `h_C=3`, `h_m=0.2`, `h_s=30`, `lambda_1=0`, `lambda_2=1` |
| E.H. AM-Softmax | `loss_type=ram`, `h_C=3`, `h_m=0.2`, `h_s=30`, `lambda_1=0.3`, `lambda_2=0.7` |

## Evaluation and Outputs

During training, the script evaluates periodically according to the dataset setting in [utils.py](utils.py). Results and checkpoints are saved under:

```text
exps/<experiment_name>/
  train.log
  result_<experiment_name>.csv
  Vox-EH.csv
  model/
```

When training on VoxCeleb2 (`--dataset v2`), the final model is additionally evaluated on VoxCeleb-E and VoxCeleb-H with the fast evaluation dataset. The results are saved to `Vox-EH.csv`.


## Citation

If this repository is helpful for your research, please cite:

```bibtex
@INPROCEEDINGS{11463316,
  author={Fang, Zhihua and He, Liang},
  booktitle={ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title={Hyperbolic Additive Margin Softmax with Hierarchical Information for Speaker Verification},
  year={2026},
  pages={19017-19021},
  doi={10.1109/ICASSP55912.2026.11463316}
}
```
