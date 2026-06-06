# MfamaNet

**Multiscale Feature Aligned Multimodal Aggregation Network** for Parkinson's disease severity assessment via hand tremor analysis.

> 📝 This repository contains the **model architecture and inference code** for the paper *"A Multiscale Feature-Aligned Multimodal Aggregation Network for Parkinson's Disease Severity Assessment via Hand Tremor Analysis"*, currently under review.
>
> ⚠️ **Partial Code Release**: The full training pipeline, data preprocessing, and evaluation scripts are not included in this release. They will be made available upon paper acceptance. This repository focuses on the core model implementation, pre-trained weights, and inference examples.

## Architecture

MfamaNet is a dual-branch multimodal neural network:

- **SFE (Spatial Feature Extractor)** — ConvNeXt-based multi-scale image encoder with channel-spatial attention
- **TRE (Temporal Relation Extractor)** — Multi-scale transformer-based signal encoder with cross-attention fusion
- **Adaptive Aggregation** — Choquet-inspired fuzzy fusion of multimodal features
- **MFA Loss** — Multiscale Feature Alignment (per-scale cosine embedding + global cross-modal contrastive)

<p align="center">
  <img src="assets/architecture.png" alt="MfamaNet Architecture" width="800">
</p>

## Directory Structure

```
MfamaNet/
├── models/
│   ├── __init__.py              # Package entry
│   ├── mfama_net.py             # MfamaNet main class
│   ├── convnext_blocks.py       # ConvNeXt LayerNorm, Block, DropPath
│   ├── transformer.py           # Transformer encoder (RMSNorm, GQA, SwiGLU)
│   ├── sfe.py                   # Spatial Feature Extractor modules
│   ├── tre.py                   # Temporal Relation Extractor modules
│   └── components.py            # AdaptiveAggregation, ClassificationHead, etc.
├── params/
│   ├── meander.pth              # Pre-trained weights (Meander task)
│   ├── spiral.pth               # Pre-trained weights (Spiral task)
│   └── circle.pth               # Pre-trained weights (Circle task)
├── samples/
│   ├── meander/                 # Sample image + signal for Meander
│   ├── spiral/                  # Sample image + signal for Spiral
│   └── circle/                  # Sample image + signal for Circle
├── inference.py                 # Inference script
├── LICENSE                      # MIT License
└── README.md
```

## Datasets

This work uses two datasets:

### NewHandPD (Public)

The [NewHandPD](https://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/) benchmark is publicly available. Sample data and pre-trained weights for the Meander, Spiral, and Circle handwriting tasks are included in this repository.

### BimodalPD (Private)

The BimodalPD dataset contains sensitive patient information and **cannot be made publicly available** to protect patient privacy. Researchers interested in accessing this dataset should contact the corresponding author. Data access is subject to institutional review and data usage agreements.

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0
- torchvision
- pandas
- Pillow

Install dependencies:

```bash
pip install torch torchvision pandas Pillow
```

## Quick Start

### 1. Run inference

```bash
python inference.py
```

Output:

```
==================================================
  Task: meander
==================================================
   healthy:  pred=Healthy   probs=[0.9983, 0.0017]
   patient:  pred=Patient   probs=[0.0167, 0.9833]

==================================================
  Task: spiral
==================================================
   healthy:  pred=Healthy   probs=[0.9970, 0.0030]
   patient:  pred=Patient   probs=[0.0013, 0.9987]

==================================================
  Task: circle
==================================================
   healthy:  pred=Healthy   probs=[0.9856, 0.0144]
   patient:  pred=Patient   probs=[0.0259, 0.9741]
```


## Model Configuration

| Parameter | Description | Default |
|---|---|---|
| `num_classes` | Number of output classes | `2` |
| `num_hiddens` | Hidden dimension | `96` |
| `dropout` | Dropout rate | `0.3` |
| `signal_chans` | Signal input channels | `6` |
| `image_chans` | Image input channels | `3` |
| `scale_lst` | Multi-scale factors | `[625, 1250, ...]` |
| `num_layers` | TRE layers per scale | `[2, 4, 6, 8, 8]` |
| `num_blocks` | SFE blocks per scale | `[2, 4, 4, 6, 6]` |
| `scale_kernels` | SFE kernel sizes per scale | `[3, 5, 7, 11, 17]` |
| `drop_path_rate` | Stochastic depth rate | `0.1` |
| `tre_type` | TRE variant: `'token'` or `'res2net'` | `'token'` |
| `sfe_type` | SFE variant: `'convnext'` or `'token'` | `'convnext'` |
| `fusion_type` | Fusion method: `'choquet'`, `'concat'`, `'add'`, `'attention'` | `'choquet'` |
| `use_mfa` | Enable MFA loss | `True` |
| `mfa_weight` | MFA loss weight | `0.6` |


## License

This project is released under the [MIT License](LICENSE).

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{mfamanet,
  title={A Multiscale Feature-Aligned Multimodal Aggregation Network for Parkinson's Disease Severity Assessment via Hand Tremor Analysis},
  author={},
  journal={},
  year={2026},
  note={under review}
}
```
*(Please note that this citation will be updated with journal details upon acceptance.)*