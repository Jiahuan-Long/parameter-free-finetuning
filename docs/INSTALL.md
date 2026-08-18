# Installation

## Base environment

Python 3.10 or 3.11 is recommended; Python 3.8 remains supported for parity
with the original experiments. Install the PyTorch build matching the
machine's CUDA runtime, then install the project:

```bash
pip install -e ".[dev]"
```

## SAM

The optional `sam` extra installs the upstream Segment Anything repository at a
pinned revision:

```bash
pip install -e ".[sam,dev]"
```

Download one or more official checkpoints with:

```bash
python scripts/download_sam_checkpoints.py --model vit_b
```

## SAM 2

Follow the upstream installation instructions for
[`facebookresearch/sam2`](https://github.com/facebookresearch/sam2). Install it
in the same environment and download the matching checkpoint/config pair.

Example backend arguments for SAM 2.1 Hiera-B+:

```text
--backend sam2
--checkpoint checkpoints/sam2.1_hiera_base_plus.pt
--model-config configs/sam2.1/sam2.1_hiera_b+.yaml
```

The SAM 2 backend relies on predictor feature-cache fields because upstream does
not expose a public feature-injection API. Run the test command after changing
the SAM 2 revision.

## Verification

```bash
pytest
ruff check .
pfft-re --help
```
