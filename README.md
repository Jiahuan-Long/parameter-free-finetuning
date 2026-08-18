# Parameter-Free Fine-tuning via Redundancy Elimination

Reference implementation of **Parameter-Free Fine-tuning via Redundancy
Elimination for Vision Foundation Models** (AAAI 2026).

This method adapts a frozen vision model without training or adding parameters.
It finds redundant feature channels and replaces them with more effective
channels already present in the pretrained representation.

## How it works

```mermaid
flowchart LR
    A[Search images] --> B[Cache encoder features]
    B --> C[Search all single replacements]
    C --> D[Keep Top-N pairs]
    D --> E[Search pair combinations]
    E --> F[Export best configuration]
    F --> G[Evaluate once on test data]
```

For SAM, a pair `source:target` means: overwrite feature channel `source` with
channel `target`. Model weights remain frozen during every step.

## Quick start

### 1. Install

Python 3.10 is recommended. Install the PyTorch build matching your CUDA
version first, then run:

```bash
git clone https://github.com/Jiahuan-Long/parameter-free-finetuning.git
cd parameter-free-finetuning

conda create -n pfft-re python=3.10 -y
conda activate pfft-re
pip install -e ".[sam,dev]"
```

Download the official SAM ViT-B checkpoint:

```bash
python scripts/download_sam_checkpoints.py --model vit_b
```

See [docs/INSTALL.md](docs/INSTALL.md) for SAM 2 and environment details.

### 2. Prepare a dataset manifest

Each split uses same-stem image, binary mask, and point-prompt files:

```text
search-1024/
├── example.jpg   # RGB image
├── example.png   # binary mask
└── example.txt   # x1,y1,x2,y2,point_x,point_y
```

Index and validate the split without copying its images:

```bash
pfft-re prepare-manifest \
  data/COCO/search-1024 \
  data/manifests/coco-search.jsonl

pfft-re validate-manifest data/manifests/coco-search.jsonl
```

### 3. Search single-channel replacements

```bash
pfft-re search-single \
  --backend sam \
  --model-type vit_b \
  --checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --manifest data/manifests/coco-search.jsonl \
  --max-samples 50 \
  --output runs/coco/sam-vit-b-single.jsonl
```

SAM has 256 embedding channels, so this evaluates 65,280 directed,
non-identity pairs. Searches resume automatically. Use `--shard-index` and
`--shard-count` to distribute the work.

### 4. Search combinations and export the best one

```bash
pfft-re search-combinations \
  --backend sam \
  --model-type vit_b \
  --checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --manifest data/manifests/coco-search.jsonl \
  --max-samples 50 \
  --single-results runs/coco/sam-vit-b-single.jsonl \
  --top-n 10 \
  --output runs/coco/sam-vit-b-combinations.jsonl

pfft-re select \
  runs/coco/sam-vit-b-combinations.jsonl \
  configs/replacements/sam_vit_b/coco.json \
  --dataset COCO \
  --backend sam
```

### 5. Evaluate the fixed configuration on the test split

```bash
pfft-re evaluate \
  --backend sam \
  --model-type vit_b \
  --checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --manifest data/manifests/coco-test.jsonl \
  --config configs/replacements/sam_vit_b/coco.json \
  --output runs/coco/sam-vit-b-test.jsonl
```

Do not select channels on the test split. Search on `search`, freeze the
exported configuration, and evaluate it once on `test`.

## Main commands

| Command | Purpose |
|---|---|
| `prepare-manifest` | Convert a flat image/mask/prompt split to JSONL |
| `validate-manifest` | Check every referenced image and mask |
| `search-single` | Evaluate every directed channel replacement |
| `search-combinations` | Evaluate all non-empty subsets of the Top-N pairs |
| `select` | Export the best combination as reusable JSON |
| `evaluate` | Evaluate an exported config or explicit `--pairs` |

Run `pfft-re <command> --help` for all options.

## Repository layout

```text
src/redundancy_elimination/
├── backends/       # SAM and SAM 2 feature adapters
├── channels.py     # replacement and combination generation
├── datasets.py     # manifests, masks, and point prompts
├── search.py       # feature caching and mIoU search loop
├── records.py      # resumable JSONL results and config export
└── cli.py          # command-line interface
scripts/            # checkpoint, manifest, and legacy-result utilities
tests/              # replacement, dataset, record, and search tests
docs/               # method, installation, provenance, and release notes
```

## Data and third-party assets

SAM/SAM 2 source, checkpoints, and third-party datasets are intentionally not
vendored. Dataset files have independent licenses. Official download sources
and redistribution notes for COCO, VOC2012, PerSeg, ISIC2016, BUSI,
Kvasir-SEG, CAMO, COD10K, and CHAMELEON are in
[docs/DATASETS.md](docs/DATASETS.md).

## Reproducibility notes

- Replacement is simultaneous and order-independent by default.
- `--sequential` reproduces the order-dependent historical scripts.
- Every JSONL result begins with the frozen-model baseline.
- Exact DINOv2 classification/depth scripts were not present in the recovered
  archive; do not claim reproduction of those tables yet.

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) and
[docs/RELEASE_CHECKLIST.md](docs/RELEASE_CHECKLIST.md) before publishing new
paper-result claims.

## Development

```bash
ruff check .
pytest
```

## Citation

```bibtex
@inproceedings{long2026parameterfree,
  title={Parameter-Free Fine-tuning via Redundancy Elimination for Vision Foundation Models},
  author={Long, Jiahuan and Jiang, Tingsong and Yao, Wen and Xiong, Yizhe and Xu, Zhengqin and Jia, Shuai and Liu, Hanqing and Ma, Chao},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={28},
  pages={24035--24043},
  year={2026},
  doi={10.1609/aaai.v40i28.39581}
}
```

## License

Repository code is released under Apache-2.0. Dataset, checkpoint, SAM, and
SAM 2 licenses apply separately.
