# Dataset samples

This directory contains 100 normalized samples for each of eight datasets, or
800 records in total. Seven datasets use `search-1024`; the recovered PerSeg
archive contains only 90 search images, so its 100 non-duplicated examples come
from `test-1024`. This bundle is for workflow smoke tests, not for reporting
paper benchmark results or distributing the complete benchmarks.

Every record has three same-stem files:

- `.jpg`: RGB input image;
- `.png`: binary segmentation mask;
- `.txt`: positive point prompt in `x,y` form.

The repository-relative [`manifest.jsonl`](manifest.jsonl) indexes all 800
records and can be checked immediately after installation:

```bash
pfft-re validate-manifest data/samples/manifest.jsonl
```

The images and annotations are third-party material and are **not** covered by
the repository's Apache-2.0 license. Copyright and dataset-specific conditions
remain with their respective owners. Use these files only as permitted by the
original source terms, and download the full datasets from the official links
below.

| Directory | Dataset | Source split | Official source | Terms noted by the source/release |
|---|---|---|---|---|
| `busi` | BUSI | `search-1024` | <https://doi.org/10.1016/j.dib.2019.104863> | No explicit redistribution license confirmed |
| `camo` | CAMO | `search-1024` | <https://sites.google.com/view/ltnghia/research/camo> | CC BY-NC-SA 4.0 |
| `coco` | COCO | `search-1024` | <https://cocodataset.org/#download> | Per-image Flickr licenses and COCO terms apply |
| `cod10k` | COD10K | `search-1024` | <https://mmcheng.net/cod/> | No explicit redistribution grant confirmed |
| `isic2016` | ISIC 2016 | `search-1024` | <https://challenge.isic-archive.com/data/> | 2016 Task 1 files listed as CC0 |
| `kvasir-seg` | Kvasir-SEG | `search-1024` | <https://www.simulamet.no/research/kvasir-seg-segmented-polyp-dataset> | Verify publisher release terms |
| `perseg` | PerSeg | `test-1024` | <https://github.com/ZrrSkywalker/Personalize-SAM> | Verify dataset-specific terms |
| `voc2012` | PASCAL VOC 2012 | `search-1024` | <https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/> | Source Flickr terms apply |

For complete provenance and redistribution guidance, see
[`docs/DATASETS.md`](../../docs/DATASETS.md).

Maintainers with the normalized source datasets can recreate this directory:

```bash
python scripts/export_dataset_samples.py \
  --dataset-root /path/to/SAM-finetuning \
  --output-root data/samples \
  --samples-per-dataset 100
```
