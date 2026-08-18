# Data directory

Third-party datasets are deliberately excluded from version control. Download
them from their official sources, prepare same-stem image/mask/prompt triples,
and generate JSONL manifests with `pfft-re prepare-manifest`.

See [`docs/DATASETS.md`](../docs/DATASETS.md) for sources and license notes.

## Recovered local inventory

During extraction, the manifest generator validated the following normalized
image/mask/prompt triples. These counts describe the local research archive;
they do not grant permission to redistribute the underlying files.

| Dataset | Search | Test | Train |
|---|---:|---:|---:|
| COCO | 100 | 778 | 1,814 |
| VOC2012 | 102 | 416 | 970 |
| PerSeg | 90 | 176 | 40 |
| ISIC2016 | 100 | 379 | 900 |
| BUSI | 100 | 195 | 452 |
| Kvasir-SEG | 100 | 300 | 700 |
| CAMO | 100 | 375 | 874 |
| COD10K | 100 | 555 | 1,292 |

CHAMELEON was found separately but not in the normalized segmentation layout,
so it remains a release-checklist item rather than a generated manifest.
