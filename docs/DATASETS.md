# Datasets

Dataset files are not covered by this repository's Apache-2.0 license. Download
each dataset from its official source and comply with its terms. The preparation
tools generate metadata only and do not alter ownership of the underlying data.

| Dataset | Official source | Redistribution guidance |
|---|---|---|
| COCO | <https://cocodataset.org/#download> | Do not mirror the image bundle; individual Flickr image licenses and COCO terms apply. |
| VOC2012 | <https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/> | Do not mirror; the official page requires respecting the source Flickr terms. |
| PerSeg | <https://github.com/ZrrSkywalker/Personalize-SAM> | Download through the official PerSAM links; verify dataset-specific permission before redistribution. |
| ISIC2016 | <https://challenge.isic-archive.com/data/> | The official challenge page lists the 2016 Task 1 data as CC0; preserve citations and provenance. |
| BUSI | <https://doi.org/10.1016/j.dib.2019.104863> | The original release does not clearly grant redistribution rights; provide a download pointer only. |
| Kvasir-SEG | <https://www.simulamet.no/research/kvasir-seg-segmented-polyp-dataset> | Download from the publisher and verify the release terms before mirroring. |
| CAMO | <https://sites.google.com/view/ltnghia/research/camo> | CC BY-NC-SA 4.0; non-commercial and share-alike restrictions apply. |
| COD10K | <https://mmcheng.net/cod/> | Use the official training/testing links; no explicit redistribution grant is stated on the project page. |
| CHAMELEON | <https://www.polsl.pl/rau6/chameleon-database-animal-camouflage-analysis/> | Use official image and annotation downloads; confirm reuse terms with the dataset owner before mirroring. |

## Normalized split format

Each prepared split contains same-stem image, binary mask, and prompt files:

```text
<dataset>/<split>/
├── 000001.jpg
├── 000001.png
└── 000001.txt
```

The first prompt line is comma-separated. The final two values are interpreted
as the positive point `(x, y)`. Earlier values may contain a bounding box and
are retained only for compatibility with the recovered experiment layout.

Create all manifests from an existing normalized dataset root:

```bash
python scripts/prepare_all_manifests.py \
  --dataset-root /path/to/SAM-finetuning \
  --output-root data/manifests
```

Generated manifests contain paths to local files. Do not publish manifests with
private absolute paths; regenerate them with repository-relative paths before a
public release if the underlying files are legally distributable.
