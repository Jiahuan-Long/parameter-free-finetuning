# Reproducibility protocol

## Segmentation experiments

1. Download and normalize the dataset.
2. Generate `search` and `test` manifests.
3. Use the first 50 search records (`--max-samples 50`) for single-pair search.
4. Select Top-10 pairs with unique source channels.
5. Evaluate all 1,023 non-empty Top-10 combinations on the search split.
6. Export the best combination.
7. Evaluate that fixed combination once on the test split.

Keep the following fixed and report them with every result:

- model family, backbone, upstream commit, and checkpoint hash;
- dataset version, exact split manifest, and prompt-generation rule;
- simultaneous or legacy sequential replacement semantics;
- software environment, device, and random seed;
- result JSONL and exported replacement configuration.

## Sharding

Single-pair searches can be distributed deterministically:

```bash
# worker 0 of 4
pfft-re search-single ... --shard-index 0 --shard-count 4 \
  --output runs/coco/single-worker-0.jsonl
```

Use separate output files per worker. Concatenate the JSONL files after the jobs
finish; duplicate baseline records are harmless when selecting candidates.

## Historical-result warning

The historical scripts often searched 90-102 images even though the paper
describes a 50-image search subset, used absolute paths, and performed sequential
replacement. Treat legacy text logs as audit evidence, not as the canonical
protocol for a new release.
