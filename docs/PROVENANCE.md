# Research-workspace provenance

This repository is a clean extraction rather than a direct publication of the
historical experiment folders. The main recovered implementation was duplicated
across several scripts under `segment-anything/notebooks/images`.

| Historical role | Recovered script family | New location |
|---|---|---|
| Cache encoder features and evaluate replacements | `search-v4-*.py`, `fine-tuning-final*.py` | `search.py`, `backends/` |
| Copy target channel into source channel | `features_copy[0,a]=features_copy[0,b]` | `channels.apply_replacements` |
| Rank single pairs and remove duplicate sources | `find_top_n_and_remove_duplicates` | `records.select_top_candidates` |
| Enumerate Top-N subsets | `generate_combinations` | `channels.generate_candidate_combinations` |
| Channel-deactivation ablation | `特征值置0.py` | expressible as a future zero-channel operation |
| SAM 2 feature injection | `sam2/search.py`, `sam2/search-v2.py` | `backends.sam2.SAM2Backend` |
| Dataset search logs | `*-search-1024.txt` | JSONL result schema |

Historical score files can be migrated with
`scripts/convert_legacy_results.py`. The converter uses `ast.literal_eval`, not
the unsafe `eval()` calls found in the original scripts.

The original workspaces also contained full copies of SAM/SAM 2, checkpoints,
IDE metadata, repeated scripts, training data, and absolute workstation paths.
Those artifacts are intentionally excluded.
