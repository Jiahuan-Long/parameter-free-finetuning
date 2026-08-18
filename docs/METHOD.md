# Method and implementation

For a cached encoder output `X` with `C` channels, a directed pair `(i, j)`
means that channel `i` is replaced with channel `j`. The model parameters remain
frozen throughout the procedure.

The implementation follows four stages:

1. Encode a small search subset once and cache the features.
2. Evaluate every directed single replacement and record the mean IoU change.
3. Retain the Top-N pairs, with at most one replacement per source channel.
4. Evaluate every non-empty subset of those Top-N pairs and retain the best.

For SAM's 256-channel image embedding, stage 2 evaluates 65,280 non-identity
pairs. With `N=10`, stage 4 evaluates 1,023 combinations.

## Replacement semantics

The paper defines a mapping from the original tensor to a transformed tensor.
The default implementation therefore reads every target channel from the
original feature tensor. Results do not depend on the order of the pairs.

The recovered research scripts applied replacements sequentially in-place.
When a target is also an earlier source, this can be order-dependent. Use the
`--sequential` flag to reproduce that legacy behavior.

## Search records

JSONL is used instead of ad-hoc text so jobs are resumable and easy to merge.
Every line contains the replacement pairs, mIoU, improvement over the frozen
baseline, and number of evaluated samples.

## Memory

`--cache-device cpu` minimizes GPU memory use but transfers a cached embedding
for every prediction. `--cache-device cuda` is faster when the search features
fit in GPU memory.
