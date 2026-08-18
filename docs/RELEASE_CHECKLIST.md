# Public-release checklist

The repository structure and segmentation implementation are ready for review.
The following items require author confirmation before describing the release as
a complete reproduction of every paper result:

- [ ] Recover the exact DINOv2 CIFAR classification experiment. The archived
  `Dinov2-test.py` contains only `import torch`.
- [ ] Recover the exact DINOv2 NYUv2 depth-estimation experiment.
- [ ] Confirm the final 50-image search manifests and publish their stable IDs.
- [ ] Normalize and verify the CHAMELEON segmentation split used in the paper.
- [ ] Confirm simultaneous versus legacy sequential replacement for reported tables.
- [ ] Confirm that final pair combinations were selected only on the search split;
  recovered scripts point combination search at directories named `test-1024`.
- [ ] Re-run SAM and SAM 2 backbones from clean environments and archive JSONL logs.
- [ ] Confirm dataset redistribution permissions; do not upload third-party ZIPs by default.
- [ ] Confirm repository copyright holder, contact email, and public GitHub URL.
- [ ] Add checkpoint SHA-256 hashes and the exact upstream SAM/SAM 2 revisions.
- [ ] Run a secret scan before pushing the first public commit.
