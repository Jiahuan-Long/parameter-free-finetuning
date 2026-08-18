# Replacement configurations

Store reviewed, fixed replacement pairs here after running the clean search
protocol. Historical `*-test-1024.txt` files are not promoted to canonical
configs because the recovered scripts selected combinations on paths named
`test-1024`, which may leak test-set information.

Recommended layout:

```text
sam_vit_b/
├── coco.json
├── voc2012.json
└── ...
```
