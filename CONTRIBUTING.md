# Contributing

Contributions are welcome through focused pull requests.

Before submitting a change:

1. Install the development dependencies with `pip install -e ".[dev]"`.
2. Run `ruff check .` and `pytest`.
3. Add tests for changes to replacement semantics or result selection.
4. Do not commit model weights, datasets, private paths, credentials, or full
   search logs.
5. Document the exact upstream model revision when changing a backend.

Bug reports should include the command, model/checkpoint identity, manifest
hash, environment, and a minimal traceback.
