# Contributing

Keep changes focused on the library's small backend-neutral operation surface.

1. Create a branch.
2. Add or update tests for behavior changes.
3. Run `pytest`.
4. Run `ruff check .`.
5. Open a pull request explaining the behavior being added or changed.

When adding an operation, implement it consistently in both `NumpyMatOps` and
`TorchMatOps` whenever the backends provide equivalent semantics.
