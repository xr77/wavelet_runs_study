# Contributing

Use Python 3.10–3.12 and an isolated environment with `pip install -e ".[dev]"`.

```bash
pytest
ruff check src tests examples tools
ruff format --check src tests examples tools
python tools/check_release.py
```

Use explicit imports, descriptive snake_case names, short functions, and documented array axes. Package code belongs in `src/wavelet_runs`, runnable examples in `examples`, and tests in `tests`. Test numerical changes, run splitting, file I/O, and invalid inputs. Document scientific assumptions.

Use generated test inputs only. Do not commit notebooks, scans, masks, participant information, conditions, outputs, or private backups. Use synthetic examples in issues and pull requests. Statistical inference changes need explicit methods and review.
