# Code-only releases

Version 0.2.0 is published at https://doi.org/10.5281/zenodo.23228383. Its 26,856-byte source ZIP has SHA-256 `48121126a96e8b3172201d684b6d23c988697c983876b3cf16230c9527e075d5`; the Zenodo MD5 checksum is `b4a596713146d66368518be491a3c18f`. The release tag is immutable; the main branch's citation metadata were updated after DOI publication.

Publish only source code, configuration, tests, examples, and documentation. Never include laboratory data, notebook outputs, saved results, or private backups. Public Git history must also be clean: deleting files from the latest commit does not remove historical copies.

1. Update versions in `pyproject.toml`, `src/wavelet_runs/__init__.py`, and `CITATION.cff`.
2. Run `pytest`, `ruff check src tests examples tools`, and `ruff format --check src tests examples tools`.
3. Stage intended files and run `python tools/check_release.py`. Review source for hard-coded research results.
4. Build with `python -m build` and verify installation from the wheel in a fresh environment.
5. Export the reviewed commit with `git archive`. Run `python tools/check_release.py --archive /path/to/source.zip` before upload.
6. Tag that commit on GitHub. Create a new Zenodo version, upload only the verified archive, and enter matching author/version/license metadata.
7. Verify the published record, archive, and resolving DOI, then update the citation.

The 0.1.0 archive was restricted after the owner clarified that lab-derived outputs must not be distributed. Do not restore its public access. Its original Git history must remain private.

Official Zenodo guidance: [manage files](https://help.zenodo.org/docs/deposit/manage-files/) and [new versions](https://help.zenodo.org/docs/deposit/manage-versions/).
