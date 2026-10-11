# Initial publication record - 2026-10-11

## Included material

The initial source snapshot consists of seven Python files, one historical Markdown debugging note, and one PDF reflection. Their exact sizes and SHA-256 hashes are recorded in [source_manifest.json](source_manifest.json). All nine source files were copied without changes.

Repository packaging adds an English README, data setup instructions, dependency versions, Git ignore rules, this record, the source manifest, and the captured synthetic test output. No new model implementation, financial experiment, or data release was produced.

## Verification

Command, run at the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s codes -p test_jpx_debug.py -v
```

Result: **27 tests passed**, with no failures or errors. See [test_output.txt](test_output.txt).

Environment: macOS, Python 3.14.2, NumPy 2.4.6, pandas 3.0.3, scikit-learn 1.8.0, PyTorch 2.12.0. Tests use small synthetic CPU examples and do not require the market CSVs. This was a run in an existing environment, not a fresh dependency installation or cross-platform verification.

The source test file contains historical descriptions, including a window-one design note. Its passing status should not be interpreted as validating every setting in the current 42-row experiment.

Text source files were checked for common credential patterns; none were detected. The PDF text and metadata were checked for obvious credentials and private information. These are bounded publication checks, not a comprehensive security audit.

## Historical evidence and current limitations

The debugging note describes an earlier implementation and reports an older failing test count. It is retained as historical material; the fresh test output above describes this snapshot.

The PDF contains experimenter-supplied observations and explicitly does not represent a training rerun or source-code verification. Its 10-epoch description and reference architecture are not established by the current code: the epoch argument is unused, and a 42-row convolution leaves one time step for the LSTM. Both training and evaluation currently use the same dataset, with full-input standardization. No out-of-sample performance claim is made.

Full-data training, market-data semantics, backtesting, and reproduction of the PDF's numerical observations were not performed for this upload.

## Exclusions and preservation

Raw competition CSVs (approximately 1.1 GiB in the supplied folder), Python bytecode caches, and macOS metadata are excluded from Git. The original source folder and raw data were retained unchanged. No historical release was overwritten or deleted, and no new large-data release is needed for this source-only publication.
