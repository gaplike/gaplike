# Installation

`gaplike` needs only `numpy` and `scipy`.

```bash
git clone https://github.com/gaplike/gaplike
cd gaplike
uv venv && uv pip install -e .
```

Plain `pip` works identically — there is nothing uv-specific in the package.

## Extras

| extra | pulls in | needed for |
|---|---|---|
| `pe` | `emcee`, `corner`, `matplotlib` | sampling, the notebooks, every paper figure |
| `dev` | `pytest` | the test suite |
| `docs` | `sphinx`, `furo`, `myst-parser`, `sphinx-copybutton` | building this documentation |
| `mbhb` | `lisabeta` | the MBHB waveform adapter only |

```bash
uv pip install -e ".[pe,dev]"
```

## The MBHB waveform adapter

[lisabeta](https://gitlab.in2p3.fr/marsat/lisabeta) is **not** pulled in by
`[pe]`. It is needed by exactly one factory,
{func}`gaplike.waveform.lisabeta_mbhb_ae`, and the import happens inside that
factory — so every likelihood, every gap builder, the conjugate-gradient
solver and all the chain-based paper figures work without it.

```bash
uv pip install -e ".[pe,mbhb]"
```

If you would rather build the GitLab sources than install from PyPI, note that
the build runs CMake and needs FFTW present first:

```bash
brew install fftw          # macOS;  apt install libfftw3-dev on Debian/Ubuntu
SKBUILD_CMAKE_DEFINE="FFTW_ROOT=$(brew --prefix fftw)" \
  uv pip install "lisabeta @ git+https://gitlab.in2p3.fr/marsat/lisabeta.git"
```

## Tests

```bash
uv pip install -e ".[dev]"
pytest
```

Around forty seconds. `tests/test_paper_regression.py` rebuilds the paper's
three gap scenarios from package primitives and reproduces the original
pipeline's windows, covariances, noise realization and every likelihood value
to about 1e-9; it is skipped when `tests/data/reference.json` is absent, and
its waveform fingerprint is skipped without lisabeta.

## Building the documentation

```bash
uv pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```
