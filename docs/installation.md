# Installation

`gaplike` needs only `numpy` and `scipy`.

```bash
pip install gaplike
```

That is everything the package itself needs: all the likelihoods, the gap
builders, the windowed covariances and the matrix-free conjugate-gradient
solver. Install from a clone instead only if you also want the paper
reproduction pipeline in `paper/`, the notebooks, or the test suite:

```bash
git clone https://github.com/gaplike/gaplike
cd gaplike && pip install -e ".[pe,dev]"
```

`uv pip` can be substituted for `pip` throughout — there is nothing
uv-specific in the package. Whichever you choose, install and run from the
*same* environment; mixing the two is the commonest way to lose an afternoon
here.

## Extras

| extra | pulls in | needed for |
|---|---|---|
| `pe` | `emcee`, `corner`, `matplotlib` | sampling, the notebooks, every paper figure |
| `dev` | `pytest` | the test suite |
| `docs` | `sphinx`, `furo`, `myst-parser`, `sphinx-copybutton` | building this documentation |
| `mbhb` | `lisabeta` | the MBHB waveform adapter only |

```bash
pip install "gaplike[pe]"       # from PyPI
pip install -e ".[pe,dev]"      # from a clone
```

## The MBHB waveform adapter

[lisabeta](https://gitlab.in2p3.fr/marsat/lisabeta) is **not** pulled in by
`[pe]`. It is needed by exactly one factory,
{func}`gaplike.waveform.lisabeta_mbhb_ae`, and the import happens inside that
factory — so every likelihood, every gap builder, the conjugate-gradient
solver and all the chain-based paper figures work without it.

```bash
pip install "gaplike[pe,mbhb]"
```

Installing lisabeta from PyPI gives wheels. If you would rather build the
GitLab sources, the build runs CMake and needs FFTW present first:

```bash
brew install fftw          # macOS;  apt install libfftw3-dev on Debian/Ubuntu
SKBUILD_CMAKE_DEFINE="FFTW_ROOT=$(brew --prefix fftw)" \
  pip install "lisabeta @ git+https://gitlab.in2p3.fr/marsat/lisabeta.git"
```

## Tests

These need a clone, since the test data lives in `tests/`.

```bash
pip install -e ".[dev]"
pytest
```

Around forty seconds. `tests/test_paper_regression.py` rebuilds the paper's
three gap scenarios from package primitives and reproduces the original
pipeline's windows, covariances, noise realization and every likelihood value
to about 1e-9; it is skipped when `tests/data/reference.json` is absent, and
its waveform fingerprint is skipped without lisabeta.

## Building the documentation

Also from a clone.

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```
