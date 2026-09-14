# Third-Party, Data, Model, API, and License Disclosure

This disclosure is provided for the GOAI Open Exploration finalist package.
It distinguishes project-authored code/data from external software used to run
the environment.

## External data, models, and services

| Category | Disclosure |
| --- | --- |
| External datasets | **None.** No external scientific dataset is used to produce the committed results. |
| External trained models | **None.** The Gaussian processes are fitted locally from project-generated campaign/atlas data. |
| Commercial APIs | **None.** No commercial API is required for exploration, scoring, reproduction, or the dashboard. |
| Closed-source models | **None.** |
| Proprietary engine geometry or measurements | **None.** The chamber/injector representation is an abstract synthetic 2-D environment. |
| Generated artifacts | OpenFOAM mixing fields and derived JSON/JSONL/figures generated specifically for this project. |

## Project license

Project-authored source code is released under the repository `LICENSE`
(**MIT License**).

## Direct Python dependencies

The authoritative direct dependency list and minimum versions are in
`pyproject.toml`; the exact resolved environment, including transitive
packages, is in `uv.lock`.

| Dependency | Role in this project | Upstream license |
| --- | --- | --- |
| NumPy | numerical arrays / sampling calculations | BSD-3-Clause |
| SciPy | Latin hypercube utilities and scientific routines | BSD-3-Clause |
| scikit-learn | Gaussian-process surrogate models | BSD-3-Clause |
| Matplotlib | static figures | Matplotlib license (PSF-based, BSD-compatible) |
| Plotly.py | interactive plots | MIT |
| pandas | tabular artifact/dashboard handling | BSD-3-Clause |
| Streamlit | dashboard application | Apache-2.0 |
| Typer | command-line interface | MIT |
| pytest (development extra) | test runner | MIT |

## External solver and toolchain

| Software | Role | License |
| --- | --- | --- |
| OpenFOAM Foundation OpenFOAM 14 | live 2-D laminar CFD mixer | GNU GPL v3 |
| uv | environment/package manager used by reproduction commands | MIT OR Apache-2.0 |
| Python 3.11+ | runtime | PSF License |

Upstream license references:

- OpenFOAM: https://openfoam.org/licence/
- NumPy: https://github.com/numpy/numpy
- SciPy: https://github.com/scipy/scipy
- scikit-learn: https://github.com/scikit-learn/scikit-learn
- Matplotlib: https://github.com/matplotlib/matplotlib
- Plotly.py: https://github.com/plotly/plotly.py
- pandas: https://github.com/pandas-dev/pandas
- Streamlit: https://github.com/streamlit/streamlit
- Typer: https://github.com/fastapi/typer
- pytest: https://github.com/pytest-dev/pytest
- uv: https://github.com/astral-sh/uv

This file is a disclosure/index, not a replacement for upstream license texts.
Users who redistribute third-party software should follow the corresponding
upstream license terms.
