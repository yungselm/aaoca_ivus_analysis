# AAOCA IVUS Analysis

## Overview
This project is analysis arranged IVUS images from ["aaoca_compression_simulation"](https://github.com/yungselm/aaoca_compression_simulation) and pressure data from ["pressure_curve_processing"](https://github.com/AI-in-Cardiovascular-Medicine/AIVUS-CAA) and analysis pathophysiology behind lumen deformation.

## Structure
```plaintext
aaoca_ivus_analysis/
├── data/                   # Raw data files
├── src
│   ├── main.py
│   └── ...
└── .venv/                  # Virtual environment (ignored by git)
```

## Usage
This project uses `uv`, a modern Python package manager and virtual environment manager.

### Requirements

- Python 3.12 or higher
- uv (`pip install uv` or [official installer](https://docs.astral.sh/uv/getting-started/installation/))

### Installation

1. Clone the repository
2. Create and activate virtual environment in the directory:
```bash
uv venv .venv
# create a environment with specific python if several exist
uv venv .venv --python=3.12
```
```bash
# On Windows:
.\.venv\Scripts\activate  

# On Mac/Linux:
source .venv/bin/activate
```
3.  Install dependencies:
`uv pip install -e .`
4. Run the project: `streamlit run src/main.py`
