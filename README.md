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
3.  Install dependencies:
`uv pip install -e .`
4. Run the project: `streamlit run src/main.py`

## Running Tests
This project includes unit tests for the core logic (data loading, filtering, and aggregation) implemented in the DataManager class.
1. Make sure the virtual environment is activated:


   ```bash
   .\.venv\Scripts\activate  # On Windows

   source .venv/bin/activate  # On Mac/Linux
   ```

2. Then run the tests using pytest:

   ```bash
   pytest src/tests -v
   ```
All tests should pass if the dataset is available in the data/ folder.

