# Contributing to NeuroSim

Thank you for your interest in contributing to NeuroSim! This project is developed
as part of GSoC 2026 under the INCF mentorship program.

## Development Setup

```bash
git clone https://github.com/TheRoy666/neurosim
cd neurosim
pip install -e ".[dev]"
pip install -r requirements.txt
```

## Running Tests

```bash
pytest tests/ -v --tb=short
```

## Code Style

We use `black` for formatting and `isort` for import ordering:

```bash
black neurosim/ tests/
isort neurosim/ tests/
```

## Mathematical Contributions

NeuroSim is a mathematically rigorous library. Any new algorithm must be accompanied by:

1. **Docstring with full mathematical derivation** (equations in NumPy/LaTeX notation).
2. **Unit tests** verifying correctness against a known reference implementation.
3. **Complexity annotation** — state the time and space complexity explicitly.
4. **Reference** to the primary literature (author, year, journal).

## Reporting Bugs

Please open a GitHub Issue with:
- Python version and OS
- Minimal reproducible example
- Expected vs. actual output

## Neuroscience Domain Questions

For questions about NCT methodology, connectomics, or clinical applications,
open a Discussion thread with the `neuroscience` label.
