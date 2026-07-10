# Contributing to NeuroSim

Thank you for your interest in NeuroSim. This project is developed as part of
GSoC 2026 under INCF mentorship at the National Brain Research Centre (NBRC).

> **External contributions are welcome from 20th July 2026 onwards**, once the GSoC
> core architecture is stable. Until then, the codebase is under active
> development. Please do not open unsolicited PRs against `main`.

---

## Before You Contribute

Open a discussion first. Do not open a PR cold.

**GitHub Discussions** is the right place for all pre-contribution conversation:
https://github.com/TheRoy666/NeuroSim/discussions

- **Bug reports / questions** → Q&A category
- **Feature proposals / collaboration** → Ideas category
- **Showing work built on NeuroSim** → Show and tell category

Wait for a response from the maintainer before writing any code.

---

## Development Setup

```bash
git clone https://github.com/TheRoy666/NeuroSim
cd NeuroSim

# Core only
pip install -e ".[dev]"

# With neuroimaging data loading
pip install -e ".[neuroimaging]"

# Everything
pip install -e ".[all]"
```

---

## Running Tests

```bash
# Full suite (141 tests)
PYTHONPATH=. pytest tests/ -v --tb=short

# Single module
pytest tests/test_physics.py -v
```

CI runs on Python 3.9, 3.10, and 3.11. All tests must pass before a PR
will be reviewed.

---

## Code Style

```bash
black neurosim/ tests/
isort neurosim/ tests/
```

---

## Standards for New Contributions

NeuroSim is a mathematically rigorous library. Every new algorithm must include:

1. **Docstring with full mathematical derivation** — Equations in NumPy/LaTeX
   notation, not prose descriptions.
2. **Unit tests** — Correctness verified against a known reference
   implementation. Minimum coverage: shape, dtype, boundary conditions,
   and at least one scientific property (e.g. PSD, symmetry, non-negativity).
3. **Complexity annotation** — State time and space complexity explicitly in
   the docstring.
4. **Literature reference** — Author, year, journal. No algorithm without a
   citation.

---

## Reporting Bugs

Open a Q&A discussion at https://github.com/TheRoy666/NeuroSim/discussions
with:

- Python version and OS
- Minimal reproducible example
- Expected vs. actual output

If the bug is confirmed, the maintainer will convert it to a GitHub Issue.

---

## Maintainer and Mentor

**Ritam Kanti Roy** — Maintainer
GSoC 2026 Contributor, INCF Project #39

**Dr. Khushbu Agarwal** — Mentor
Computational Neuroscience Laboratory, NBRC

All editorial decisions rest with the maintainer and mentor.
PRs are merged at the maintainer's and mentor's discretion regardless of review status.
