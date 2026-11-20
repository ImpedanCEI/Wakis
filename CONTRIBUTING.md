# Contributing to Wakis

Thank you for your interest in contributing to Wakis!

Wakis is an open-source electromagnetic simulation code developed at CERN. We welcome contributions from the accelerator physics and computational electromagnetics communities, including new numerical methods, performance improvements, bug fixes, documentation, and examples.

Our development philosophy is:

> **Physics first, simple implementations, meaningful validation, and documentation that enables others to understand and reproduce the results.**

We aim to develop scientifically reliable software that is accessible, maintainable, and transparent, encouraging collaboration and reproducibility. Our practices are informed by the [CERN ABP Clean Coding Guidelines](https://abpcomputing.web.cern.ch/guides/clean_coding/), with Wakis-specific expectations for numerical validation and scientific documentation.

Suggestions, feature requests, and bug reports can be submitted through [GitHub Issues](https://github.com/ImpedanCEI/Wakis/issues). Code contributions should be submitted through [Pull Requests](https://github.com/ImpedanCEI/Wakis/pulls), preferably linked to an existing issue to provide context and facilitate discussion.

## Scientific and numerical correctness

- Explain the physical problem and the motivation for a new method before implementing it.
- Preserve and clearly identify units, sign conventions, discrete field locations, update order, boundary conditions, and applicability limits.
- For new formulations, provide governing equations, approximations, numerical discretization or algorithm, and supporting scientific references. Distinguish published methods from Wakis-specific modifications.
- Prioritize correctness, then tests, then performance. Profile or benchmark before performance-driven refactoring, and verify that optimization preserves numerical behavior and reproducibility.

## Code quality

- Prefer the smallest maintainable change, not merely the fewest modified lines. If existing code is fragile or duplicated, make the minimal refactoring needed to extend it safely; keep broader refactors separate.
- Follow [PEP 8](https://peps.python.org/pep-0008/) and the formatting/import conventions in `.pre-commit-config.yaml` (Ruff formatter and isort). Keep changes focused; avoid unrelated reformatting.
- Write readable functions with clear responsibilities and minimal unnecessary abstractions. Prefer explicit inputs and outputs, minimal unexpected side effects, and clear separation of physical models, numerics, configuration, and plotting; avoid hidden global state. Prefer straightforward `if`/`elif`/`else` logic, guard clauses where useful, and specific exceptions. Avoid unnecessary nesting, `else` after an unconditional return, broad `try`/`except` blocks, silent fallbacks, and exceptions used for ordinary control flow.
- Use descriptive `snake_case` names for functions and variables and `PascalCase` for classes. Reserve uppercase names for genuine constants, not ordinary simulation parameters. Preserve established public API names for compatibility.
- Keep library output appropriately quiet: reserve warnings for actionable, recoverable problems and exceptions for invalid inputs or unrecoverable states. Avoid repeated progress prints in numerical loops.
- Add comments where they clarify the *reason* behind a formula, algorithm, or design choice. Maintain NumPy-style docstrings for public functionality, including units and assumptions.
- Consider effects on supported CPU, GPU, and MPI execution paths when modifying the solver.

## Tests and validation

Every new feature and bug fix should have a reproducible test. Prefer **behavioral, integration, and end-to-end tests** that exercise realistic simulations using the public API, rather than tests coupled to internal implementation details.

- Validate against analytical solutions, conservation properties, limiting cases, converged solutions, or independent simulation/measurement benchmarks when possible.
- Use physically justified numerical tolerances and small deterministic cases for CI; mark expensive tests consistently with the existing pytest configuration.
- Avoid tests that only check function signatures, mock calls, or discretization stencils without verifying observable numerical correctness. Targeted low-level tests remain useful when an invariant is otherwise difficult to exercise.
- Include regression coverage for fixed bugs. Report which backends and configurations were actually tested.

The test suite runs automatically when a pull request is opened or updated. You can also run tests locally after installing pytest (and Wakis with its required dependencies):

```bash
python -m pip install pytest
python -m pytest -v -s
```

Run a specific test file with `python -m pytest tests/<test_file>.py`. Wakis also defines these optional pytest flags in `tests/conftest.py`:

- `--gpu`: enable GPU execution in supported tests (requires a working GPU/CuPy environment).
- `--interactive`: enable on-screen PyVista rendering instead of off-screen rendering (requires a display).
- `--debug-plots`: display simulation-versus-reference comparison plots for regression checks.

For example, `python -m pytest -v -s --debug-plots` displays available comparison plots. Flags can be combined; use the defaults for ordinary automated runs.

## Documentation and examples

Documentation is part of the feature, not a follow-up task.

- Update the relevant user/API documentation whenever user-facing behavior changes.
- Add new physical formalisms and substantial numerical methods to `docs/physicsguide.md`, with equations, notation, assumptions, numerical details, limitations, and **verifiable references**. A notebook alone is not sufficient for a new core method.
- Demonstrate new functionality through the **public API**; do not hide essential algorithms or reusable calculations inside examples. Move them into the library instead.
- Prefer one cohesive, reproducible tutorial with motivation, setup, execution, plots, and interpretation over many small overlapping examples. Notebooks (`.ipynb`) are preferred for narrative tutorials when appropriate; existing `.py` examples remain valid for runnable scripts and specific workflows.
- Make examples portable, modest in runtime/data size, and executable from a clean environment. Ensure notebook-based documentation continues to build correctly where applicable.
- Do not commit generated PDFs, large result files, caches, or other derived binary artifacts merely to show output. Use tracked Markdown/notebook sources and `.gitignore` entries as appropriate.

## AI-assisted development

Generative AI coding assistants may be used to support development and contributions. When using these tools, follow the additional instructions in [AGENTS.md](AGENTS.md), which complement this contribution guide.

All contributions, whether AI-assisted or not, must meet the same standards for scientific correctness, code quality, testing, and documentation.

## Contribution workflow

1. Open or find an issue for significant changes and discuss the intended physical/numerical approach when useful.
2. Fork the repository or create a focused branch; implement the change with corresponding tests and documentation.
3. Run relevant tests and formatting hooks (for example, `pre-commit run --all-files` when configured). Keep commits and diffs reviewable.
4. Open a pull request using [the repository template](.github/pull_request_template.md). Describe the problem, solution, scientific references, validation results, documentation updates, limitations, and related issues. Mark checklist items accurately.
5. Respond to review feedback collaboratively. Review evaluates scientific validity, maintainability, and usability—not only whether the code executes.

For details on using Wakis, see the [documentation](https://wakis.readthedocs.io/) and [README](README.md). Please follow the repository's [code of conduct](CODE_OF_CONDUCT.md).
