# AI Coding Assistant Guidelines for Wakis

> **Physics first, simple implementations, meaningful validation, and documentation that enables others to understand and reproduce the results.**

These instructions apply to AI-assisted development in Wakis, independently of the coding assistant used.

Read `CONTRIBUTING.md` before making changes; it is the source of truth for contribution standards. Inspect nearby modules, tests, docs, examples, `.pre-commit-config.yaml`, and `.github/pull_request_template.md`. Match existing public APIs without copying legacy anti-patterns. Make the smallest maintainable change that fully solves the problem; minimize modified lines, files, and unrelated refactoring. Follow the [CERN ABP Clean Coding Guidelines](https://abpcomputing.web.cern.ch/guides/clean_coding/) alongside the Wakis-specific rules below.

## Implementations

- Before adding code, search for existing Wakis utilities and suitable functions in core dependencies (e.g. NumPy/SciPy). Reuse them rather than duplicating functionality.
- Prefer simple, readable numerical code and focused functions. Add only necessary lines, helpers, wrappers, classes, dependencies, and files; do not refactor unrelated code or maintain parallel implementations of the same feature. If existing code is too fragile or duplicated to extend safely, perform only the minimal refactoring necessary; suggest broader improvements separately.
- Preserve public API compatibility unless a breaking change is requested, but do not add speculative fallback paths, legacy compatibility layers, or deprecation machinery.
- Keep physics models, numerical operations, configuration, and visualization logically separated. Prefer explicit inputs and outputs over hidden global state; avoid unexpected side effects while respecting the intentionally stateful solver API.
- Follow PEP 8, Ruff formatting, and isort. Use `snake_case` for functions/variables, `PascalCase` for classes, and ALL_CAPS only for genuine constants (not adjustable simulation inputs). Preserve public names unless an API change is explicitly requested.
- Use clear guard clauses and `if`/`elif`/`else`; avoid needless nesting, repeated conditions, and `else` after unconditional `return`/`raise`.
- Validate inputs at appropriate API boundaries; trust internal invariants rather than repeatedly checking guaranteed states. Avoid speculative defensive checks and silent fallbacks that can hide numerical errors. Catch only specific, expected exceptions that can be handled or usefully contextualized; do not use `try`/`except` for normal control flow, silently suppress errors, or hide bugs behind broad `except` blocks.
- Keep normal execution quiet. Use established verbosity/logging conventions for optional diagnostics; emit concise, nonrepeating, actionable warnings for recoverable problems and raise exceptions for invalid states.
- Preserve physical units, signs, mesh/index conventions, field locations, update order, boundary treatment, numerical precision, and CPU/GPU/MPI behavior. Explain any deliberate change to them.
- Prioritize correctness, then tests, then performance. Optimize when profiling or benchmarks identify a meaningful bottleneck, with tests protecting numerical behavior. In hot paths, avoid unnecessary Python loops, repeated allocations/conversions, and printing or logging inside time-stepping loops; use efficient array/sparse operations where appropriate.

## Comments, documentation, and references

- Comment **why**, especially for equations, discrete approximations, boundary operations, indexing, and numerical tradeoffs. Avoid obvious narration.
- Update NumPy-style public docstrings when changing behavior: parameters, defaults, units, outputs, exceptions, assumptions, and side effects.
- New core numerical methods or physical formalisms require updates to `docs/physicsguide.md` with governing equations, definitions, discretization, assumptions, limits, and verifiable scientific citations. Distinguish literature-derived methods from Wakis adaptations; never invent sources.
- Update user guides, API documentation, and examples for new or changed features. Documentation is part of completion.

## Validation

- Prefer small deterministic integration/end-to-end tests through Wakis's public API over implementation-detail unit tests, signatures, mock call counts, or bare stencil checks.
- Compare with analytical cases, numerical convergence, invariants, independent benchmarks, or established regressions where feasible. Choose tolerances based on numerical/physical error, not arbitrary exact equality.
- Add regression coverage for bug fixes. Before creating new tests, inspect existing cases and extend or parametrize a relevant test when it can cover the new behavior clearly; minimize duplicate setup, runs, and overlapping assertions. Keep distinct tests when combining them would obscure failures.
- Organize test files into clear functional groups (e.g. longitudinal, transverse, or the public method being exercised). Use short, meaningful `test_...` names identifying the functionality, not a full description of the scenario. Explain the setup, physical assumptions, and expected behavior in concise test docstrings; keep names distinct enough to identify pytest failures.
- Plotting is generally noncritical: use lightweight smoke tests to check that plotting functions run without errors. Parametrize meaningful plotting branches and keyword arguments rather than inspecting cosmetic details or images, unless a specific visual regression requires it. Do not display figures during ordinary automated tests; show them only when `--interactive` or `--debug-plots` is enabled, as appropriate, and otherwise use off-screen/headless rendering and close figures.
- Use existing pytest markers for expensive or special-platform tests and report untested GPU/MPI paths honestly.

## Examples and notebooks

- Examples must **teach the public API**, not recreate substantial library functionality. Move reusable calculations/algorithms into Wakis; simple case setup and parameter sweeps are fine.
- Prefer one cohesive example with physical motivation, setup, execution, figures, expected behavior, and interpretation over multiple redundant scripts.
- Prefer `.ipynb` for narrative tutorials where practical, but respect existing `.py` examples. Make notebooks executable top-to-bottom, portable, lightweight, and compatible with documentation builds where used.
- Avoid excessive tracked generated output, binary PDFs, large data, caches, and unrelated artifacts; maintain `.gitignore` as needed.

## Before finishing

- Run relevant pytest cases and configured pre-commit checks when possible; consult `CONTRIBUTING.md` for local pytest commands and the optional `--gpu`, `--interactive`, and `--debug-plots` flags. Do not claim runs that did not happen.
- Write PR descriptions using `.github/pull_request_template.md`. Report motivation, key changes, theory/references, validation, documentation, limitations, and related issues; tick only completed checklist items.
- Give a brief final summary of changes, tests run and not run, documentation updates, and known limitations. Do not create extra Markdown reports or artifacts unless requested.
- If keeping the change narrowly scoped exposed a worthwhile architectural improvement, mention it briefly as an **optional follow-up**, explaining how it could improve readability or maintainability. Do not implement that refactor unless requested; avoid speculative suggestions when the existing design is adequate.
