# 📣 Releases
Release notes of every `Wakis` packaged version in [PyPI](https://pypi.org/project/wakis/), also available in [Zenodo](https://zenodo.org/records/15527405) and [Github Releases](https://github.com/ImpedanCEI/wakis/releases).

```{contents}
:depth: 2
```

## Wakis v0.6.0

### 🚀 New Features

* 🖼️ **Plotting**
  * Unified plotting tools for both MPI and non-MPI simulations.
  * `plot1D` now supports field visualization independently of MPI use.
  * `Plot2D` supports parallel execution.
  * Error handling added for `plot3D` and `plot3DonSTL` when `use_mpi = True`.
  * Support for `dpi` and `return_handles` in plot utilities to further customize plots.
  * Enhanced examples and notebook suite visualization cells.

* 🧱 **GridFIT3D**
  * Added `mpi_initialize()` to handle domain decomposition (Z-slicing).
  * New method `mpi_gather_asGrid()` to retrieve the full global grid from distributed subdomains.
  * Full support for multi-GPU domain decomposition through `cupy` (CUDA-aware, Linux only).
  * Improved communication layer for subdomain synchronization via ghost cells.

* ⚡ **SolverFIT3D**
  * MPI-compatible time-stepping routine `mpi_one_step()` using a leapfrog scheme.
  * `mpi_communicate()` to send/receive boundary field values between subdomains.
  * `mpi_gather()` to retrieve field data as a NumPy array and `mpi_gather_asField()` to reconstruct a `Field` object.
  * MPI-safe support integrated into `update()` and field getter logic.
  * Introduced `save_state()` method for checkpointing during MPI runs together with `load_state()`. Added support for MPI save state too.
  * Feature in progress: multiGPU support (`use_GPU=True` when `use_MPI=True`) for distributed simulations (*Linux only!*).
  * Added missing docstrings (Numpy-style)

* 📥 **Sources**
  * **Beam**:
    * Added `plot(t)` to visualize beam current evolution.
    * Generalized `update()` to work with or without MPI.
    * Enhanced support for time-aligned injection with beta and MPI shifts.
    * New example for MPI+GPU simulation (*topic in progress*).

* 🌊 **WakeSolver**
  * Refactored to internally store the full longitudinal domain.
  * `skip_cells` now acts only at analysis level, preserving resolution.
    * `add_space` and `use_edt` retained for compatibility, but `add_space` is deprecated for new parameter `skip_cells`, more adequate to its utility (i.e. skip cells in the integration path).
  * Future-ready structure for distributed wake solving with MPI-aware GPU.
  * Improved numerical robustness by preventing indexing errors in `WakePotential` integration -> solves [issue #12](https://github.com/ImpedanCEI/wakis/issues/12)
  * Enhanced extrapolation method with `iddefix`.
  * New example `004` for wakefield simulation with MPI+GPU configuration.

* 🛡️ **Security & Documentation**
  * Added `SECURITY.md` to describe supported versions and vulnerability reporting.
  * Improved installation guide with Miniforge (supports both Windows/Linux) and MPI setup instructions.
  * Added new issue templates for **Bug Report** and **Feature Request** with markdown formatting and emojis for readability.
  * Addition of the **Physics Guide**, with detailed physics models and numerical methods explanations.
  * User's guide updated to include Wake extrapolation with `iddefix`, Wake function calculation with `neffint`, and power loss calculation with `BIHC`.
  * Added a **Table of Contents (ToC)** to the documentation for easier navigation.
  * Expanded installation guide with multiGPU configuration and MPI-aware domain partitioning.

### 💗 Other Tag highlights

* 🔁 Nightly tests with GitHub Actions:
  * Enabled infrastructure for MPI-based test cases (`test_003`, `test_005`).
  * Improved test coverage for MPI and GPU simulations.
  * MultiGPU end-to-end tests for distributed domain synchronization.

* 📁 **Examples**:
  * notebook  `003` → MPI wakefield simulation using `mpi4py`.
  * notebook  `005` → Full MPI simulation inside Jupyter using `ipyparallel` + `mpi4py`.
  * New example `003` with MPI + GPU configuration for large-scale simulations.

* 📁 **Notebooks**:
  * `005` → Full MPI simulation inside Jupyter using `ipyparallel` + `mpi4py`.
  * New Jupyter notebook showcasing multiGPU configuration.

### 🐛 **Bugfixes**
* Fixed crash in `plot3D` and `plot3DonSTL` when `use_mpi=True`.
* Fixed default `use_mpi=True` to now default to `False` for general usage.
* Fixed a typo in beam injection routine `solver.z.min()`.
* Fixed potential rounding error in wake potential integration with negligible performance impact (~0.1ns) -> solves [issue #12](https://github.com/ImpedanCEI/wakis/issues/12)
* Corrected default beam injection time to align with CST Wakefield Solver reference in beta<1 cases.
* Fixed minor typos in the documentation.
* Fixed synchronization issues with MPI runs when saving states.
* Resolved encoding issues when installing in Windows editable mode.
* Corrected result folder naming in GPU example `002`.

### 👋👩‍💻 **New Contributors**
* [**@mctfr**](https://github.com/mctfr) – [PR #11](https://github.com/ImpedanCEI/wakis/pull/11) Contributed improvements to installation instructions in the documentation.

### 📝 **Full changelog**

|   **78 commits**   | 📚 Docs | 🧪 Tests | 🐛 Fixes | 🎨 Style | ✨ Features | Other |
|----------------|---------|----------|-----------|------------|--------------| ----- |
| % of Commits   | 30.3%   | 10.5%    | 9.2%     | 10.5%      | 22.4%        |  17.1     |

**Full Changelog**: https://github.com/ImpedanCEI/wakis/compare/v0.5.1...v0.6.0

----

## Wakis v0.5.1
> Minor fixes and updates

### 🚀 New Features
* Plotting
  * Allow passing camera position to solver's 3D plots `plot3D` and `plor3DnSTL`

### 💗 Other Tag highlights
* 🔁 Nightly tests with GitHub actions:
  * 003 -> coverage for grid `inspect` and `plot_solid`
* 📁 Examples:
  * 003 -> MPI wakefield simulation with `mpi4py`
* 📁 Notebooks:
  * 005 -> MPI example in jupyter notebooks with `ipyparallel`+ `mpi4py`

### 🐛 Bugfixes
* `__version__` now matches PyPI release and tag
* `gridFIT3D.plot_solids()` fix typo in the opacity assignment
* `example/001` fixed stl_solids parameter in grid.inspect() call

### 📝Full changelog
**Full Changelog**: https://github.com/ImpedanCEI/wakis/compare/v0.5.0...v0.5.1

-----

## Wakis v0.5.0

### 🚀 New Features
* 🧱 Geometry import:
    * Functions to read `.STP` files, exporting each solid into an `.STL` file indicating the name and material: `wakis.geometry.generate_stl_solids_from_stp(stp_file)`
    * Functions to extract from `.STP` files `solid` names, `colors`, and `materials`: `wakis.geometry.extract_XXX(stp_file)` to easily build the input dictionaries needed for `GridFIT3D`

* ⚡Solver:
    * New maximal timestep calculation for high-conductive regions based on CFL + relaxation time criterion
    * New methods: `save_state()`, `load_state()` to export and import the fields at a particular simulation timestep (HDF5 format). Method `reset_fields()` to clear fields before restarting a simulation.
    * Perfect Matching Layers (PML) boundary conditions: First version out!

* 🖼️ Plotting:
    * `solver.plot3DonSTL` Field on STL solid using `pyvista.sample` interpolation algorithm
        * Interactive plane clipping on `plot3DonSTL`
        * Field shown on clipping plane
    * `grid.plot_solids()` 3D plot with the imported solids and the position in the simulation bounding box when `bounding_box=True`

* 📥Sources:
    * Add `plot(t)` method to plot the source over the simulation time `t`
    * Custom amplitude as an attribute `self.amplitude`
    * Custom phase as an attribute `self.phase`
    * Custom injection time `self.tinj`
    * For `PlaneWave` allow for truncation at specific number of `self.nodes` injected

* 🌱 Ecosytem:
    * Wake extrapolation of partially decayed wakes coupling with [`IDDEFIX`]: https://github.com/ImpedanCEI/IDDEFIX:
        * IDDEFIX is a physics-informed machine learning framework that fits a resonator-based model (parameterized by R, f, Q) to wakefield simulation data using Evolutionary Algorithms. It leverages Differential Evolution to optimize these parameters, enabling efficient classification and extrapolation of electromagnetic wakefield behavior. This allows for reduced simulation time while maintaining long-term accuracy, akin to time-series forecasting in machine learning

    * Impedance to wake function conversion using non-equidistant Fourier transform with: [`neffint]: https://github.com/ImpedanCEI/neffint
        * Neffint is an acronym for Non-equidistant Filon Fourier integration. This is a python package for computing Fourier integrals using a method based on Filon's rule with non-equidistant grid spacing.

    * Beam-induced power loss calculations for different beam shapes and filling schemes using ['BIHC`]: https://github.com/ImpedanCEI/BIHC
        * Beam Induced Heating Computation (BIHC) tool is a package that allows the estimation of the dissipated power due to the passage of a particle beam inside an accelerator component. The dissipated power value depends on the characteristics of the particle beam (beam spectrum and intensity) and on the characteristics of the considered accelerator component (beam-coupling impedance).

### 💗 Other Tag highlights
* 🔁 Nightly tests with GitHub actions: 000 - 005
* 📁  notebooks: containing interactive examples
* 📁  examples: major cleanup, examples on CPU and GPU

### 🐛 Bugfixes
* Patch representation when a list is passed in `Plot2D`
* `ipympl` added as a dependency to `wakis['notebook']` installation
* Injection time to account for relativistic beta in sources

### 👋👩‍💻New Contributors
* @MaltheRaschke made their first contribution in https://github.com/ImpedanCEI/wakis/pull/4

**Full Changelog**: https://github.com/ImpedanCEI/wakis/compare/v0.4.0...v0.5.0
