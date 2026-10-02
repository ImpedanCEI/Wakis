---
sd_hide_title: true
---
# 🔎 Overview

## Welcome to `wakis` documentation

> **wakis**: 3D Time-domain **Wak**e and **I**mpedance **S**olver

`wakis` is a **3D Time-domain Electromagnetic solver** that solves the Integral form of Maxwell's equations using the Finite Integration Technique (FIT) numerical method. It computes the longitudinal and transverse **wake potential and beam-coupling impedance** from the simulated electric and magnetic fields. It is hence focused on simulations for particle accelerator components, but it is also a multi-purpose solver; capable of simulating planewaves interaction with nano-structures, optical diffraction, and much more!

🚀 Some of `wakis` features:

* Wake potential and impedance calculations for particle beams with different relativistic $\beta$
* Material tensors: permittivity $\varepsilon$, permeability $\mu$, conductivity $\sigma$. Possibility of anisotropy.
* CAD geometry importer (`STL` & `STEP` format) for embedded boundaries and material regions, based on [`pyvista`](https://github.com/pyvista/pyvista), with sub-cell smoothing
* Boundary conditions: PEC, PMC, periodic, first-order Mur ABC (`abc`), PML, and CPML
* Different time-domain sources: particle beam, planewave, gaussian wavepacket
* 100% python, fully exposed API (material tensors, fields $E$, $H$, $J$). Matrix operators based on `numpy` and `scipy.sparse` routines ensure fast calculations.
* 1d, 2d, 3d built-in plotting on-the-fly
* Optimized memory consumption
* GPU acceleration using `cupy/cupyx`
* MPI domain decomposition in the $z$ direction with `mpi4py`

🧩 Other complementary tools in the ecosystem:
* Wakefield extrapolation with PIML [`iddefix`](https://github.com/ImpedanCEI/IDDEFIX) evolutionary algorithms
* Non-equidistant Filon Fourier integration with [`neffint`](https://github.com/ImpedanCEI/neffint)
* Beam-induced heating estimation due to impedance with [`bihc`](https://github.com/ImpedanCEI/BIHC)


The source code is available in the `wakis` [GitHub repository](https://github.com/ImpedanCEI/wakis).

```{toctree}
:caption: Table of Contents
:maxdepth: 2

installation.md
usersguide.md
physicsguide.md
releases.md
wakis_api.rst
```
