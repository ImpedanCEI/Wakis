import os
import sys

import numpy as np
from scipy.constants import c, mu_0
from tqdm import tqdm

sys.path.append("../")
import wakis

flag_interactive = False  # Set to true to run plot tests


class TestPML:
    def test_timestep(self):
        # TODO: check for dt < relaxation time
        pass

    def test_reflection_planewave(self):
        print("\n---------- Initializing simulation ------------------")
        # Domain bounds and grid
        xmin, xmax = -1.0, 1.0
        ymin, ymax = -1.0, 1.0
        zmin, zmax = 0.0, 1.0

        Nx, Ny = 20, 20
        Nz = 200

        grid = wakis.GridFIT3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz)

        # Boundary conditions and solver
        bc_low = ["periodic", "periodic", "pec"]
        bc_high = ["periodic", "periodic", "pml"]

        # Test different eps_r and sigma case
        eps_r = 1.0
        sigma = 0.0

        # Solver
        solver = wakis.SolverFIT3D(
            grid,
            use_stl=False,
            bg=[eps_r, 1.0, sigma],
            bc_low=bc_low,
            bc_high=bc_high,
            n_pml=30,
            dtype=np.float32,
        )

        # Source
        amplitude = 100.0
        nodes = 7
        f = 15 / ((solver.z.max() - solver.z.min()) / c)
        planeWave = wakis.sources.PlaneWave(
            xs=slice(0, Nx),
            ys=slice(0, Ny),
            zs=0,
            beta=1.0,
            amplitude=amplitude,
            f=f,
            nodes=nodes,
            phase=np.pi / 2,
        )
        solver.dt = 1 / f / 200  # ensure right amplitude

        # Simulation time is extended by a factor eps_r
        # to ensure the wave is fully absorbed in the PML

        z_pml = 0
        for k in range(solver.n_pml):
            z_pml += solver.dz[Nz - 1 - k]
        Nt = int(eps_r * 2.0 * (zmax - zmin - z_pml) / c / solver.dt)

        for n in tqdm(range(Nt)):
            planeWave.update(solver, n * solver.dt)
            solver.one_step()

            if flag_interactive and n % int(Nt / 100) == 0:
                solver.plot1D(
                    "Hy",
                    ylim=(-amplitude, amplitude),
                    pos=[0.5, 0.35, 0.2, 0.1],
                    off_screen=True,
                    title="005_Hy",
                    n=n,
                )
                solver.plot1D(
                    "Ex",
                    ylim=(-amplitude * c * mu_0, amplitude * c * mu_0),
                    pos=[0.5, 0.35, 0.2, 0.1],
                    off_screen=True,
                    title="005_Ex",
                    n=n,
                )

        reflectionH = solver.H.get_abs()[Nx // 2, Ny // 2, :].max()
        reflectionE = solver.E.get_abs()[Nx // 2, Ny // 2, :].max() / (mu_0 * c)
        assert reflectionH <= 10, (
            f"PML H reflection >10% with eps_r={eps_r}, sigma={sigma}"
        )
        assert reflectionE <= 10, (
            f"PML E reflection >10% with eps_r={eps_r}, sigma={sigma}"
        )

        if flag_interactive:
            os.system("convert -delay 10 -loop 0 005_Hy*.png 005_Hy_planewave.gif")
            os.system("convert -delay 10 -loop 0 005_Ex*.png 005_Ex_planewave.gif")
            os.system("rm 005_Hy*.png")
            os.system("rm 005_Ex*.png")

            solver.plot2D(
                "Ex",
                plane="ZX",
                pos=0.5,
                cmap="bwr",
                interpolation="spline36",
                n=n,
                vmin=-amplitude * c * mu_0,
                vmax=amplitude * c * mu_0,
                off_screen=True,
                title="005_Ex2d",
            )

            solver.plot2D(
                "Hy",
                plane="ZX",
                pos=0.5,
                cmap="bwr",
                interpolation="spline36",
                n=n,
                vmin=-amplitude,
                vmax=amplitude,
                off_screen=True,
                title="005_Hy2d",
            )

    def test_reflection_planewave_resistive_material(self):
        print("\n---------- Initializing simulation ------------------")
        # Domain bounds and grid
        xmin, xmax = -1.0, 1.0
        ymin, ymax = -1.0, 1.0
        zmin, zmax = 0.0, 1.0

        Nx, Ny = 20, 20
        Nz = 200

        grid = wakis.GridFIT3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz)

        # Boundary conditions and solver
        bc_low = ["periodic", "periodic", "pec"]
        bc_high = ["periodic", "periodic", "pml"]

        # Test different eps_r and sigma case
        eps_r = 3.0
        sigma = 1.0e-3

        # Solver
        solver = wakis.SolverFIT3D(
            grid,
            use_stl=False,
            bg=[eps_r, 1.0, sigma],
            bc_low=bc_low,
            bc_high=bc_high,
            dtype=np.float32,
            n_pml=30,
        )

        # Source
        amplitude = 100.0
        nodes = 7
        f = 15 / ((solver.z.max() - solver.z.min()) / c)
        planeWave = wakis.sources.PlaneWave(
            xs=slice(0, Nx),
            ys=slice(0, Ny),
            zs=0,
            beta=1.0,
            amplitude=amplitude,
            f=f,
            nodes=nodes,
            phase=np.pi / 2,
        )
        solver.dt = 1 / f / 200  # ensure right amplitude

        # Simulation

        # Simulation time is extended by a factor eps_r
        # to ensure the wave is fully absorbed in the PML
        z_pml = 0
        for k in range(solver.n_pml):
            z_pml += solver.dz[Nz - 1 - k]
        Nt = int(eps_r * 2.0 * (zmax - zmin - z_pml) / c / solver.dt)

        for n in tqdm(range(Nt)):
            planeWave.update(solver, n * solver.dt)
            solver.one_step()

            if flag_interactive and n % int(Nt / 100) == 0:
                solver.plot1D(
                    "Hy",
                    ylim=(-amplitude, amplitude),
                    pos=[0.5, 0.35, 0.2, 0.1],
                    off_screen=True,
                    title="005_Hy",
                    n=n,
                )
                solver.plot1D(
                    "Ex",
                    ylim=(-amplitude * c * mu_0, amplitude * c * mu_0),
                    pos=[0.5, 0.35, 0.2, 0.1],
                    off_screen=True,
                    title="005_Ex",
                    n=n,
                )

        reflectionH = solver.H.get_abs()[Nx // 2, Ny // 2, :].max()
        reflectionE = solver.E.get_abs()[Nx // 2, Ny // 2, :].max() / (mu_0 * c)
        assert reflectionH <= 10, (
            f"PML H reflection >10% with eps_r={eps_r}, sigma={sigma}"
        )
        assert reflectionE <= 10, (
            f"PML E reflection >10% with eps_r={eps_r}, sigma={sigma}"
        )

        if flag_interactive:
            # os.system(f'convert -delay 10 -loop 0 005_Hy*.png 005_Hy_planewave.gif')
            # os.system(f'convert -delay 10 -loop 0 005_Ex*.png 005_Ex_planewave.gif')
            # os.system(f'rm 005_Hy*.png')
            # os.system(f'rm 005_Ex*.png')

            solver.plot2D(
                "Ex",
                plane="ZX",
                pos=0.5,
                cmap="bwr",
                interpolation="spline36",
                n=n,
                vmin=-amplitude * c * mu_0,
                vmax=amplitude * c * mu_0,
                off_screen=True,
                title="005_Ex2d",
            )

            solver.plot2D(
                "Hy",
                plane="ZX",
                pos=0.5,
                cmap="bwr",
                interpolation="spline36",
                n=n,
                vmin=-amplitude,
                vmax=amplitude,
                off_screen=True,
                title="005_Hy2d",
            )

    def test_reflection_planewave_high_resistivity_material(self):
        print("\n---------- Initializing simulation ------------------")
        # Domain bounds and grid
        xmin, xmax = -1.0, 1.0
        ymin, ymax = -1.0, 1.0
        zmin, zmax = 0.0, 1.0

        Nx, Ny = 20, 20
        Nz = 200

        grid = wakis.GridFIT3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz)

        # Boundary conditions and solver
        bc_low = ["periodic", "periodic", "pec"]
        bc_high = ["periodic", "periodic", "pml"]

        # Test different eps_r and sigma case
        eps_r = 1.0
        sigma = 1

        # Solver
        solver = wakis.SolverFIT3D(
            grid,
            use_stl=False,
            bg=[eps_r, 1.0, sigma],
            bc_low=bc_low,
            bc_high=bc_high,
            dtype=np.float32,
            n_pml=30,
        )

        # Source
        amplitude = 100.0
        nodes = 7
        f = 15 / ((solver.z.max() - solver.z.min()) / c)
        planeWave = wakis.sources.PlaneWave(
            xs=slice(0, Nx),
            ys=slice(0, Ny),
            zs=0,
            beta=1.0,
            amplitude=amplitude,
            f=f,
            nodes=nodes,
            phase=np.pi / 2,
        )
        solver.dt = 1 / f / 200  # ensure right amplitude

        # Simulation

        # Simulation time is extended by a factor eps_r
        # to ensure the wave is fully absorbed in the PML
        z_pml = 0
        for k in range(solver.n_pml):
            z_pml += solver.dz[Nz - 1 - k]
        Nt = int(eps_r * 2.0 * (zmax - zmin - z_pml) / c / solver.dt)

        for n in tqdm(range(Nt)):
            planeWave.update(solver, n * solver.dt)
            solver.one_step()

            if flag_interactive and n % int(Nt / 100) == 0:
                solver.plot1D(
                    "Hy",
                    ylim=(-amplitude, amplitude),
                    pos=[0.5, 0.35, 0.2, 0.1],
                    off_screen=True,
                    title="005_Hy",
                    n=n,
                )
                solver.plot1D(
                    "Ex",
                    ylim=(-amplitude * c * mu_0, amplitude * c * mu_0),
                    pos=[0.5, 0.35, 0.2, 0.1],
                    off_screen=True,
                    title="005_Ex",
                    n=n,
                )

        reflectionH = solver.H.get_abs()[Nx // 2, Ny // 2, :].max()
        reflectionE = solver.E.get_abs()[Nx // 2, Ny // 2, :].max() / (mu_0 * c)
        assert reflectionH <= 10, (
            f"PML H reflection >10% with eps_r={eps_r}, sigma={sigma}"
        )
        assert reflectionE <= 10, (
            f"PML E reflection >10% with eps_r={eps_r}, sigma={sigma}"
        )

        if flag_interactive:
            # os.system(f'convert -delay 10 -loop 0 005_Hy*.png 005_Hy_planewave.gif')
            # os.system(f'convert -delay 10 -loop 0 005_Ex*.png 005_Ex_planewave.gif')
            # os.system(f'rm 005_Hy*.png')
            # os.system(f'rm 005_Ex*.png')

            solver.plot2D(
                "Ex",
                plane="ZX",
                pos=0.5,
                cmap="bwr",
                interpolation="spline36",
                n=n,
                vmin=-amplitude * c * mu_0,
                vmax=amplitude * c * mu_0,
                off_screen=True,
                title="005_Ex2d",
            )

            solver.plot2D(
                "Hy",
                plane="ZX",
                pos=0.5,
                cmap="bwr",
                interpolation="spline36",
                n=n,
                vmin=-amplitude,
                vmax=amplitude,
                off_screen=True,
                title="005_Hy2d",
            )

    def _pml_func():
        pml_lo = 0.005
        pml_hi = 0.1
        n_pml = 10

        lin = np.linspace(pml_lo, pml_hi, n_pml)
        geom = np.geomspace(pml_lo, pml_hi, n_pml)  # r=(pml_hi/pml_lo)**(1/(n_pml-1))

        x = np.linspace(0, 1, n_pml)
        quad = pml_lo + (pml_hi - pml_lo) * x**2
        cub = pml_lo + (pml_hi - pml_lo) * x**3
        quart = pml_lo + (pml_hi - pml_lo) * x**4
        quint = pml_lo + (pml_hi - pml_lo) * x**5

        # Plot func profiles
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        ax.plot(lin, label="linspace")
        ax.plot(geom, label="geomspace")
        ax.plot(quad, label="quadratic")
        ax.plot(cub, label="cubic")
        ax.plot(quart, label="quartic")
        ax.plot(quint, label="quintic")
        ax.set_xlabel("PML cells n")
        ax.set_ylabel(r"conductivity profile $\sigma(n)$")
        ax.legend()

        # plt.show()

        # Plot func derivative
        fig, ax = plt.subplots(dpi=200)

        ax.plot(lin[1:] - lin[:-1], label="linspace")
        ax.plot(geom[1:] - geom[:-1], label="geomspace")
        ax.plot(quad[1:] - quad[:-1], label="quadratic")
        ax.plot(cub[1:] - cub[:-1], label="cubic")
        ax.plot(quart[1:] - quart[:-1], label="quartic")
        ax.plot(quint[1:] - quint[:-1], label="quintic")
        ax.set_xlabel("PML cells n")
        ax.set_ylabel(r"gradient of conductivity profile $\sigma(n)$")
        ax.legend()
