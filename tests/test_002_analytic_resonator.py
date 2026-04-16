import sys

import numpy as np
import pytest
from scipy.constants import c as c_light
from tqdm import tqdm

sys.path.append("../../wakis")
from wakis import GridFIT3D, SolverFIT3D


def test_mode_101():
    m = 1
    n = 0
    p = 1
    theta = 0

    # Analytic solution of cubic resonator
    # Ref: http://faculty.pccu.edu.tw/~meng/new%20EM6.pdf pp.20/24

    def analytic_sol_Hz(x, y, z, t):
        Rm = np.array(
            [
                [np.cos(-theta), -np.sin(-theta), 0],
                [np.sin(-theta), np.cos(-theta), 0],
                [0, 0, 1],
            ]
        )
        [x_0, y_0, z_0] = np.dot(Rm, np.array([x, y, z]))

        return (
            np.cos(m * np.pi / Lx * (x_0 - Lx / 2))
            * np.cos(n * np.pi / Ly * (y_0 - Ly / 2))
            * np.sin(p * np.pi / Lz * (z_0 - Lz / 2))
            * np.cos(np.sqrt(2) * np.pi / Lx * c_light * t)
        )

    def analytic_sol_Hy(x, y, z, t):
        Rm = np.array(
            [
                [np.cos(-theta), -np.sin(-theta), 0],
                [np.sin(-theta), np.cos(-theta), 0],
                [0, 0, 1],
            ]
        )
        [x_0, y_0, z_0] = np.dot(Rm, np.array([x, y, z]))
        h_2 = (m * np.pi / Lx) ** 2 + (n * np.pi / Ly) ** 2 + (p * np.pi / Lz) ** 2

        return (
            -2
            / h_2
            * (n * np.pi / Ly)
            * (p * np.pi / Lz)
            * np.cos(m * np.pi / Lx * (x_0 - Lx / 2))
            * np.sin(n * np.pi / Ly * (y_0 - Ly / 2))
            * np.cos(p * np.pi / Lz * (z_0 - Lz / 2))
            * np.cos(np.sqrt(2) * np.pi / Lx * c_light * t)
        )

    def analytic_sol_Hx(x, y, z, t):
        Rm = np.array(
            [
                [np.cos(-theta), -np.sin(-theta), 0],
                [np.sin(-theta), np.cos(-theta), 0],
                [0, 0, 1],
            ]
        )
        [x_0, y_0, z_0] = np.dot(Rm, np.array([x, y, z]))
        h_2 = (m * np.pi / Lx) ** 2 + (n * np.pi / Ly) ** 2 + (p * np.pi / Lz) ** 2

        return (
            -2
            / h_2
            * (m * np.pi / Lx)
            * (p * np.pi / Lz)
            * np.sin(m * np.pi / Lx * (x_0 - Lx / 2))
            * np.cos(n * np.pi / Ly * (y_0 - Ly / 2))
            * np.cos(p * np.pi / Lz * (z_0 - Lz / 2))
            * np.cos(np.sqrt(2) * np.pi / Lx * c_light * t)
        )

    # ---- Domain definition ----#
    L = 1.0  # Domain length
    N = 30  # Number of mesh cells

    Nx = N
    Ny = N
    Nz = N
    Lx = L
    Ly = L
    Lz = L
    dx = L / Nx
    dy = L / Ny
    dz = L / Nz

    xmin = -Lx / 2 + dx / 2
    xmax = Lx / 2 + dx / 2
    ymin = -Ly / 2 + dy / 2
    ymax = Ly / 2 + dy / 2
    zmin = -Lz / 2 + dz / 2
    zmax = Lz / 2 + dz / 2

    print("\n---------- Initializing simulation ------------------")
    grid = GridFIT3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz)
    solver = SolverFIT3D(
        grid, bc_low=["pec", "pec", "pec"], bc_high=["pec", "pec", "pec"]
    )

    # ---- Initial conditions ------------#
    for ii in range(Nx):
        for jj in range(Ny):
            for kk in range(Nz):
                x = ii * dx + xmin
                y = jj * dy + ymin
                z = kk * dz + zmin
                solver.H[ii, jj, kk, "z"] = analytic_sol_Hz(x, y, z, -0.5 * solver.dt)

                x = ii * dx + xmin
                y = jj * dy + ymin
                z = kk * dz + zmin
                solver.H[ii, jj, kk, "y"] = analytic_sol_Hy(x, y, z, -0.5 * solver.dt)

                x = ii * dx + xmin
                y = jj * dy + ymin
                z = kk * dz + zmin
                solver.H[ii, jj, kk, "x"] = analytic_sol_Hx(x, y, z, -0.5 * solver.dt)

    # ----- Time loop -----#
    # global Ey_000
    Ey_000 = []
    for nt in tqdm(range(5000)):
        Ey_000.append(solver.E[Nx // 2, Ny // 2, Nz // 2, "y"])
        solver.one_step()

    # ------ Resonator frequency -------#
    Ey_000 = np.array(Ey_000)
    FEy_000 = np.fft.fftshift(np.fft.fft(Ey_000))[len(Ey_000) // 2 :]
    freqs = np.fft.fftshift(np.fft.fftfreq(len(Ey_000), d=solver.dt))[
        len(Ey_000) // 2 :
    ]
    resonator_f = freqs[np.argmax(FEy_000)]

    # analytic frequency: https://learnemc.com/ext/calculators/cavity_resonance/rect-res.html
    assert resonator_f == pytest.approx(2.121e8, 0.05), "Ey frequency error >5%"
