"""Small end-to-end propagation tests for waveform sources."""

import numpy as np
import pytest
from scipy.constants import c, mu_0

from wakis import GridFIT3D, SolverFIT3D
from wakis.sources import (
    Dipole,
    GaussianPacket,
    ModePacket,
    PlaneWave,
    Pulse,
    WavePacket,
)

SOURCE_Z = 8
PROBE_Z = 20


def _make_solver(transverse_boundary="periodic"):
    grid = GridFIT3D(
        -0.03,
        0.03,
        -0.03,
        0.03,
        0.0,
        0.12,
        7,
        7,
        40,
        verbose=0,
    )
    return SolverFIT3D(
        grid,
        bc_low=[transverse_boundary, transverse_boundary, "abc"],
        bc_high=[transverse_boundary, transverse_boundary, "abc"],
        source_type="direct",
        verbose=0,
    )


def _simulate(
    solver,
    source,
    injected_component,
    observed_component,
    companion_component,
    steps=60,
):
    cx = solver.Nx // 2
    cy = solver.Ny // 2
    injected_field, injected_axis = injected_component
    observed_field, observed_axis = observed_component
    companion_field, companion_axis = companion_component

    injected = np.empty(steps)
    observed = np.empty(steps)
    companion = np.empty(steps)
    profile_x = np.empty((steps, solver.Nx))
    profile_y = np.empty((steps, solver.Ny))

    for step in range(steps):
        time = step * solver.dt
        source.update(solver, time)
        injected[step] = getattr(solver, injected_field)[
            cx, cy, SOURCE_Z, injected_axis
        ]

        solver.one_step()
        field = getattr(solver, observed_field)
        observed[step] = field[cx, cy, PROBE_Z, observed_axis]
        companion[step] = getattr(solver, companion_field)[
            cx, cy, PROBE_Z, companion_axis
        ]
        profile_x[step] = np.array(field[:, cy, PROBE_Z, observed_axis], copy=True)
        profile_y[step] = np.array(field[cx, :, PROBE_Z, observed_axis], copy=True)

    return {
        "time": np.arange(steps) * solver.dt,
        "injected": injected,
        "observed": observed,
        "companion": companion,
        "profile_x": profile_x,
        "profile_y": profile_y,
    }


def _first_active(values):
    values = np.abs(values)
    return np.flatnonzero(values > 1e-4 * values.max())[0]


def _assert_causal_propagation(result):
    assert np.max(np.abs(result["injected"])) > 0.0
    assert np.max(np.abs(result["observed"])) > 1e-8
    assert _first_active(result["observed"]) > _first_active(result["injected"])


def _plot_result(plot_source_simulation, result, title):
    plot_source_simulation(
        result["time"],
        {
            "Injection plane": result["injected"],
            "Downstream field": result["observed"],
            "Companion field": result["companion"],
        },
        title,
    )


def test_pulse_propagates_a_transverse_harris_wave(plot_source_simulation):
    solver = _make_solver()
    source = Pulse(
        field="Ex",
        xs=slice(None),
        ys=slice(None),
        zs=SOURCE_Z,
        shape="harris",
        L=8 * c * solver.dt,
    )

    result = _simulate(solver, source, ("E", "x"), ("E", "x"), ("H", "y"))

    _assert_causal_propagation(result)
    assert np.max(np.abs(result["companion"])) > 1e-5
    assert np.min(result["injected"]) >= 0.0
    assert np.max(result["injected"]) == pytest.approx(1.0)
    _plot_result(plot_source_simulation, result, "Pulse")


def test_dipole_radiates_an_oscillating_localized_field(plot_source_simulation):
    solver = _make_solver()
    source = Dipole(
        field="Ex",
        xs=solver.Nx // 2,
        ys=solver.Ny // 2,
        zs=SOURCE_Z,
        f=8e9,
    )

    result = _simulate(
        solver,
        source,
        ("E", "x"),
        ("E", "x"),
        ("H", "y"),
        steps=80,
    )

    _assert_causal_propagation(result)
    assert result["observed"].min() < 0.0 < result["observed"].max()
    assert np.max(np.abs(result["companion"])) > 1e-5
    peak = np.argmax(np.abs(result["observed"]))
    assert abs(result["profile_x"][peak, solver.Nx // 2]) > 2 * abs(
        result["profile_x"][peak, 0]
    )
    _plot_result(plot_source_simulation, result, "Dipole")


def test_gaussian_packet_preserves_its_sign_and_transverse_peak(
    plot_source_simulation,
):
    solver = _make_solver()
    source = GaussianPacket(
        zs=SOURCE_Z,
        sigmaz=4 * c * solver.dt,
        sigmaxy=0.02,
        tinj=12 * c * solver.dt,
    )

    result = _simulate(solver, source, ("H", "y"), ("E", "x"), ("H", "y"))

    _assert_causal_propagation(result)
    assert np.max(result["injected"]) <= 0.0
    peak = np.argmax(np.abs(result["observed"]))
    assert result["observed"][peak] * result["companion"][peak] > 0.0
    assert abs(result["profile_x"][peak, solver.Nx // 2]) > 1.1 * abs(
        result["profile_x"][peak, 0]
    )
    _plot_result(plot_source_simulation, result, "GaussianPacket")


def test_plane_wave_stays_uniform_and_impedance_matched(plot_source_simulation):
    solver = _make_solver()
    source = PlaneWave(zs=SOURCE_Z, f=5e9, nodes=1)

    result = _simulate(solver, source, ("H", "y"), ("E", "x"), ("H", "y"))

    _assert_causal_propagation(result)
    peak = np.argmax(np.abs(result["observed"]))
    profile = result["profile_x"][peak]
    np.testing.assert_allclose(profile, profile[0], rtol=1e-12, atol=1e-12)
    assert result["observed"][peak] * result["companion"][peak] > 0.0
    assert result["observed"][peak] / result["companion"][peak] == pytest.approx(
        mu_0 * c, rel=0.25
    )
    _plot_result(plot_source_simulation, result, "PlaneWave")


def test_wave_packet_propagates_a_finite_matched_gaussian(
    plot_source_simulation,
):
    solver = _make_solver()
    source = WavePacket(
        zs=SOURCE_Z,
        sigmaz=4 * c * solver.dt,
        sigmaxy=0.02,
        tinj=12 * c * solver.dt,
        f=5e9,
    )

    result = _simulate(solver, source, ("H", "y"), ("E", "x"), ("H", "y"))

    _assert_causal_propagation(result)
    peak = np.argmax(np.abs(result["observed"]))
    assert result["observed"][peak] * result["companion"][peak] > 0.0
    assert abs(result["profile_x"][peak, solver.Nx // 2]) > 1.4 * abs(
        result["profile_x"][peak, 0]
    )
    assert result["observed"][peak] / result["companion"][peak] == pytest.approx(
        mu_0 * c, rel=0.4
    )
    _plot_result(plot_source_simulation, result, "WavePacket")


def test_mode_packet_propagates_the_te01_profile(plot_source_simulation):
    solver = _make_solver(transverse_boundary="pec")
    source = ModePacket(
        zs=SOURCE_Z,
        f=5e9,
        sigma_t=4 * solver.dt,
        tinj=12 * solver.dt,
    )

    result = _simulate(solver, source, ("E", "x"), ("E", "x"), ("H", "y"))

    _assert_causal_propagation(result)
    assert np.max(np.abs(result["companion"])) > 1e-5
    peak = np.argmax(np.abs(result["observed"]))
    profile = result["profile_y"][peak]
    assert profile[0] == pytest.approx(0.0, abs=1e-12)
    assert profile[-1] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(profile, profile[::-1], rtol=1e-12, atol=1e-12)
    assert abs(profile[solver.Ny // 2]) == np.max(np.abs(profile))
    _plot_result(plot_source_simulation, result, "ModePacket TE01")
