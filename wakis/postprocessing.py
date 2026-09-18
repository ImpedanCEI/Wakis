import numpy as np
from scipy.constants import c as c_light

# unified trapezoidal integrator
try:
    trap = np.trapezoid
except AttributeError:
    trap = np.trapz


def compute_wake_factors(
    s=None,
    t=None,
    WP=None,
    WPx=None,
    WPy=None,
    lambdas=None,
    x_offset=None,
    y_offset=None,
    filename="wake_factors.txt",
):
    """
    Compute loss and kick factors and save results to a file.

    Outputs:
        - loss factor
        - kick factors (x, y)
        - offsets

    File is saved as plain text.
    """

    # ---------------------------
    # 1. Define longitudinal axis
    # ---------------------------
    if s is None:
        if t is None:
            raise ValueError("Either s or t must be provided")
        s = c_light * t

    s = np.asarray(s)

    # ---------------------------
    # 2. Checks
    # ---------------------------
    if WP is None or lambdas is None:
        raise ValueError("WP and lambdas are required")

    WP = np.asarray(WP)
    lambdas = np.asarray(lambdas)

    if not (len(s) == len(WP) == len(lambdas)):
        raise ValueError("s, WP, lambdas must have same length")

    # ---------------------------
    # 3. Charge
    # ---------------------------
    q = -trap(lambdas, s)

    if q == 0:
        raise ValueError("λ integrates to zero charge!")

    # ---------------------------
    # 4. Convert units
    # ---------------------------
    WP_SI = WP * 1e12  # V/pC → V/C

    # ---------------------------
    # 5. Loss factor
    # ---------------------------
    k_loss = abs (trap(WP_SI * lambdas, s) / q) *1e-12

    # ---------------------------
    # 6. Transverse kick factors
    # ---------------------------
    kx, ky = None, None

    if WPx is not None and x_offset not in (None, 0):
        WPx = np.asarray(WPx)
        WPx_SI = WPx * 1e12
        kx = abs (trap(WPx_SI * lambdas, s) / (q * x_offset)) *1e-12

    if WPy is not None and y_offset not in (None, 0):
        WPy = np.asarray(WPy)
        WPy_SI = WPy * 1e12
        ky = abs (trap(WPy_SI * lambdas, s) / (q * y_offset)) *1e-12

    # ---------------------------
    # 7. Save to file
    # ---------------------------
    with open(filename, "w") as f:
        f.write("Wake Factors Results\n")
        f.write("--------------------\n\n")

        f.write(f"Total charge q [C] = {q:.6e}\n\n")

        f.write(f"Loss factor [V/pC] = {k_loss:.6e}\n\n")
        f.write(f"Kick factor x [V/pC/m] = {kx}\n")
        f.write(f"Kick factor y [V/pC/m] = {ky}\n")
        f.write(f"x_offset [m] = {x_offset}\n")
        f.write(f"y_offset [m] = {y_offset}\n\n")

    return {
        "k_loss": k_loss,
        "kx": kx,
        "ky": ky,
        "q": q,
    }
