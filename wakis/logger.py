# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2025.                   #
# ########################################### #


import json
import os

import numpy as np
from scipy.constants import mu_0 as mu_0


class Logger:
    """Simple structured logger for grid, solver and wakeSolver metadata.

    The Logger stores small metadata dictionaries for different components
    of a simulation (``grid``, ``solver``, ``wakeSolver``) and can persist
    them into a human-readable log file inside a results folder.

    Attributes
    ----------
    grid : dict
        Arbitrary metadata about the generated grid (sizes, bounds, files).
    solver : dict
        Metadata produced by the numerical solver (tolerances, timings).
    wakeSolver : dict
        Metadata produced by the wake solver and global simulation settings.
    """

    def __init__(self):
        """Create a fresh empty logger with separate sections.

        The three dictionaries are empty on construction and intended to be
        populated by the application before calling :meth:`save_logs`.
        """
        self.grid = {}
        self.solver = {}
        self.wakeSolver = {}

    def save_logs(self, results_folder=None):
        """Persist logger sections into a JSON-like log file.

        The method writes the contents of ``wakeSolver``, ``solver`` and
        ``grid`` into a UTF-8 text file named ``wakis.log`` inside the
        provided ``results_folder``. Non-serializable objects are converted
        to strings to ensure the file remains readable.

        Parameters
        ----------
        results_folder : str, optional
            Output folder where ``wakis.log`` will be created. If provided
            it is stored into ``self.wakeSolver['results_folder']``. The
            folder is created if it does not exist.
        """
        if results_folder is not None:
            self.wakeSolver["results_folder"] = results_folder

        logfile = os.path.join(self.wakeSolver["results_folder"], "wakis.log")

        # Write sections
        if not os.path.exists(self.wakeSolver["results_folder"]):
            os.mkdir(self.wakeSolver["results_folder"])

        with open(logfile, "w", encoding="utf-8") as fh:
            fh.write("Simulation Parameters\n")
            fh.write("""=====================\n\n""")

            sections = [
                ("WakeSolver Logs", self.wakeSolver),
                ("Solver Logs", self.solver),
                ("Grid Logs", self.grid),
            ]

            for title, data in sections:
                fh.write(f"\n## {title} ##\n")
                if not data:
                    fh.write("(empty)\n")
                    continue

                # convert non-serializable values to strings recursively
                def _convert(obj):
                    if isinstance(obj, dict):
                        return {k: _convert(v) for k, v in obj.items()}
                    if isinstance(obj, (list, tuple)):
                        return [_convert(v) for v in obj]
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    try:
                        json.dumps(obj)
                        return obj
                    except Exception:
                        return str(obj)

                clean = _convert(data)
                fh.write(json.dumps(clean, indent=2, ensure_ascii=False))
                fh.write("\n")
