# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""
Material library dictionary

Format (non-conductive):
{
    'material key' : [eps_r, mu_r],
}

Format (conductive):
{
    'material key' : [eps_r, mu_r, sigma[S/m]],
}

! Note:
* 'material key' in lower case only
* eps = eps_r*eps_0 and mu = mu_r*mu_0
"""

import numpy as np

# fmt: off
material_lib = {
    'pec' : [np.inf, 1.],

    'vacuum' : [1.0, 1.0],

    'dielectric' : [10., 1.0],

    'lossy metal' : [10, 1.0, 10],

    'copper' : [5.8e+07, 1.0, 5.8e+07],

    'berillium' : [2.5e+07, 1.0, 2.5e+07],
}

material_colors = {
    'pec' : 'white',

    'vacuum' : 'tab:blue',

    'dielectric' : 'tab:green',

    'lossy metal' : 'tab:orange',

    'copper' : [0.82745099, 0.698039, 0.49019599],

    'berillium' : [0.82745099, 0.698039, 0.49019599],

    'stainless steel' : 'silver',

    'martensite' : [0.,1.,1.],

    'other' : 'cyan',
}
# fmt: on
