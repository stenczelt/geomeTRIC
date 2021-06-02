import numpy as np

from geometric.coordinate_systems.slots import (
    CartesianX,
    CartesianY,
    CartesianZ,
    Distance,
    LinearAngle,
    TranslationX,
    TranslationY,
    TranslationZ,
)
from geometric.nifty import bohr2ang


def convert_angstroms_degrees(prims, values):
    """ Convert values of primitive ICs (or differences) from
    weighted atomic units to Angstroms and degrees. """
    converted = np.array(values).copy()
    for ic, c in enumerate(prims):
        if type(c) in [TranslationX, TranslationY, TranslationZ]:
            w = 1.0
        elif hasattr(c, "w"):
            w = c.w
        else:
            w = 1.0
        if type(c) in [
            TranslationX,
            TranslationY,
            TranslationZ,
            CartesianX,
            CartesianY,
            CartesianZ,
            Distance,
            LinearAngle,
        ]:
            factor = bohr2ang
        elif c.isAngular:
            factor = 180.0 / np.pi
        converted[ic] /= w
        converted[ic] *= factor
    return converted
