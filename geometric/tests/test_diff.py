"""
A set of tests for using the QCEngine project
"""

import copy
import numpy as np
import json, os, shutil

import geometric.coordinate_systems.cartesian
import geometric.coordinate_systems.delocalised
import geometric.coordinate_systems.primitive
from . import addons
import geometric
import pytest
import itertools

localizer = addons.in_folder
datad = addons.datad

def test_diff_h2o2_h2o(localizer):
    M = geometric.molecule.Molecule(os.path.join(datad, 'h2o2_h2o.pdb'))
    IC = geometric.coordinate_systems.delocalised.DelocalizedInternalCoordinates(M, build=True, connect=False, addcart=False)
    IC1 = geometric.coordinate_systems.primitive.PrimitiveInternalCoordinates(M, build=True, connect=False, addcart=False)
    IC2 = geometric.coordinate_systems.cartesian.CartesianCoordinates(M)
    IC3 = geometric.coordinate_systems.delocalised.DelocalizedInternalCoordinates(M, build=True, connect=True, addcart=False)
    assert IC1.repr_diff(IC) == "Primitive -> Delocalized"
    assert IC.repr_diff(IC1) == "Delocalized -> Primitive"
    assert IC.repr_diff(IC) == ""
    assert IC1.repr_diff(IC1) == ""
    assert len(IC.repr_diff(IC2).split('\n')) == 45
    assert len(IC.repr_diff(IC3).split('\n')) == 24
