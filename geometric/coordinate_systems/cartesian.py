import numpy as np

from geometric.coordinate_systems.slots import CartesianX, CartesianY, CartesianZ
from geometric.internal import PrimitiveInternalCoordinates


class CartesianCoordinates(PrimitiveInternalCoordinates):
    """
    Cartesian coordinate system, written as a kind of internal coordinate class.
    This one does not support constraints, because that requires adding some
    primitive internal coordinates.
    """
    def __init__(self, molecule, **kwargs):
        super(CartesianCoordinates, self).__init__(molecule)
        self.Internals = []
        self.cPrims = []
        self.cVals = []
        self.elem = molecule.elem
        for i in range(molecule.na):
            self.add(CartesianX(i, w=1.0))
            self.add(CartesianY(i, w=1.0))
            self.add(CartesianZ(i, w=1.0))
        if kwargs.get('remove_tr', False):
            raise RuntimeError('Do not use remove_tr with Cartesian coordinates')
        if 'constraints' in kwargs and kwargs['constraints'] is not None:
            raise RuntimeError('Do not use constraints with Cartesian coordinates')

    def guess_hessian(self, xyz):
        return 0.5*np.eye(len(xyz.flatten()))