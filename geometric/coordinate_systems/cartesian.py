import numpy as np

from .ic_simple import SimpleIC
from .slots import CartesianX, CartesianY, CartesianZ
from geometric.molecule import Molecule


class CartesianCoordinates(SimpleIC):
    """
    Cartesian coordinate system, written as a kind of internal coordinate class.
    This one does not support constraints, because that requires adding some
    primitive internal coordinates.
    """

    def __init__(self, molecule: Molecule, **kwargs):
        super(CartesianCoordinates, self).__init__(molecule)
        self.Internals = []
        self.cPrims = []
        self.cVals = []
        self.elem = molecule.elem
        for i in range(molecule.na):
            self.add(CartesianX(i, w=1.0))
            self.add(CartesianY(i, w=1.0))
            self.add(CartesianZ(i, w=1.0))
        if kwargs.get("remove_tr", False):
            raise RuntimeError("Do not use remove_tr with Cartesian coordinates")
        if "constraints" in kwargs and kwargs["constraints"] is not None:
            raise RuntimeError("Do not use constraints with Cartesian coordinates")

    def guess_hessian(self, xyz):
        return 0.5 * np.eye(len(xyz.flatten()))

    def addConstraint(self, cPrim=None, cVal=None, xyz=None):
        raise NotImplementedError(
            "Constraints not supported with Cartesian coordinates"
        )

    def haveConstraints(self):
        raise NotImplementedError(
            "Constraints not supported with Cartesian coordinates"
        )

    def makeConstraints(self, molecule, constraints, cvals):
        raise NotImplementedError(
            "Constraints not supported with Cartesian coordinates"
        )
