from abc import ABC

import numpy as np
from numpy.linalg import multi_dot

from .ic_simple import SimpleIC
from .internal_base import InternalCoordinateSystemBase


class MixIC(InternalCoordinateSystemBase, ABC):
    """
    Mixing internal coordinate systems: Internals are linear combination of other internals

    The over-complete IC system is stored in self.Prims
    """

    # type hints
    Prims: SimpleIC

    def __init__(self, molecule):
        super(MixIC, self).__init__(molecule)

        self.Prims = None
        self.Vecs = None

    def add(self, dof):
        # for now we are not adding anything here, that is done when we are building the Prims
        pass

    def calculate(self, coords):
        """ Calculate the DLCs given the Cartesian coordinates. """
        PrimVals = self.Prims.calculate(coords)
        Answer = np.dot(PrimVals, self.Vecs)
        # To obtain the primitive coordinates from the delocalized internal coordinates,
        # simply multiply self.Vecs*Answer.T where Answer.T is the column vector of delocalized
        # internal coordinates. That means the "c's" in Equation 23 of Schlegel's review paper
        # are simply the rows of the Vecs matrix.
        # print np.dot(np.array(self.Vecs[0,:]).flatten(), np.array(Answer).flatten())
        # print PrimVals[0]
        # raw_input()
        return np.array(Answer).flatten()

    def derivatives(self, coords):
        """ Obtain the change of the DLCs with respect to the Cartesian coordinates. """
        PrimDers = self.Prims.derivatives(coords)
        # The following code does the same as "tensordot"
        # print PrimDers.shape
        # print self.Vecs.shape
        # Answer = np.zeros((self.Vecs.shape[1], PrimDers.shape[1], PrimDers.shape[2]), dtype=float)
        # for i in range(self.Vecs.shape[1]):
        #     for j in range(self.Vecs.shape[0]):
        #         Answer[i, :, :] += self.Vecs[j, i] * PrimDers[j, :, :]
        # print Answer.shape
        Answer1 = np.tensordot(self.Vecs, PrimDers, axes=(0, 0))
        return np.array(Answer1)

    def __repr__(self):
        return self.Prims.__repr__()

    def __eq__(self, other):
        return self.Prims == other.Prims

    def __ne__(self, other):
        return not self.__eq__(other)

    def second_derivatives(self, coords):
        """ Obtain the second derivatives of the DLCs with respect to the Cartesian coordinates. """
        PrimDers = self.Prims.second_derivatives(coords)
        Answer2 = np.tensordot(self.Vecs, PrimDers, axes=(0, 0))
        return np.array(Answer2)

    def torsionConstraintLinearAngles(self, coords, thre=175):
        """ Check if certain problems might be happening due to three consecutive atoms in a torsion angle becoming linear. """
        return self.Prims.torsionConstraintLinearAngles(coords, thre)

    def linearRotCheck(self):
        """ Check if certain problems might be happening due to rotations of linear molecules. """
        return self.Prims.linearRotCheck()

    def largeRots(self):
        """ Determine whether a molecule has rotated by an amount larger than some threshold (hardcoded in Prims.largeRots()). """
        return self.Prims.largeRots()

    def printRotations(self, xyz):
        return self.Prims.printRotations(xyz)

    def calcDiff(self, coord1, coord2):
        """ Calculate difference in internal coordinates (coord1-coord2), accounting for changes in 2*pi of angles. """
        PMDiff = self.Prims.calcDiff(coord1, coord2)
        Answer = np.dot(PMDiff, self.Vecs)
        return np.array(Answer).flatten()

    def resetRotations(self, xyz):
        """ Reset the reference geometries for calculating the orientational variables. """
        self.Prims.resetRotations(xyz)

    def repr_diff(self, other):
        if hasattr(other, "Prims"):
            return self.Prims.repr_diff(other.Prims)
        else:
            if self.Prims.repr_diff(other) == "":
                return "Delocalized -> Primitive"
            else:
                return "Delocalized -> Primitive\n" + self.Prims.repr_diff(other)

    def guess_hessian(self, coords):
        """ Build the guess Hessian, consisting of a diagonal matrix
        in the primitive space and changed to the basis of DLCs. """
        Hprim = self.Prims.guess_hessian(coords)
        return multi_dot([self.Vecs.T, Hprim, self.Vecs])

    def addConstraint(self, cPrim, cVal, xyz):
        self.Prims.addConstraint(cPrim, cVal, xyz)

    def getConstraints_from(self, other):
        self.Prims.getConstraints_from(other.Prims)

    def haveConstraints(self):
        return len(self.Prims.cPrims) > 0

    def getConstraintNames(self):
        return self.Prims.getConstraintNames()

    def getConstraintTargetVals(self, units=True):
        return self.Prims.getConstraintTargetVals(units=units)

    def getConstraintCurrentVals(self, xyz, units=True):
        return self.Prims.getConstraintCurrentVals(xyz, units=units)

    def calcConstraintDiff(self, xyz, units=False):
        return self.Prims.calcConstraintDiff(xyz, units=units)

    def maxConstraintViolation(self, xyz):
        return self.Prims.maxConstraintViolation(xyz)

    def printConstraints(self, xyz, thre=1e-5):
        self.Prims.printConstraints(xyz, thre=thre)

    def update(self, other):
        return self.Prims.update(other.Prims)

    def join(self, other):
        return self.Prims.join(other.Prims)

    def clearCache(self):
        super(MixIC, self).clearCache()
        self.Prims.clearCache()
