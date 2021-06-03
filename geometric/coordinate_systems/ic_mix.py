from abc import ABC

import numpy as np

from geometric.coordinate_systems.internal_base import InternalCoordinateSystemBase
from geometric.coordinate_systems.ic_simple import SimpleIC


class MixIC(InternalCoordinateSystemBase, ABC):
    """
    Mixing internal coordinate systems: Internals are linear combination of other internals

    The over-complete IC system is stored in self.Prims
    """

    def __init__(self, molecule):
        super(MixIC, self).__init__(molecule)

        self.Prims: SimpleIC = None

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
