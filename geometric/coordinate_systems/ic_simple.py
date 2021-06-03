from abc import ABC

import numpy as np

from geometric.coordinate_systems.internal_base import InternalCoordinateSystemBase


class SimpleIC(InternalCoordinateSystemBase, ABC):
    """
    Simple internal coordinate system: internals are used directly
    """

    def __init__(self, molecule):
        super(SimpleIC, self).__init__(molecule)

        self.Internals = []

    def add(self, dof):
        if dof not in self.Internals:
            self.Internals.append(dof)

    def calculate(self, xyz):
        answer = []
        for Internal in self.Internals:
            answer.append(Internal.value(xyz))
        return np.array(answer)

    def derivatives(self, xyz):
        self.calculate(xyz)
        answer = []
        for Internal in self.Internals:
            answer.append(Internal.derivative(xyz))
        # This array has dimensions:
        # 1) Number of internal coordinates
        # 2) Number of atoms
        # 3) 3
        return np.array(answer)
