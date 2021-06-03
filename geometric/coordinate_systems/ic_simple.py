from abc import ABC
from collections import OrderedDict

import numpy as np

from geometric.molecule import Elements, Radii
from geometric.nifty import ang2bohr, bohr2ang, logger
from .internal_base import InternalCoordinateSystemBase
from .slots import (
    Angle,
    CartesianX,
    CartesianY,
    CartesianZ,
    Dihedral,
    Distance,
    LinearAngle,
    MultiAngle,
    MultiDihedral,
    OutOfPlane,
    RotationA,
    RotationB,
    RotationC,
    Rotator,
    TranslationX,
    TranslationY,
    TranslationZ,
)


class SimpleIC(InternalCoordinateSystemBase, ABC):
    """
    Simple internal coordinate system: internals are used directly
    """

    def __init__(self, molecule):
        super(SimpleIC, self).__init__(molecule)

        self.Internals = []
        self.Rotators = OrderedDict()
        self.elem = molecule.elem

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

    def __repr__(self):
        lines = ["Internal coordinate system (atoms numbered from 1):"]
        typedict = OrderedDict()
        for Internal in self.Internals:
            lines.append(Internal.__repr__())
            if str(type(Internal)) not in typedict:
                typedict[str(type(Internal))] = 1
            else:
                typedict[str(type(Internal))] += 1
        if len(lines) > 1000:
            # Print only summary if too many
            lines = []
        for k, v in typedict.items():
            lines.append("%s : %i" % (k, v))
        return "\n".join(lines)

    def __eq__(self, other):
        answer = True
        for i in self.Internals:
            if i not in other.Internals:
                answer = False
        for i in other.Internals:
            if i not in self.Internals:
                answer = False
        return answer

    def __ne__(self, other):
        return not self.__eq__(other)

    def guess_hessian(self, coords):
        """
        Build a guess Hessian that roughly follows Schlegel's guidelines.
        """
        xyzs = coords.reshape(-1, 3) * bohr2ang
        Hdiag = []

        def covalent(a, b):
            r = np.linalg.norm(xyzs[a] - xyzs[b])
            rcov = (
                Radii[Elements.index(self.elem[a]) - 1]
                + Radii[Elements.index(self.elem[b]) - 1]
            )
            return r / rcov < 1.2

        for ic in self.Internals:
            if type(ic) is Distance:
                r = np.linalg.norm(xyzs[ic.a] - xyzs[ic.b]) * ang2bohr
                elem1 = min(
                    Elements.index(self.elem[ic.a]), Elements.index(self.elem[ic.b])
                )
                elem2 = max(
                    Elements.index(self.elem[ic.a]), Elements.index(self.elem[ic.b])
                )
                A = 1.734
                if elem1 < 3:
                    if elem2 < 3:
                        B = -0.244
                    elif elem2 < 11:
                        B = 0.352
                    else:
                        B = 0.660
                elif elem1 < 11:
                    if elem2 < 11:
                        B = 1.085
                    else:
                        B = 1.522
                else:
                    B = 2.068
                if covalent(ic.a, ic.b):
                    Hdiag.append(A / (r - B) ** 3)
                else:
                    Hdiag.append(0.1)
            elif type(ic) in [Angle, LinearAngle, MultiAngle]:
                if type(ic) in [Angle, LinearAngle]:
                    a = ic.a
                    c = ic.c
                else:
                    a = ic.a[-1]
                    c = ic.c[0]
                if (
                    min(
                        Elements.index(self.elem[a]),
                        Elements.index(self.elem[ic.b]),
                        Elements.index(self.elem[c]),
                    )
                    < 3
                ):
                    A = 0.160
                else:
                    A = 0.250
                if covalent(a, ic.b) and covalent(ic.b, c):
                    Hdiag.append(A)
                else:
                    Hdiag.append(0.1)
            elif type(ic) in [Dihedral, MultiDihedral]:
                r = np.linalg.norm(xyzs[ic.b] - xyzs[ic.c])
                rcov = (
                    Radii[Elements.index(self.elem[ic.b]) - 1]
                    + Radii[Elements.index(self.elem[ic.c]) - 1]
                )
                # Hdiag.append(0.1)
                Hdiag.append(0.023)
                # The value recommended in Schlegel's paper does not appear to improve performance for larger systems.
                # Hdiag.append(max(0.0023, 0.0023-0.07*(r-rcov)))
            elif type(ic) is OutOfPlane:
                r1 = xyzs[ic.b] - xyzs[ic.a]
                r2 = xyzs[ic.c] - xyzs[ic.a]
                r3 = xyzs[ic.d] - xyzs[ic.a]
                d = 1 - np.abs(
                    np.dot(r1, np.cross(r2, r3))
                    / np.linalg.norm(r1)
                    / np.linalg.norm(r2)
                    / np.linalg.norm(r3)
                )
                # Hdiag.append(0.1)
                if (
                    covalent(ic.a, ic.b)
                    and covalent(ic.a, ic.c)
                    and covalent(ic.a, ic.d)
                ):
                    Hdiag.append(0.045)
                else:
                    Hdiag.append(0.023)
            elif type(ic) in [CartesianX, CartesianY, CartesianZ]:
                Hdiag.append(0.05)
            elif type(ic) in [TranslationX, TranslationY, TranslationZ]:
                Hdiag.append(0.05)
            elif type(ic) in [RotationA, RotationB, RotationC]:
                Hdiag.append(0.05)
            else:
                raise RuntimeError(
                    "Failed to build guess Hessian matrix. Make sure all IC types are supported"
                )
        return np.diag(Hdiag)

    def second_derivatives(self, xyz):
        self.calculate(xyz)
        answer = []
        for Internal in self.Internals:
            answer.append(Internal.second_derivative(xyz))
        # This array has dimensions:
        # 1) Number of internal coordinates
        # 2) Number of atoms
        # 3) 3
        # 4) Number of atoms
        # 5) 3
        return np.array(answer)

    def torsionConstraintLinearAngles(self, coords, thre=175):
        """
        Check if a torsion constrained optimization is about to fail
        because three consecutive atoms are nearly linear.
        """

        coords = coords.copy().reshape(-1, 3)

        def measure_angle_degrees(i, j, k):
            x1 = coords[i]
            x2 = coords[j]
            x3 = coords[k]
            v1 = x1 - x2
            v2 = x3 - x2
            n = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(n)
            return angle * 180 / np.pi

        linear_torsion_angles = {}
        for Internal in self.cPrims:
            if type(Internal) is Dihedral:
                a, b, c, d = Internal.a, Internal.b, Internal.c, Internal.d
                abc = measure_angle_degrees(a, b, c)
                bcd = measure_angle_degrees(b, c, d)
                if abc > thre:
                    linear_torsion_angles[(a, b, c)] = abc
                elif bcd > thre:
                    linear_torsion_angles[(b, c, d)] = bcd
        return linear_torsion_angles

    def linearRotCheck(self):
        # Check if certain problems might be happening due to rotations of linear molecules.
        for Internal in self.Internals:
            if type(Internal) is LinearAngle:
                if Internal.stored_dot2 > 0.75:
                    # Linear angle is almost parallel to reference axis
                    return True
            if type(Internal) in [RotationA, RotationB, RotationC]:
                if Internal in self.cPrims:
                    continue
                if Internal.Rotator.stored_dot2 > 0.9:
                    # Linear molecule is almost parallel to reference axis
                    return True
        return False

    def largeRots(self):
        for Internal in self.Internals:
            if type(Internal) in [RotationA, RotationB, RotationC]:
                if Internal in self.cPrims:
                    continue
                if Rotator.stored_norm > 0.9 * np.pi:
                    # # Molecule has rotated by almost pi
                    if type(Internal) is RotationA:
                        logger.info(
                            "Large rotation: %s = %.3f*pi\n"
                            % (str(Internal), Rotator.stored_norm / np.pi,)
                        )
                    return True
        return False

    def getRotatorNorms(self):
        rots = []
        for Internal in self.Internals:
            if type(Internal) in [RotationA]:
                rots.append(Rotator.stored_norm)
        return rots

    def printRotations(self, xyz):
        rotNorms = self.getRotatorNorms()
        if len(rotNorms) > 0:
            logger.info(
                "Rotator Norms: " + " ".join(["% .4f" % i for i in rotNorms]) + "\n"
            )
        rotDots = self.getRotatorDots()
        if len(rotDots) > 0 and np.max(rotDots) > 1e-5:
            logger.info(
                "Rotator Dots : " + " ".join(["% .4f" % i for i in rotDots]) + "\n"
            )
        linAngs = [ic.value(xyz) for ic in self.Internals if type(ic) is LinearAngle]
        if len(linAngs) > 0:
            logger.info(
                "Linear Angles: " + " ".join(["% .4f" % i for i in linAngs]) + "\n"
            )

    def calcDiff(self, xyz1, xyz2):
        """Calculate difference in internal coordinates (coord1-coord2), accounting for changes in 2*pi of angles."""
        answer = []
        for Internal in self.Internals:
            answer.append(Internal.calcDiff(xyz1, xyz2))
        return np.array(answer)

    def resetRotations(self, xyz):
        for Internal in self.Internals:
            if type(Internal) is LinearAngle:
                Internal.reset(xyz)
        for rot in self.Rotators.values():
            rot.reset(xyz)
