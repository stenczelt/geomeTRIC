from abc import ABC
from collections import OrderedDict

import numpy as np

from geometric.molecule import Elements, Radii, Molecule
from geometric.nifty import ang2bohr, bohr2ang, logger
from .convert import convert_angstroms_degrees
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

    def __init__(self, molecule: Molecule):
        super(SimpleIC, self).__init__(molecule)

        self.Internals = []
        self.cPrims = []
        self.cVals = []
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

    def update(self, other):
        Changed = False
        for i in self.Internals:
            if i not in other.Internals:
                if hasattr(i, "inactive"):
                    i.inactive += 1
                else:
                    i.inactive = 0
                if i.inactive == 1:
                    logger.info("Deleting:" + str(i) + "\n")
                    self.Internals.remove(i)
                    Changed = True
            else:
                i.inactive = 0
        for i in other.Internals:
            if i not in self.Internals:
                logger.info("Adding:  " + str(i) + "\n")
                self.Internals.append(i)
                Changed = True
        return Changed

    def join(self, other):
        Changed = False
        for i in other.Internals:
            if i not in self.Internals:
                logger.info("Adding:  " + str(i) + "\n")
                self.Internals.append(i)
                Changed = True
        return Changed

    def repr_diff(self, other):
        if hasattr(other, "Prims"):
            output = ["Primitive -> Delocalized"]
            otherPrims = other.Prims
        else:
            output = []
            otherPrims = other
        alines = ["-- Added: --"]
        for i in otherPrims.Internals:
            if i not in self.Internals:
                alines.append(i.__repr__())
        dlines = ["-- Deleted: --"]
        for i in self.Internals:
            if i not in otherPrims.Internals:
                dlines.append(i.__repr__())
        if len(alines) > 1:
            output += alines
        if len(dlines) > 1:
            output += dlines
        return "\n".join(output)

    def addConstraint(self, cPrim, cVal=None, xyz=None):
        if cVal is None and xyz is None:
            raise RuntimeError("Please provide either cval or xyz")
        if cVal is None:
            # If coordinates are provided instead of a constraint value,
            # then calculate the constraint value from the positions.
            # If both are provided, then the coordinates are ignored.
            cVal = cPrim.value(xyz)
        if cPrim in self.cPrims:
            iPrim = self.cPrims.index(cPrim)
            if np.abs(cVal - self.cVals[iPrim]) > 1e-6:
                logger.info("Updating constraint value to %.4e\n" % cVal)
            self.cVals[iPrim] = cVal
        else:
            if cPrim not in self.Internals:
                self.Internals.append(cPrim)
            self.cPrims.append(cPrim)
            self.cVals.append(cVal)

    def getConstraints_from(self, other):
        if other.haveConstraints():
            for cPrim, cVal in zip(other.cPrims, other.cVals):
                self.addConstraint(cPrim, cVal)
        self.reorderPrimitives()

    def reorderPrimitives(self):
        # Reorder primitives to be in line with cc's code
        newPrims = []
        for cPrim in self.cPrims:
            newPrims.append(cPrim)
        for typ in [
            Distance,
            Angle,
            LinearAngle,
            MultiAngle,
            OutOfPlane,
            Dihedral,
            MultiDihedral,
            CartesianX,
            CartesianY,
            CartesianZ,
            TranslationX,
            TranslationY,
            TranslationZ,
            RotationA,
            RotationB,
            RotationC,
        ]:
            for p in self.Internals:
                if type(p) is typ and p not in self.cPrims:
                    newPrims.append(p)
        if len(newPrims) != len(self.Internals):
            raise RuntimeError(
                "Not all internal coordinates have been accounted for. You may need to add something to reorderPrimitives()"
            )
        self.Internals = newPrims

    def getConstraintNames(self):
        return [str(c) for c in self.cPrims]

    def getConstraintTargetVals(self, units=False):
        if units:
            return convert_angstroms_degrees(self.cPrims, self.cVals)
        else:
            return self.cVals

    def getConstraintCurrentVals(self, xyz, units=False):
        answer = []
        for ic, c in enumerate(self.cPrims):
            value = c.value(xyz)
            answer.append(value)
        if units:
            return convert_angstroms_degrees(self.cPrims, np.array(answer))
        else:
            return np.array(answer)

    def calcConstraintDiff(self, xyz, units=False):
        """Calculate difference between
        (constraint ICs evaluated at provided coordinates - constraint values).

        If units=True then the values will be returned in units of Angstrom and degrees
        for distance and angle degrees of freedom respectively.
        """
        cDiffs = np.zeros(len(self.cPrims), dtype=float)
        for ic, c in enumerate(self.cPrims):
            # Calculate the further change needed in this constrained variable
            if type(c) is RotationA:
                ca = c
                cb = self.cPrims[ic + 1]
                cc = self.cPrims[ic + 2]
                if type(cb) is not RotationB or type(cc) is not RotationC:
                    raise RuntimeError(
                        "In primitive internal coordinates, RotationA must be followed by RotationB and RotationC."
                    )
                if len(set([ca.w, cb.w, cc.w])) != 1:
                    raise RuntimeError(
                        "The triple of rotation ICs need to have the same weight."
                    )
                cDiffs[ic] = ca.calcDiff(xyz, val2=self.cVals[ic : ic + 3] / c.w)
                cDiffs[ic + 1] = cb.calcDiff(xyz, val2=self.cVals[ic : ic + 3] / c.w)
                cDiffs[ic + 2] = cc.calcDiff(xyz, val2=self.cVals[ic : ic + 3] / c.w)
            elif type(c) in [RotationB, RotationC]:
                pass
            else:
                cDiffs[ic] = c.calcDiff(xyz, val2=self.cVals[ic])
        if units:
            return convert_angstroms_degrees(self.cPrims, cDiffs)
        else:
            return cDiffs

    def maxConstraintViolation(self, xyz):
        cDiffs = self.calcConstraintDiff(xyz, units=True)
        return np.max(np.abs(cDiffs))

    def printConstraints(self, xyz, thre=1e-5):
        nc = len(self.cPrims)
        out_lines = []
        header = "Constraint                         Current      Target       Diff."
        curr = self.getConstraintCurrentVals(xyz, units=True)
        refs = self.getConstraintTargetVals(units=True)
        diff = self.calcConstraintDiff(xyz, units=True)
        for ic, c in enumerate(self.cPrims):
            if np.abs(diff[ic]) > thre:
                out_lines.append(
                    "%-30s  % 10.5f  % 10.5f  % 10.5f"
                    % (str(c), curr[ic], refs[ic], diff[ic])
                )
        if len(out_lines) > 0:
            logger.info(header + "\n")
            logger.info("\n".join(out_lines) + "\n")

    def haveConstraints(self):
        return len(self.cPrims) > 0

    def makeConstraints(self, molecule, constraints, cvals):
        # Add the list of constraints.
        xyz = molecule.xyzs[0].flatten() * ang2bohr
        if constraints is not None:
            if len(constraints) != len(cvals):
                raise RuntimeError(
                    "List of constraints should be same length as constraint values"
                )
            for cons, cval in zip(constraints, cvals):
                self.addConstraint(cons, cval, xyz)

    def delete(self, dof):
        for ii in range(len(self.Internals))[::-1]:
            if dof == self.Internals[ii]:
                del self.Internals[ii]

    def getRotatorDots(self):
        # fixme: seem not to be used at all
        dots = []
        for Internal in self.Internals:
            if type(Internal) in [RotationA]:
                dots.append(Rotator.stored_dot2)
        return dots
