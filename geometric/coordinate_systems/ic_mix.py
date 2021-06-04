from abc import ABC

import numpy as np
from numpy.linalg import multi_dot

from geometric.nifty import logger
from geometric.molecule import Molecule
from .ic_simple import SimpleIC
from .internal_base import InternalCoordinateSystemBase
from .slots import (
    RotationA,
    RotationB,
    RotationC,
    TranslationX,
    TranslationY,
    TranslationZ,
)


class MixIC(InternalCoordinateSystemBase, ABC):
    """
    Mixing internal coordinate systems: Internals are linear combination of other internals

    The over-complete IC system is stored in self.Prims
    """

    # type hints
    Prims: SimpleIC

    def __init__(self, molecule: Molecule):
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

    def remove_TR(self, xyz):
        """
        Project overall translation and rotation out of the DLCs.
        This feature is intended to be used when an optimization job appears
        to contain slow rotations of the whole system, which sometimes happens.
        Uses the same logic as build_dlc_1.
        """
        # Create three translation and three rotation primitive ICs for the whole system
        na = int(len(xyz) / 3)
        alla = range(na)
        sel = xyz.reshape(-1, 3).copy()
        TRPrims = []
        TRPrims.append(TranslationX(alla, w=np.ones(na) / na))
        TRPrims.append(TranslationY(alla, w=np.ones(na) / na))
        TRPrims.append(TranslationZ(alla, w=np.ones(na) / na))
        sel -= np.mean(sel, axis=0)
        rg = np.sqrt(np.mean(np.sum(sel ** 2, axis=1)))
        TRPrims.append(RotationA(alla, xyz, self.Prims.Rotators, w=rg))
        TRPrims.append(RotationB(alla, xyz, self.Prims.Rotators, w=rg))
        TRPrims.append(RotationC(alla, xyz, self.Prims.Rotators, w=rg))
        # If these primitive ICs are already there, then move them to the front
        primorder = []
        addPrims = []
        for prim in TRPrims:
            if prim in self.Prims.Internals:
                primorder.append(self.Prims.Internals.index(prim))
            else:
                addPrims.append(prim)
        for iprim, prim in enumerate(self.Prims.Internals):
            if prim not in TRPrims:
                primorder.append(iprim)
        self.Prims.Internals = addPrims + [self.Prims.Internals[p] for p in primorder]
        self.Vecs = np.vstack(
            (
                np.zeros((len(addPrims), self.Vecs.shape[1]), dtype=float),
                self.Vecs[np.array(primorder), :],
            )
        )

        self.clearCache()
        # Build DLCs with six extra in the front corresponding to the overall translations and rotations
        subVecs = self.Vecs.copy()
        self.Vecs = np.zeros((self.Vecs.shape[0], self.Vecs.shape[1] + 6), dtype=float)
        self.Vecs[:6, :6] = np.eye(6)
        self.Vecs[:, 6:] = subVecs.copy()
        # pmat2d(self.Vecs, precision=3, format='f')
        # sys.exit()
        # This is the number of nonredundant DLCs that we expect to see
        G = self.Prims.GMatrix(xyz)
        Expect = np.sum(np.linalg.eigh(G)[0] > 1e-6)

        # Carry out Gram-Schmidt orthogonalization
        # Define a function for computing overlap
        def ov(vi, vj):
            return multi_dot([vi, G, vj])

        V = self.Vecs.copy()
        nv = V.shape[1]
        Vnorms = np.array([np.sqrt(ov(V[:, ic], V[:, ic])) for ic in range(nv)])
        # G_tmp = np.array([[ov(V[:, ic], V[:, jc]) for jc in range(nv)] for ic in range(nv)])
        # pmat2d(G_tmp, precision=3, format='f')
        # U holds the Gram-Schmidt orthogonalized DLCs
        U = np.zeros((V.shape[0], Expect), dtype=float)
        Unorms = np.zeros(Expect, dtype=float)
        ncon = len(self.Prims.cPrims)
        # Keep translations, rotations, and any constraints in sequential order
        # and project them out of the remaining DLCs
        for ic in range(6 + ncon):
            U[:, ic] = V[:, ic].copy()
            ui = U[:, ic]
            Unorms[ic] = np.sqrt(ov(ui, ui))
            if Unorms[ic] / Vnorms[ic] < 0.1:
                logger.warning(
                    "Constraint %i is almost redundant; after projection norm is %.3f of original\n"
                    % (ic - 6, Unorms[ic] / Vnorms[ic])
                )
            V0 = V.copy()
            # Project out newest U column from all remaining V columns.
            for jc in range(ic + 1, nv):
                vj = V[:, jc]
                vj -= ui * ov(ui, vj) / Unorms[ic] ** 2
        # Now keep the remaining DLC with the largest norm, perform projection,
        # then repeat until the expected number is found
        shift = 6 + ncon
        for ic in range(shift, Expect):
            # Pick out the V column with the largest norm
            norms = np.array(
                [np.sqrt(ov(V[:, jc], V[:, jc])) for jc in range(shift, nv)]
            )
            imax = shift + np.argmax(norms)
            # Add this column to U
            U[:, ic] = V[:, imax].copy()
            ui = U[:, ic]
            Unorms[ic] = np.sqrt(ov(ui, ui))
            # Project out the newest U column from all V columns
            for jc in range(ncon, nv):
                V[:, jc] -= ui * ov(ui, V[:, jc]) / Unorms[ic] ** 2
        # self.Vecs contains the linear combination coefficients that are our new DLCs
        self.Vecs = U[:, 6:].copy()
        self.Internals = [
            "Constraint" if i < ncon else "DLC" + " %i" % (i + 1)
            for i in range(self.Vecs.shape[1])
        ]

    def augmentGH(self, xyz, G, H):
        """
        Add extra dimensions to the gradient and Hessian corresponding to the constrained degrees of freedom.
        The Hessian becomes:  H  c
                              cT 0
        where the elements of cT are the first derivatives of the constraint function
        (typically a single primitive minus a constant) with respect to the DLCs.

        Since we picked a DLC to represent the constraint (cProj), we only set one element
        in each row of cT to be nonzero. Because cProj = a_i * Prim_i + a_j * Prim_j, we have
        d(Prim_c)/d(cProj) = 1.0/a_c where "c" is the index of the primitive being constrained.

        The extended elements of the Gradient are equal to the constraint violation.

        Parameters
        ----------
        xyz : np.ndarray
            Flat array containing Cartesian coordinates in atomic units
        G : np.ndarray
            Flat array containing internal coordinate gradient
        H : np.ndarray
            Square array containing internal coordinate Hessian

        Returns
        -------
        GC : np.ndarray
            Flat array containing gradient extended by constraint violations
        HC : np.ndarray
            Square matrix extended by partial derivatives d(Prim)/d(cProj)
        """
        # Number of internals (elements of G)
        ni = len(G)
        # Number of constraints
        nc = len(self.Prims.cPrims)
        # Total dimension
        nt = ni + nc
        # Lower block of the augmented Hessian
        cT = np.zeros((nc, ni), dtype=float)
        # The further change needed in constrained variables:
        # (Constraint values) - (Current values of constraint ICs)
        c0 = -1.0 * self.calcConstraintDiff(xyz)
        for ic, c in enumerate(self.Prims.cPrims):
            # Look up the index of the primitive that is being constrained
            iPrim = self.Prims.Internals.index(c)
            # The DLC corresponding to the constrained primitive (a.k.a. cProj) is self.Vecs[self.cDLC[ic]].
            # For a differential change in the DLC, the primitive that we are constraining changes by:
            cT[ic, self.cDLC[ic]] = 1.0 / self.Vecs[iPrim, self.cDLC[ic]]
            # The new constraint algorithm satisfies constraints too quickly and could cause
            # the energy to blow up. Thus, constraint steps are restricted to 0.1 au/radian
            if self.conmethod == 1:
                if c0[ic] < -0.1:
                    c0[ic] = -0.1
                if c0[ic] > 0.1:
                    c0[ic] = 0.1
        # Construct augmented Hessian
        HC = np.zeros((nt, nt), dtype=float)
        HC[0:ni, 0:ni] = H[:, :]
        HC[ni:nt, 0:ni] = cT[:, :]
        HC[0:ni, ni:nt] = cT.T[:, :]
        # Construct augmented gradient
        GC = np.zeros(nt, dtype=float)
        GC[0:ni] = G[:]
        GC[ni:nt] = -c0[:]
        return GC, HC

    def applyConstraints(self, xyz):
        """
        Pass in Cartesian coordinates and return new coordinates that satisfy the constraints exactly.
        """
        xyz1 = xyz.copy()
        niter = 0
        xyzs = []
        ndqs = []
        while True:
            dQ = np.zeros(len(self.Internals), dtype=float)
            cDiff = -1.0 * self.calcConstraintDiff(xyz1)
            for ic, c in enumerate(self.Prims.cPrims):
                # Look up the index of the primitive that is being constrained
                iPrim = self.Prims.Internals.index(c)
                # Look up the index of the DLC that corresponds to the constraint
                iDLC = self.cDLC[ic]
                # Calculate the further change needed in this constrained variable
                dQ[iDLC] = cDiff[ic]
                dQ[iDLC] /= self.Vecs[iPrim, iDLC]
            xyzs.append(xyz1.copy())
            ndqs.append(np.linalg.norm(dQ))
            # print("applyConstraints calling newCartesian (%i), |dQ| = %.3e" % (niter, np.linalg.norm(dQ)))
            xyz2 = self.newCartesian(xyz1, dQ, verbose=0)
            if np.linalg.norm(dQ) < 1e-6:
                return xyz2
            if niter > 1 and np.linalg.norm(dQ) > np.linalg.norm(dQ0):
                xyz1 = xyzs[np.argmin(ndqs)]
                if not self.enforce_fail_printed:
                    logger.warning(
                        "Warning: Failed to enforce exact constraint satisfaction. Please remove possible redundant constraints. See below:\n"
                    )
                    self.printConstraints(xyz1, thre=0.0)
                    self.enforce_fail_printed = True
                return xyz1
            xyz1 = xyz2.copy()
            niter += 1
            dQ0 = dQ.copy()

    def newCartesian_withConstraint(self, xyz, dQ, thre=0.1, verbose=0):
        xyz2 = self.newCartesian(xyz, dQ, verbose)
        constraintSmall = len(self.Prims.cPrims) > 0
        cDiff = self.calcConstraintDiff(xyz)
        for ic, c in enumerate(self.Prims.cPrims):
            diff = cDiff[ic]
            if np.abs(diff) > thre:
                constraintSmall = False
        if constraintSmall:
            xyz2 = self.applyConstraints(xyz2)
            if not self.enforced:
                logger.info("<<< Enforcing constraint satisfaction >>>\n")
            self.enforced = True
        else:
            self.enforced = False
        return xyz2

    def calcGradProj(self, xyz, gradx):
        """
        Project out the components of the internal coordinate gradient along the
        constrained degrees of freedom. This is used to calculate the convergence
        criteria for constrained optimizations.

        Parameters
        ----------
        xyz : np.ndarray
            Flat array containing Cartesian coordinates in atomic units
        gradx : np.ndarray
            Flat array containing gradient in Cartesian coordinates

        Returns
        -------
        np.ndarray
            Flat array containing gradient in Cartesian coordinates with forces
            along constrained directions projected out
        """
        if len(self.Prims.cPrims) == 0:
            return gradx
        q0 = self.calculate(xyz)
        Ginv = self.GInverse(xyz)
        Bmat = self.wilsonB(xyz)
        # Internal coordinate gradient
        # Gq = np.matrix(Ginv)*np.matrix(Bmat)*np.matrix(gradx).T
        Gq = multi_dot([Ginv, Bmat, gradx.T])
        Gqc = np.array(Gq).flatten()
        # Remove the directions that are along the DLCs that we are constraining
        for i in self.cDLC:
            Gqc[i] = 0.0
        # Gxc = np.array(np.matrix(Bmat.T)*np.matrix(Gqc).T).flatten()
        Gxc = multi_dot([Bmat.T, Gqc.T]).flatten()
        return Gxc
