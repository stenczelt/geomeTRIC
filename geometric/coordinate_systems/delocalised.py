import numpy as np
from numpy.linalg import multi_dot

from .slots import (
    RotationA,
    RotationB,
    RotationC,
    TranslationX,
    TranslationY,
    TranslationZ,
)
from .primitive import PrimitiveInternalCoordinates
from .ic_mix import MixIC
from geometric.nifty import ang2bohr, click, logger


class DelocalizedInternalCoordinates(MixIC):
    def __init__(
        self,
        molecule,
        imagenr=0,
        build=False,
        connect=False,
        addcart=False,
        constraints=None,
        cvals=None,
        remove_tr=False,
        cart_only=False,
        conmethod=0,
    ):
        super(DelocalizedInternalCoordinates, self).__init__(molecule)
        # cart_only is just because of how I set up the class structure.
        if cart_only:
            return
        # Set the algorithm for constraint satisfaction.
        # 0 - Original algorithm implemented in 2016, constraints are satisfied slowly unless "enforce" is enabled
        # 1 - Updated algorithm implemented on 2019-03-20, constraints are satisfied instantly, "enforce" is not needed
        self.conmethod = conmethod
        # HDLC is given by (connect = False, addcart = True)
        # Standard DLC is given by (connect = True, addcart = False)
        # TRIC is given by (connect = False, addcart = False)
        # Build a minimum spanning tree
        self.connect = connect
        # Add Cartesian coordinates to all.
        self.addcart = addcart
        # The DLC contains an instance of primitive internal coordinates.
        self.Prims = PrimitiveInternalCoordinates(
            molecule,
            connect=connect,
            addcart=addcart,
            constraints=constraints,
            cvals=cvals,
        )
        self.na = molecule.na
        # Whether constraints have been enforced previously
        self.enforced = False
        self.enforce_fail_printed = False
        # Build the DLC's. This takes some time, so we have the option to turn it off.
        xyz = molecule.xyzs[imagenr].flatten() * ang2bohr
        if build:
            self.build_dlc(xyz)
        self.remove_tr = remove_tr
        if self.remove_tr:
            self.remove_TR(xyz)

    def clearCache(self):
        super(DelocalizedInternalCoordinates, self).clearCache()
        self.Prims.clearCache()

    def __repr__(self):
        return self.Prims.__repr__()

    def update(self, other):
        return self.Prims.update(other.Prims)

    def join(self, other):
        return self.Prims.join(other.Prims)

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

    def build_dlc_0(self, xyz):
        """
        Build the delocalized internal coordinates (DLCs) which are linear
        combinations of the primitive internal coordinates. Each DLC is stored
        as a column in self.Vecs.

        In short, each DLC is an eigenvector of the G-matrix, and the number of
        nonzero eigenvalues of G should be equal to 3*N.

        After creating the DLCs, we construct special ones corresponding to primitive
        coordinates that are constrained (cProj).  These are placed in the front (i.e. left)
        of the list of DLCs, and then we perform a Gram-Schmidt orthogonalization.

        This function is called at the end of __init__ after the coordinate system is already
        specified (including which primitives are constraints).

        Parameters
        ----------
        xyz : np.ndarray
            Flat array containing Cartesian coordinates in atomic units
        """
        # Perform singular value decomposition
        click()
        G = self.Prims.GMatrix(xyz)
        # Manipulate G-Matrix to increase weight of constrained coordinates
        if self.haveConstraints():
            for ic, c in enumerate(self.Prims.cPrims):
                iPrim = self.Prims.Internals.index(c)
                G[:, iPrim] *= 1.0
                G[iPrim, :] *= 1.0
        ncon = len(self.Prims.cPrims)
        # Water Dimer: 100.0, no check -> -151.1892668451
        time_G = click()
        L, Q = np.linalg.eigh(G)
        time_eig = click()
        # print "Build G: %.3f Eig: %.3f" % (time_G, time_eig)
        LargeVals = 0
        LargeIdx = []
        for ival, value in enumerate(L):
            # print ival, value
            if np.abs(value) > 1e-6:
                LargeVals += 1
                LargeIdx.append(ival)
        Expect = 3 * self.na
        # print "%i atoms (expect %i coordinates); %i/%i singular values are > 1e-6" % (self.na, Expect, LargeVals, len(L))
        # if LargeVals <= Expect:
        self.Vecs = Q[:, LargeIdx]

        # Vecs has number of rows equal to the number of primitives, and
        # number of columns equal to the number of delocalized internal coordinates.
        if self.haveConstraints():
            click()
            # print "Projecting out constraints...",
            V = []
            for ic, c in enumerate(self.Prims.cPrims):
                # Look up the index of the primitive that is being constrained
                iPrim = self.Prims.Internals.index(c)
                # Pick a row out of the eigenvector space. This is a linear combination of the DLCs.
                cVec = self.Vecs[iPrim, :]
                cVec = np.array(cVec)
                cVec /= np.linalg.norm(cVec)
                # This is a "new DLC" that corresponds to the primitive that we are constraining
                cProj = np.dot(self.Vecs, cVec.T)
                cProj /= np.linalg.norm(cProj)
                V.append(np.array(cProj).flatten())
                # print c, cProj[iPrim]
            # V contains the constraint vectors on the left, and the original DLCs on the right
            V = np.hstack((np.array(V).T, np.array(self.Vecs)))
            # Apply Gram-Schmidt to V, and produce U.
            # The Gram-Schmidt process should produce a number of orthogonal DLCs equal to the original number
            thre = 1e-6
            while True:
                U = []
                for iv in range(V.shape[1]):
                    v = V[:, iv].flatten()
                    U.append(v.copy())
                    for ui in U[:-1]:
                        U[-1] -= ui * np.dot(ui, v)
                    if np.linalg.norm(U[-1]) < thre:
                        U = U[:-1]
                        continue
                    U[-1] /= np.linalg.norm(U[-1])
                if len(U) > self.Vecs.shape[1]:
                    thre *= 10
                elif len(U) == self.Vecs.shape[1]:
                    break
                elif len(U) < self.Vecs.shape[1]:
                    raise RuntimeError(
                        "Gram-Schmidt orthogonalization has failed (expect %i length %i)"
                        % (self.Vecs.shape[1], len(U))
                    )
            # print "Gram-Schmidt completed with thre=%.0e" % thre
            self.Vecs = np.array(U).T
            # Constrained DLCs are on the left of self.Vecs.
            self.cDLC = [i for i in range(len(self.Prims.cPrims))]
        # Now self.Internals is no longer a list of InternalCoordinate objects but only a list of strings.
        # We do not create objects for individual DLCs but
        self.Internals = [
            "Constraint-DLC" if i < ncon else "DLC" + " %i" % (i + 1)
            for i in range(self.Vecs.shape[1])
        ]

    def build_dlc_1(self, xyz):
        """
        Build the delocalized internal coordinates (DLCs) which are linear
        combinations of the primitive internal coordinates. Each DLC is stored
        as a column in self.Vecs.

        After some thought, build_dlc_0 did not implement constraint satisfaction
        in the most correct way. Constraint satisfaction was rather slow and
        the --enforce 0.1 may be passed to improve performance. Rethinking how the
        G matrix is constructed provides a more fundamental solution.

        In the new approach implemented here, constrained primitive ICs (PICs) are
        first set aside from the rest of the PICs. Next, a G-matrix is constructed
        from the rest of the PICs and diagonalized to form DLCs, called "residual" DLCs.
        The union of the cPICs and rDLCs forms a generalized set of DLCs, but the
        cPICs are not orthogonal to each other or to the rDLCs.

        A set of orthogonal DLCs is constructed by carrying out Gram-Schmidt
        on the generalized set. Orthogonalization is carried out on the cPICs in order.
        Next, orthogonalization is carried out on the rDLCs using a greedy algorithm
        that carries out projection for each cPIC, then keeps the one with the largest
        remaining norm. This way we avoid keeping rDLCs that are almost redundant with
        the cPICs. The longest projected rDLC is added to the set and continued until
        the expected number is reached.

        One special note in orthogonalization is that the "overlap" between internal
        coordinates corresponds to the G matrix element. Thus, for DLCs that's a linear
        combination of PICs, then the overlap is given by:

        v_i * B * B^T * v_j = v_i * G * v_j

        Notes on usage: 1) When this is activated, constraints tend to be satisfied very
        rapidly even if the current coordinates are very far from the constraint values,
        leading to possible blowing up of the energies. In augment_GH, maximum steps in
        constrained degrees of freedom are restricted to 0.1 a.u./radian for this reason.

        2) Although the performance of this method is generally superior to the old method,
        the old method with --enforce 0.1 is more extensively tested and recommended.
        Thus, this method isn't enabled by default but provided as an optional feature.

        Parameters
        ----------
        xyz : np.ndarray
            Flat array containing Cartesian coordinates in atomic units
        """
        click()
        G = self.Prims.GMatrix(xyz)
        nprim = len(self.Prims.Internals)
        cPrimIdx = []
        if self.haveConstraints():
            for ic, c in enumerate(self.Prims.cPrims):
                iPrim = self.Prims.Internals.index(c)
                cPrimIdx.append(iPrim)
        ncon = len(self.Prims.cPrims)
        if cPrimIdx != list(range(ncon)):
            raise RuntimeError(
                "The constraint primitives should be at the start of the list"
            )
        # Form a sub-G-matrix that doesn't include the constrained primitives and diagonalize it to form DLCs.
        Gsub = G[ncon:, ncon:]
        time_G = click()
        L, Q = np.linalg.eigh(Gsub)
        # Sort eigenvalues and eigenvectors in descending order (for cleanliness)
        L = L[::-1]
        Q = Q[:, ::-1]
        time_eig = click()
        # print "Build G: %.3f Eig: %.3f" % (time_G, time_eig)
        # Figure out which eigenvectors from the G submatrix to include
        LargeVals = 0
        LargeIdx = []
        GEigThre = 1e-6
        for ival, value in enumerate(L):
            if np.abs(value) > GEigThre:
                LargeVals += 1
                LargeIdx.append(ival)
        # This is the number of nonredundant DLCs that we expect to have at the end
        Expect = np.sum(np.linalg.eigh(G)[0] > 1e-6)

        if (ncon + len(LargeIdx)) < Expect:
            raise RuntimeError(
                "Expected at least %i delocalized coordinates, but got only %i"
                % (Expect, ncon + len(LargeIdx))
            )
        # print("%i atoms (expect %i coordinates); %i/%i singular values are > 1e-6" % (self.na, Expect, LargeVals, len(L)))

        # Create "generalized" DLCs where the first six columns are the constrained primitive ICs
        # and the other columns are the DLCs formed from the rest
        self.Vecs = np.zeros((nprim, ncon + LargeVals), dtype=float)
        for i in range(ncon):
            self.Vecs[i, i] = 1.0
        self.Vecs[ncon:, ncon : ncon + LargeVals] = Q[:, LargeIdx]

        # Perform Gram-Schmidt orthogonalization
        def ov(vi, vj):
            return multi_dot([vi, G, vj])

        if self.haveConstraints():
            click()
            V = self.Vecs.copy()
            nv = V.shape[1]
            Vnorms = np.array([np.sqrt(ov(V[:, ic], V[:, ic])) for ic in range(nv)])
            # U holds the Gram-Schmidt orthogonalized DLCs
            U = np.zeros((V.shape[0], Expect), dtype=float)
            Unorms = np.zeros(Expect, dtype=float)

            for ic in range(ncon):
                # At the top of the loop, V columns are orthogonal to U columns up to ic.
                # Copy V column corresponding to the next constraint to U.
                U[:, ic] = V[:, ic].copy()
                ui = U[:, ic]
                Unorms[ic] = np.sqrt(ov(ui, ui))
                if Unorms[ic] / Vnorms[ic] < 0.1:
                    logger.warning(
                        "Constraint %i is almost redundant; after projection norm is %.3f of original\n"
                        % (ic, Unorms[ic] / Vnorms[ic])
                    )
                V0 = V.copy()
                # Project out newest U column from all remaining V columns.
                for jc in range(ic + 1, nv):
                    vj = V[:, jc]
                    vj -= ui * ov(ui, vj) / Unorms[ic] ** 2

            for ic in range(ncon, Expect):
                # Pick out the V column with the largest norm
                norms = np.array(
                    [np.sqrt(ov(V[:, jc], V[:, jc])) for jc in range(ncon, nv)]
                )
                imax = ncon + np.argmax(norms)
                # Add this column to U
                U[:, ic] = V[:, imax].copy()
                ui = U[:, ic]
                Unorms[ic] = np.sqrt(ov(ui, ui))
                # Project out the newest U column from all V columns
                for jc in range(ncon, nv):
                    V[:, jc] -= ui * ov(ui, V[:, jc]) / Unorms[ic] ** 2

            # self.Vecs contains the linear combination coefficients that are our new DLCs
            self.Vecs = U.copy()
            # Constrained DLCs are on the left of self.Vecs.
            self.cDLC = [i for i in range(len(self.Prims.cPrims))]

        self.Internals = [
            "Constraint" if i < ncon else "DLC" + " %i" % (i + 1)
            for i in range(self.Vecs.shape[1])
        ]
        # # LPW: Coefficients of DLC's are in each column and DLCs corresponding to constraints should basically be like (0 1 0 0 0 ..)
        # pmat2d(self.Vecs, format='f', precision=2)
        # B = self.Prims.wilsonB(xyz)
        # Bdlc = np.einsum('ji,jk->ik', self.Vecs, B)
        # Gdlc = np.dot(Bdlc, Bdlc.T)
        # # Expect to see a diagonal matrix here
        # print("Gdlc")
        # pmat2d(Gdlc, format='e', precision=2)
        # # Expect to see "large" eigenvalues here (no less than 0.1 ideally)
        # print("L, Q")
        # L, Q = np.linalg.eigh(Gdlc)
        # print(L)

    def build_dlc(self, xyz):
        # 0 - Original algorithm implemented in 2016, constraints are satisfied slowly unless "enforce" is enabled
        # 1 - Updated algorithm implemented on 2019-03-20, constraints are satisfied instantly, "enforce" is not needed
        if self.conmethod == 1:
            return self.build_dlc_1(xyz)
        elif self.conmethod == 0:
            return self.build_dlc_0(xyz)
        else:
            raise RuntimeError("Unsupported value of conmethod %i" % self.conmethod)

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

    def weight_vectors(self, xyz):
        """
        Not used anymore: Multiply each DLC by a constant so that a small displacement along each produces the
        same Cartesian displacement. Otherwise, some DLCs "move by a lot" and others only "move by a little".

        Parameters
        ----------
        xyz : np.ndarray
            Flat array containing Cartesian coordinates in atomic units
        """
        Bmat = self.wilsonB(xyz)
        Ginv = self.GInverse(xyz, None)
        eps = 1e-6
        dxdq = np.zeros(len(self.Internals))
        for i in range(len(self.Internals)):
            dQ = np.zeros(len(self.Internals), dtype=float)
            dQ[i] = eps
            dxyz = multi_dot([Bmat.T, Ginv, dQ.T])
            rmsd = np.sqrt(np.mean(np.sum(np.array(dxyz).reshape(-1, 3) ** 2, axis=1)))
            dxdq[i] = rmsd / eps
        dxdq /= np.max(dxdq)
        for i in range(len(self.Internals)):
            self.Vecs[:, i] *= dxdq[i]

    def __eq__(self, other):
        return self.Prims == other.Prims

    def __ne__(self, other):
        return not self.__eq__(other)

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

    def second_derivatives(self, coords):
        """ Obtain the second derivatives of the DLCs with respect to the Cartesian coordinates. """
        PrimDers = self.Prims.second_derivatives(coords)
        Answer2 = np.tensordot(self.Vecs, PrimDers, axes=(0, 0))
        return np.array(Answer2)

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

    def resetRotations(self, xyz):
        """ Reset the reference geometries for calculating the orientational variables. """
        self.Prims.resetRotations(xyz)
