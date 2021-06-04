import numpy as np
from numpy.linalg import multi_dot

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
        conmethod=0,
    ):
        super(DelocalizedInternalCoordinates, self).__init__(molecule)

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
