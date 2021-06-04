import itertools
from collections import defaultdict
from copy import deepcopy

import networkx as nx
import numpy as np

from geometric.molecule import Molecule
from geometric.nifty import ang2bohr
from .ic_simple import SimpleIC
from .slots import (
    Angle,
    CartesianX,
    CartesianY,
    CartesianZ,
    Dihedral,
    Distance,
    LinearAngle,
    OutOfPlane,
    RotationA,
    RotationB,
    RotationC,
    TranslationX,
    TranslationY,
    TranslationZ,
)


class PrimitiveInternalCoordinates(SimpleIC):
    def __init__(
        self,
        molecule,
        connect=False,
        addcart=False,
        constraints=None,
        cvals=None,
        **kwargs
    ):
        super(PrimitiveInternalCoordinates, self).__init__(molecule)
        self.connect = connect
        self.addcart = addcart

        # non covalent bonds in topology,
        # fixme: this should live in the molecule in fact
        self.noncov = set()

        for i in range(len(self.molecule)):
            self.makePrimitives(self.molecule[i])

        # Assume we're using the first image for constraints
        self.makeConstraints(self.molecule[0], constraints, cvals)

        # Reorder primitives for checking with cc's code in TC.
        # Note that reorderPrimitives() _must_ be updated with each new InternalCoordinate class written.
        self.reorderPrimitives()

    def build_all_cartesians(self, molecule: Molecule):
        # adding all Cartesians of the molecule
        # use for HDLC
        for i in range(molecule.na):
            self.add(CartesianX(i, w=1.0))
            self.add(CartesianY(i, w=1.0))
            self.add(CartesianZ(i, w=1.0))

    def build_bonds(self, molecule: Molecule):
        # builds bonds

        # Add an internal coordinate for all interatomic distances
        for (a, b) in molecule.topology.edges():
            self.add(Distance(a, b))

    def build_dlc_connections(self, molecule: Molecule):
        # DLC kind of connections

        # Make a distance matrix mapping atom pairs to interatomic distances
        AtomIterator, dxij = molecule.distance_matrix(pbc=False)
        D = {}
        for i, j in zip(AtomIterator, dxij[0]):
            assert i[0] < i[1]
            D[tuple(i)] = j
        dgraph = nx.Graph()
        for i in range(molecule.na):
            dgraph.add_node(i)
        for k, v in D.items():
            dgraph.add_edge(k[0], k[1], weight=v)
        mst = sorted(list(nx.minimum_spanning_edges(dgraph, data=False)))

        # Build a list of noncovalent distances
        for edge in mst:
            if edge not in list(molecule.topology.edges()):
                # print "Adding %s from minimum spanning tree" % str(edge)
                molecule.topology.add_edge(edge[0], edge[1])
                self.noncov.add(edge)

        # now you should call build_bonds and build_angles!

    def build_tric_connections(self, molecule: Molecule):
        # connections between molecules for TRIC type coordinates

        # find the fragments
        molecule.build_topology()
        if "resid" in molecule.Data.keys():
            frags = []
            current_resid = -1
            for i in range(molecule.na):
                if molecule.resid[i] != current_resid:
                    frags.append([i])
                    current_resid = molecule.resid[i]
                else:
                    frags[-1].append(i)
        else:
            frags = [m.nodes() for m in molecule.molecules]

        # coordinates in Angstrom
        coords = molecule.xyzs[0].flatten()

        # build TRIC coords
        for i in frags:
            if len(i) >= 2:
                self.add(TranslationX(i, w=np.ones(len(i)) / len(i)))
                self.add(TranslationY(i, w=np.ones(len(i)) / len(i)))
                self.add(TranslationZ(i, w=np.ones(len(i)) / len(i)))
                # Reference coordinates are given in Bohr.
                sel = coords.reshape(-1, 3)[i, :] * ang2bohr
                sel -= np.mean(sel, axis=0)
                rg = np.sqrt(np.mean(np.sum(sel ** 2, axis=1)))
                self.add(RotationA(i, coords * ang2bohr, self.Rotators, w=rg))
                self.add(RotationB(i, coords * ang2bohr, self.Rotators, w=rg))
                self.add(RotationC(i, coords * ang2bohr, self.Rotators, w=rg))
            else:
                for j in i:
                    self.add(CartesianX(j, w=1.0))
                    self.add(CartesianY(j, w=1.0))
                    self.add(CartesianZ(j, w=1.0))

    def makePrimitives(self, molecule: Molecule):
        # Connections of fragments for each coordinate type
        if self.connect:
            # this is DLC
            self.build_dlc_connections(molecule)
        else:
            if self.addcart:
                # this HDLC
                self.build_all_cartesians(molecule)
            else:
                # this is TRIC
                self.build_tric_connections(molecule)

        # bonds with non-covalent bonds included if built
        self.build_bonds(molecule)

        # angles and dihedrals
        self.build_angles_and_dihedrals(molecule)

    def build_angles_and_dihedrals(self, molecule):
        # coordinates in Angstrom
        coords = molecule.xyzs[0].flatten()

        # Add an internal coordinate for all angles
        # LinThre = 0.99619469809174555
        # LinThre = 0.999
        # This number works best for the iron complex
        LinThre = 0.95
        AngDict = defaultdict(list)
        for b in molecule.topology.nodes():
            for a in molecule.topology.neighbors(b):
                for c in molecule.topology.neighbors(b):
                    if a < c:
                        # if (a, c) in molecule.topology.edges() or (c, a) in molecule.topology.edges(): continue
                        Ang = Angle(a, b, c)

                        # logger.info("LPW: cosine of angle", a, b, c, "is", np.abs(np.cos(Ang.value(coords))))
                        if np.abs(np.cos(Ang.value(coords))) < LinThre:
                            self.add(Angle(a, b, c))
                            AngDict[b].append(Ang)
                        elif self.connect or not self.addcart:  # DLC & TRIC
                            # logger.info("Adding linear angle")
                            # Add linear angle IC's
                            # LPW 2019-02-16: Linear angle ICs work well for "very" linear angles in molecules (e.g. HCCCN)
                            # but do not work well for "almost" linear angles in noncovalent systems (e.g. H2O6).
                            # Bringing back old code to use "translations" for the latter case, but should be investigated
                            # more deeply in the future.
                            if (min(a, b), max(a, b)) not in self.noncov and (
                                min(b, c),
                                max(b, c),
                            ) not in self.noncov:
                                self.add(LinearAngle(a, b, c, 0))
                                self.add(LinearAngle(a, b, c, 1))
                            else:
                                # Unit vector connecting atoms a and c
                                nac = molecule.xyzs[0][c] - molecule.xyzs[0][a]
                                nac /= np.linalg.norm(nac)
                                # Dot products of this vector with the Cartesian axes
                                dots = [np.abs(np.dot(ei, nac)) for ei in np.eye(3)]
                                # Functions for adding Cartesian coordinate
                                # carts = [CartesianX, CartesianY, CartesianZ]
                                trans = [TranslationX, TranslationY, TranslationZ]
                                w = np.array([-1.0, 2.0, -1.0])
                                # Add two of the most perpendicular Cartesian coordinates
                                for i in np.argsort(dots)[:2]:
                                    self.add(trans[i]([a, b, c], w=w))

        for b in molecule.topology.nodes():
            for a in molecule.topology.neighbors(b):
                for c in molecule.topology.neighbors(b):
                    for d in molecule.topology.neighbors(b):
                        if a < c < d:
                            for i, j, k in sorted(
                                list(itertools.permutations([a, c, d], 3))
                            ):
                                Ang1 = Angle(b, i, j)
                                Ang2 = Angle(i, j, k)
                                if np.abs(np.cos(Ang1.value(coords))) > LinThre:
                                    continue
                                if np.abs(np.cos(Ang2.value(coords))) > LinThre:
                                    continue
                                if (
                                    np.abs(
                                        np.dot(
                                            Ang1.normal_vector(coords),
                                            Ang2.normal_vector(coords),
                                        )
                                    )
                                    > LinThre
                                ):
                                    self.delete(Angle(i, b, j))
                                    self.add(OutOfPlane(b, i, j, k))
                                    break

        # Find groups of atoms that are in straight lines
        atom_lines = [list(i) for i in molecule.topology.edges()]
        while True:
            # For a line of two atoms (one bond):
            # AB-AC
            # AX-AY
            # i.e. AB is the first one, AC is the second one
            # AX is the second-to-last one, AY is the last one
            # AB-AC-...-AX-AY
            # AB-(AC, AX)-AY
            atom_lines0 = deepcopy(atom_lines)
            for aline in atom_lines:
                # Imagine a line of atoms going like ab-ac-ax-ay.
                # Our job is to extend the line until there are no more
                ab = aline[0]
                ay = aline[-1]
                for aa in molecule.topology.neighbors(ab):
                    if aa not in aline:
                        # If the angle that AA makes with AB and ALL other atoms AC in the line are linear:
                        # Add AA to the front of the list
                        if all(
                            [
                                np.abs(np.cos(Angle(aa, ab, ac).value(coords)))
                                > LinThre
                                for ac in aline[1:]
                                if ac != ab
                            ]
                        ):
                            aline.insert(0, aa)
                for az in molecule.topology.neighbors(ay):
                    if az not in aline:
                        if all(
                            [
                                np.abs(np.cos(Angle(ax, ay, az).value(coords)))
                                > LinThre
                                for ax in aline[:-1]
                                if ax != ay
                            ]
                        ):
                            aline.append(az)
            if atom_lines == atom_lines0:
                break
        atom_lines_uniq = []
        for i in atom_lines:  #
            if tuple(i) not in set(atom_lines_uniq):
                atom_lines_uniq.append(tuple(i))
        lthree = [l for l in atom_lines_uniq if len(l) > 2]
        # TODO: Perhaps should reduce the times this is printed out in reaction paths
        # if len(lthree) > 0:
        #     print "Lines of three or more atoms:", ', '.join(['-'.join(["%i" % (i+1) for i in l]) for l in lthree])

        # Normal dihedral code
        for aline in atom_lines_uniq:
            # Go over ALL pairs of atoms in a line
            for (b, c) in itertools.combinations(aline, 2):
                if b > c:
                    (b, c) = (c, b)
                # Go over all neighbors of b
                for a in molecule.topology.neighbors(b):
                    # Go over all neighbors of c
                    for d in molecule.topology.neighbors(c):
                        # Make sure the end-atoms are not in the line and not the same as each other
                        if a not in aline and d not in aline and a != d:
                            # print aline, a, b, c, d
                            Ang1 = Angle(a, b, c)
                            Ang2 = Angle(b, c, d)
                            # Eliminate dihedrals containing angles that are almost linear
                            # (should be eliminated already)
                            if np.abs(np.cos(Ang1.value(coords))) > LinThre:
                                continue
                            if np.abs(np.cos(Ang2.value(coords))) > LinThre:
                                continue
                            self.add(Dihedral(a, b, c, d))
