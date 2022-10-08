from geometric.ase_optimizer import GeomeTRIC
from xtb.ase.calculator import XTB
import ase.io


if __name__ == '__main__':

    atoms = ase.io.read("water6.xyz")
    atoms.calc = XTB(method="GFN2-xTB")
    opt = GeomeTRIC(atoms)
    opt.run()
