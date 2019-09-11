from pymatgen.symmetry.analyzer import PointGroupAnalyzer
from pymatgen.core.structure import Molecule


for i, g in enumerate(geo):
    mol = Molecule(atomic_nums, g)
    pga = PointGroupAnalyzer(mol)
    if str(pga.get_pointgroup()) != "C1":
        print(i)
