from ase import *

def get_example1():
    name = "O2"
    atoms = Atoms(name, cell=[[4, 0.00, 0.00],	
						[0.00, 4, 0.00],
						[0.00, 0.00, 4]],
                positions=[[1, 1, 1],
                                [3.5, 3.5, 3.5]],
                pbc=True)
    return atoms, name


def get_example2():
    name = "SrO"
    atoms = Atoms(name, cell=[[4, 0.00, 0.00],	
						[0.00, 5, 0.00],
						[0.00, 0.00, 6]],
                positions=[[0, 0, 0],
                                [2, 2, 2]],
                pbc=True)
    return atoms, name


def get_example3():
    name = "SrO"
    atoms = Atoms(name, cell=[[4, 0.00, 0.00],	
						[0.00, 4, 0.00],
						[0.00, 0.00, 4]],
                positions=[[0, 0, 0],
                                [4, 4, 4]],
                pbc=True)
    return atoms, name