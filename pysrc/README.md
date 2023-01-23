# Switch Implementation
This repo is intended to hold all necessary parts of a crystal structure relaxation algorithm which switches from first to second order optimisation method. It includes implementations of Buckingham-Coulomb potential, the energy function's analytical derivatives and versions of the following methods (in progress):

* Conjugate Gradient

## potential.py

Includes a *Potential* superclass and *Coulomb*, *Buckingham* as subclasses. One can set the respective required parameters with 

```
set_parameters(alpha, real_cut_off, recip_cut_off, chemical_symbols, charge_dict, filename)
```
for Coulomb and 
```
set_parameters(self, filename, chemical_symbols)
```
for Buckingham. The resulting energy is calculated by summing the returned values as below:
```
Coulomb.calc(Atoms atoms) + Buckingham.calc(Atoms atoms)
```

## forces.py

This file follows the same logic as before with a *Forces* superclass and *DCoulomb*, *DBuckingham* subclasses. The full gradient is calculated with the following calls:

```
DCoulomb.calc_real(pos, vects, N) + DCoulomb.calc_recip(pos,vects, N) + DBuckinham.calc(pos, vects, N)
```
where *pos* is Nx3 array of the ions' positions, *vects* is a 3x3 array with the vectors describing the unit cell and N is the number of ions.


## calculate_energy.py

Various calculations including energy, gradient, finite differences to compare with the values of the gradient vector. The energy can be computed using different software and can be printed with a pre-configured template. The program can be run with input file using the *-i* flag or with hard-coded structures.

### Notes for external software

In order to use LAMMPS as a library through python:
- create a conda environment with python 3.7
- install ase and lammps
- set variable ASE_LAMMPS_RUN_COMMAND