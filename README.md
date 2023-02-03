
veltiCRYS
===============

velti (from βελτιστοποίηση meaning optimization in Greek) for CRYStals is a collection of modules with functions useful for geometry optimization of ionic structures. It involves energy calculation with the Buckingham-Coulomb energy function potential and analytic first derivatives along with some local optimization methods. Documentation of the background theory can be found in our paper [First Order Methods for Geometric Optimization of Crystal Structures](https://arxiv.org/abs/2301.12941). The input should be a collection of parameters that represents a unit cell with N ions, including a Nx3 (each row corresponding to a 3-dimensional ion position) and a 3x3 numpy array (each row corresponding to one 3-dimensional lattice vector).

The energy and derivative calculations are written in [Cython](https://cython.org/) and optimization step methods are written in [Python](https://www.python.org/about/).

&nbsp;

## Dependencies

The software's functionality have several dependencies, thus it is advised to create a virtual environment using either Python's venv or Anaconda.
If you prefer venev and pip then use the instructions below:

### PIP
Create a Python environment as shown
```console
  python -m venv ~/envelti

```
and activate the environment like so
```console
  source ~/envelti/bin/activate

```
then you have two options:

#### 1. Manual 
You can use pip and install the required packages with the command ```pip install``` (please use a recent version of pip e.g. 23.0). The essential packages for the software to work include:
- numpy
- ase
- cython
- jupyter notebook (optional, only if using the run.ipynb file)

#### 2. Requirements.txt
Otherwise, while the Python environment is activated, you can install the required packages using the given [requirements](requirements.txt) text file (please use a recent version of Python e.g. 3.11)
```console
   pip install -r requirements.txt

```
&nbsp;

If you are familiar and are using conda, it is advisable that you create a virtual environment using this option. 

### CONDA
#### 1. Manual 
You can create the environment as shown here 
```console
  conda create -n envelti

```
activate the environment as below
```console
  conda activate envelti

```
and install

- numpy
- ase
- cython
- jupyter notebook (optional, only if using the run.ipynb file)

using the ```conda install``` command. 

#### 2. Requirements.yml
Otherwise you can create an environment using the [requirements](requirements.yml) file and a Python version 3.6.13+ like so
```console
  conda env create -f requirements.yml

```
with the corresponding [requirements](requirements.yml) file.

&nbsp;

## Installation

In order to install the package, please clone this project and run the following command inside the main folder of the repository:

```console

  python setup.py build_ext --inplace

```
Any files and folders produced with this operation can be removed by running the bash script [clean.sh](clean.sh) from the root folder of the project.

&nbsp;

## Execution


In order to perform a geometry optimization calculation, run the Python script in file [calculate_energy.py](calculate_energy.py) with

```console
  python calculate_energy.py [-h] [-i --input] [-r] [-u] [-out --out_frequency]
                             [-o --output] [-s --max_step]
                             [-m --relaxation_method]

  Define input

  optional arguments:
    -h, --help             show this help message and exit
    -i --input             .cif file to read
    -r, --relax            Perform structural relaxation
    -u, --user             Wait for user input after every iteration
    -out --out_frequency   Print files every n iterations
    -o --output            Output directory
    -s --max_step          Use upper bound step size
    -m --relaxation_method Choose updating method
    -res, --reset         Force direction reset every 3N+9 iterations.
    -d, --debug           Print numerical derivatives.

```

You will need to add the necessary elements with their charge in **charge_dict** of file *calculate_energy* and adjust the corresponding input paths in **DATAPATH** of *utils.py*. **DATAPATH** needs to contain the library files *buck.lib* with the Buckingham parameters and *radii.lib* with the radii information of the element ions in a folder *libraries*. Such files can be found in the corresponding folder of the current repository. These contain the required information for the dataset [data](data).

Alternatively, you can open the jupyter notebook file [run.ipynb](run.ipynb).


### Classes 

The essential classes for energy calculation are the Cython extension types Coulomb and Buckingham. These include methods for evaluating both of the energy potentials using the Ewald summation expansion. They can be accessed by adding the following lines to a Python script:

```python
  from cysrc.buckingham import *
  from cysrc.coulomb import *

```

The class objects are defined as follows:

&nbsp;

#### Coulomb energy
_________________________

This is the electrostatic energy existing due to positively and negatively charged ions in the structure. It is a summation of terms each one of which corresponds to a pairwise interaction dependent on the distance between the ions. The alpha parameter depends on the cutoff values and is defined according to [Catlow's work](https://www.tandfonline.com/doi/abs/10.1080/08927028808080944). The cutoff value is then used in the `inflated_cell_truncation` method of [cutoff.pxd](cysrc/cutoff.pxd), a proposed geometric method of finding the images of neighbouring ions.

```python
  Cpot = Coulomb(chemical_symbols, N, charge_dict, alpha, filename)

```

Arguments of this function include:
 
| Argument | Function | 
| ---------------- | -------------------------------------------- |
| N                | Number of ions in unit cell                  |
| chemical_symbols | Ions' chemical symbols in resp. positions    |
| charge_dict      | Dictionary of ion charge per element         |
| filename         | File with Madelung constants (optional)      |
| alpha            | Balance between reciprocal and real space (optional)  |


&nbsp;

#### Buckingham energy
_________________________

Buckingham energy potential accounts for Pauli repulsion energy and van der Waals energy between two atoms as a function of the interatomic distance between them. The two terms of each Buckingham summand represent repulsion and attraction respectively. Parameters A, C and ρ are emperically determined in literature. These parameters have to be defined in a library file (here *buck.lib*) in the following format:
```
buck
element_name_1a core element_name_1b core  A   ρ C min_dist max_dist
element_name_2a core element_name_2b core  A   ρ C min_dist max_dist
```

```python 
  Bpot = Buckingham(filename, chemical_symbols, alpha, radius_lib, radii, limit)

```

Arguments of this function include:

| Argument | Function | 
| ---------------- | -----------------------------------------    |
| filename        | Library file with Buckingham constants        |
| chemical_symbols| Ions' chemical symbols in resp. positions     |
| radius_lib      | Library file with radius value per element ion|
| radii           | Array with radius per ion position (optional) |
| limit           | Lower bound limit of pairwise distances       |
| alpha           | Balance between reciprocal and real space (optional)   |
  

&nbsp;

Both classes need to be set with cutoff parameters using method

```python
  set_cutoff_parameters(self, vects, N, accuracy, real, reciprocal)

```
before each energy calculation, if the unit cell undergoes any changes. The cutoff values are then used to calculate pairwise distances of ions in neighbouring cells using the `inflated_cell_truncation method` in [cutoff.pyx](cysrc/cutoff.pyx). Each class contains also the methods

```python
  print_parameters(self)
  calc(self, atoms, pos_array, vects_array, N)
  get_energies(self)
  calc_drv(self, atoms, pos_array, vects_array, N)

```

Arguments of these functions include:

| Argument | Function | 
| ---------------- | -----------------------------------------    |
| N                | Number of ions in unit cell                  |
| pos_array        | Array with the atom positions (Nx3)          |
| vects_array      | Array with the lattice vectors (3x3)         |
| atoms            | Object of ase Atoms class (optional)         |


Both energy (`calc`) and derivatives (`calc_drv`) can be calculated  either by defining parameters `N, pos_array, vects_array` or `atoms`.

&nbsp;

#### Descent
_________________________

This class instantiates the optimization procedure. Its method `repeat` performs repeated iterations `iter_step` of nonlinear minimization. It can be configured with various tolerances that will make up the stopping criteria of the minimization. Every `iter_step` call returns a dictionary with all values related to the current iteration, so that the returning values include the gradient of the current configuration, the new direction vector for the next step, the ion positions' array, the strains tensor, the lattice vectors' array, the iteration number, the step size used, the gradient norm of the current configuration and the energy value of the current configuration
You can always import the Descent class from [descent.py](descent.py) and define an object for running optimizations with 

```python
  descent = Descent()

```

Arguments of this function include (all optional):

| Argument | Function | 
| ---------------- | -----------------------------------------    |
| iterno           | Maximum iteration number                     |
| ftol             | Energy value difference tolerance            |
| gtol             | Gradient norm tolerance                      |
| tol              | Step (step*direction_norm)/\|\|x\|\|) tolerance  |
| gmax             | Gradient component tolerance                 |


The method that executes the optimization is

```python
  repeat(self, init_energy, atoms, potentials, outdir, outfile,
    step_func, direction_func, strains, usr_flag, max_step, out)

```

Arguments of this function include:

| Argument | Function | 
| ---------------- | -----------------------------------------    |
| init_energy      | Initial energy value                         |
| atoms            | Object of ase Atoms class                    |
| potentials       | Dict. with potential objects indexed by name |
| outdir           | Folder containing any output files           |
| outfile          | Name for output files                        |
| step_func        | Function for line search (optional)          |
| direction_func   | Function for calculating direction vector    |
| strains          | Initial strain array (optional)              |
| usr_flag         | Initiative to stop for input after each iteration (optional)|
| max_step         | Upper bound for step size   (optional)       |
| out              | Interval of output production (optional)     |



where `step_func` and `direction_func` should be procedures. Line minimisation procedures can be defined in [linmin.py](pysrc/linmin.py) and procedures for the direction vector can be defined in [direction.py](pysrc/direction.py). Please refer to the examples already defined.

&nbsp;

## Input

The crystal structures whose energy is to be calculated need be in a *.cif* file or be defined with the [Atoms](https://wiki.fysik.dtu.dk/ase/ase/atoms.html?highlight=atoms#ase.Atoms) class of ase. Example structures can be found in the folder [data](data), which contains a set of 200 crystal structures 
produced with a stable Strontium Titanate (\ch{Sr_3Ti_3O_9}) as a reference point. The length of each lattice vector is chosen from the values 4, 6, 8, 10, and 12 Angstroms and they form an orthorhombic unit cell containing 15 ions -- 3 strontium, 3 titanium and 9 oxygen ions. These ions are placed in a random manner on grid points defined by a 1 Angstrom grid spacing. The placement is such that the negative ions are placed on grid points with even indices, and positive ions and are placed on grid points with odd indices.
