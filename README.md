
veltiCRYS
===============

velti (from βελτιστοποίηση meaning optimization in Greek) for CRYStals is a collection of modules with functions useful for geometry optimization of ionic structures. It involves energy calculation with the Buckingham-Coulomb energy function potential and analytic first derivatives along with some local optimization methods. The input should be a collection of parameters that represents a unit cell with N ions, including a Nx3 (each row corresponding to a 3-dimensional ion position) and a 3x3 numpy array (each row corresponding to one 3-dimensional lattice vector).

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
- torch (if the autodiff version is to be used)

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
- torch (if the autodiff version is to be used)

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


In order to perform a geometry optimization calculation, run the Python script in file [run.py](run.py) with

```console
  python run.py [-h] [-i --input] [-r RELAX] [-u] [-out --out_frequency]
              [-o --output] [-su --max_step] [-sl --min_step]
              [-m --relaxation_method] [-d] [-ln --line_search]
              --mode
  
  --mode		    select between analytical (*analytical*) and autodifferentiation (*auto*) modes

  optional arguments:
    -h, --help              show this help message and exit
    -i --input              .cif file to read
    -r, --relax             Perform structural relaxation
    -u, --user              Wait for user input after every iteration
    -out, --out_frequency   Print files every n iterations
    -o, --output            Output directory
    -su, --max_step         Set upper bound of step size
    -sl, --min_step         Set lower bound of step size
    -m, --relaxation_method Choose updating method
    -res, --reset           Force direction reset every 3N+9 iterations.
    -d, --debug             Print numerical derivatives.
    -ln, --line_search      Type name of line search method and optional parameter value. 
    			     One of: 
      				-gnorm_scheduled_bisection <order>
				-scheduled_bisection <schedule>
				-scheduled_exp <exponent>
				-steady_step

```

You might need to add the necessary elements with their charge in **charge_dict** of file *run.py*. The library file *buck.lib* in *libraries* needs to include with the element-pair-dependent Buckingham parameters. The files already contain the required information for the dataset [data](data).

## Input

The crystal structures whose energy is to be calculated need be in a *.cif* file or be defined with the [Atoms](https://wiki.fysik.dtu.dk/ase/ase/atoms.html?highlight=atoms#ase.Atoms) class of ase. Example structures can be found in the folder [data](data), which contains a set of 200 crystal structures 
produced with a stable Strontium Titanate (\ch{Sr_3Ti_3O_9}) as a reference point. The length of each lattice vector is chosen from the values 4, 6, 8, 10, and 12 Angstroms and they form an orthorhombic unit cell containing 15 ions -- 3 strontium, 3 titanium and 9 oxygen ions. These ions are placed in a random manner on grid points defined by a 1 Angstrom grid spacing. The placement is such that the negative ions are placed on grid points with even indices, and positive ions and are placed on grid points with odd indices.
