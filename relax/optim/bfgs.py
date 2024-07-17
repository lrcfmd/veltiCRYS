import scipy
import numpy as np
from ase.geometry import wrap_positions
from relax.optim.optimizer import Optimizer
from relax.analytic_potentials.coulomb.coulomb import Coulomb
from relax.analytic_potentials.buckingham.buckingham import Buckingham
from relax.analytic_potentials.potential import get_gnorm

from ase.visualize import view
from ase.io import write

import pickle
import sys, os
np.set_printoptions(threshold=sys.maxsize)

class BFGS(Optimizer):
    
    def __init__(self, charge_dict, atoms, outfile, gtol=1e-3, max_iter=70000):
        """  INITIALISATION  """
        vects 				= np.asarray(atoms.get_cell())
        chemical_symbols	= np.array(atoms.get_chemical_symbols())
        N 					= len(atoms.positions)
        initial_energy 		= 0

        # Define Coulomb potential object
        libfile = "libraries/madelung.lib"
        Cpot = Coulomb(
            chemical_symbols=chemical_symbols,
            N=N,
            charge_dict=charge_dict,
            filename=libfile)
        Cpot.set_cutoff_parameters(
            vects=vects, 
            N=N)
        initial_energy += Cpot.energy(atoms)

        # Define Buckingham potential object
        libfile = "libraries/buck.lib"
        Bpot = Buckingham(
            filename=libfile, 
            chemical_symbols=chemical_symbols, 
            )
        Bpot.set_cutoff_parameters(
            vects=vects, 
            N=N)
        initial_energy += Bpot.energy(atoms)

        # Print Ewald-related parameters
        Cpot.print_parameters()
        Bpot.print_parameters()

        potentials = {}	
        potentials['Coulomb'] = Cpot
        potentials['Buckingham'] = Bpot

        self.potentials = potentials
        self.atoms = atoms
        self.max_iter = max_iter
        self.tol = gtol
        self.x0 = np.concatenate((atoms.positions, np.ones((3,3))), axis=0).reshape(1, -1)[0]
        self.outfile = outfile
        self.iterno = 0
        self.gnorms = []

    def get_energy(self, params, potentials):
        # Bring to initial shape of 3D vectors
        params = params.reshape(-1, 3)
        # Get positions
        pos = params[:-3]
        N = len(pos)
        # Get strains
        strains = (params[-3:]-1)+np.identity(3)
        # Get lattice vectors
        vects = np.array(self.atoms.get_cell())

        # Make sure ions stay in unit cell
        pos = wrap_positions(params[:-3], self.atoms.get_cell())

        # Apply strains to all unit cell vectors
        pos = params[:-3] @ strains.T
        vects = np.array(self.atoms.get_cell())@ strains.T

        # Calculate new point on energy surface
        self.atoms.positions = pos
        self.atoms.set_cell(vects)
        
        # Assign parameters calculated with altered volume
        for name in self.potentials:
            if hasattr(self.potentials[name], 'set_cutoff_parameters'):
                self.potentials[name].set_cutoff_parameters(vects, len(pos))

        # Calculate energy on current PES point
        energy =0
        for name in potentials:
            if hasattr(potentials[name], 'energy'):
                energy += potentials[name].energy(pos_array=pos, 
                                                  vects_array=vects, 
                                                  N_=N)

        return energy

    def get_gradient(self, params, potentials):
        # Bring to initial shape of 3D vectors
        params = params.reshape(-1, 3)
        # Get positions
        pos = params[:-3]
        N = len(pos)
        # Get lattice vectors
        vects = np.array(self.atoms.get_cell())

        # Assign parameters calculated with altered volume
        for name in potentials:
            if hasattr(potentials[name], 'set_cutoff_parameters'):
                potentials[name].set_cutoff_parameters(vects, N)

        # Gradient for current point on PES
        grad = np.zeros((N+3,3))
        for name in potentials:
            if hasattr(potentials[name], 'gradient'):
                grad += np.array(potentials[name].gradient(
                pos_array=pos, vects_array=vects, N_=N))

        # Gradient norm
        gnorm = get_gnorm(grad, N)
        self.gnorms.append(gnorm)
        # Normalise gradient
        if gnorm>0:
            grad_norm = grad/gnorm
        else:
            grad_norm = 0

        self.iterno += 1
        return grad_norm.reshape(1,-1)[0]
    
    def run(self):
        def callback(intermediate_result):
            g_index = -2
            if len(self.gnorms) < 2:
                g_index = -1

            # Keep info of this iteration
            iteration = {
            'Positions':self.atoms.positions, 'Strains':intermediate_result['x'][-3:], 
            'Cell':np.array(self.atoms.get_cell()), 'Iter':self.iterno, 
            'Step': 0, 'Gnorm':self.gnorms[g_index], 'Energy':intermediate_result['fun']
            }
            print(iteration)

            optimised = False
            if self.gnorms[-1] <= self.tol:
                optimised = True

            if not os.path.isdir(os.path.dirname(self.outfile)):
                os.mkdir(os.path.dirname(self.outfile)) 

            print("Writing result to file",
            self.outfile+"_"+str(self.iterno),"...")
            write(self.outfile+"_"+\
                str(self.iterno)+".cif", self.atoms)
            dict_file = open(
                self.outfile+"_"+\
                str(self.iterno)+".pkl", "wb")
            pickle.dump(
                {**iteration, 'Optimised': optimised}, 
                dict_file)
            dict_file.close()	

            if optimised:
                sys.exit()	
    
        res = scipy.optimize.minimize(self.get_energy, self.x0, 
                                args=(self.potentials), 
                                callback=callback,
                                method='BFGS', jac=self.get_gradient, 
                                tol=self.tol, options={'maxiter': self.max_iter})
        res_dict = {}
        for key, value in res.items():
            res_dict[key] = value

        # Termination
        print("Writing result to file",
        self.outfile+"_"+str(self.iterno),"...")
        write(self.outfile+"_final.cif", self.atoms)
        dict_file = open(
            self.outfile+"_final.pkl", "wb")
        pickle.dump(
            {**res_dict, 'gnorm': self.gnorms[-1]}, 
            dict_file)
        dict_file.close()
