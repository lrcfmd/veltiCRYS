import os, torch
import numpy as np
import pickle
import time

import sys, signal
from ase.geometry import wrap_positions
from ase.io import write

from relax.autodiff_potentials.coulomb import Coulomb
from relax.autodiff_potentials.buckingham import Buckingham
from relax.autodiff_potentials.cutoff import inflated_cell_truncation
from relax.autodiff_potentials.ewaldpotential import EwaldPotential

import shutil
COLUMNS = shutil.get_terminal_size().columns
def prettyprint(dict_):
	import pprint
	np.set_printoptions(suppress=True)
	words = ""
	for key, value in dict_.items():
		if key=="Total energy":
			words += key+" "+str(value)+" "
		else:
			print("\n", key)
			print(value)
	print(words.center(COLUMNS,"-"))


def init(charge_dict, atoms, outdir):

    """  INITIALISATION  """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N 					= len(atoms.positions)
    strains_vec			= torch.tensor([1.,0.,0.,1.,0.,1.], dtype=torch.float64, device=device, requires_grad=True)
    strains				= torch.eye(3, dtype=torch.float64, device=device)
    vects_np 			= np.asarray(atoms.get_cell())
    vects 				= torch.tensor(vects_np, dtype=torch.float64, device=device, requires_grad=False)
    scaled_pos_np 		= np.asarray(atoms.get_scaled_positions())
    scaled_pos 			= torch.tensor(scaled_pos_np, requires_grad=True, device=device)
    accuracy			= 0.000000000000000000001
    chemical_symbols	= np.array(atoms.get_chemical_symbols())
    rng                 = np.random.default_rng(0)

    # Apply strains
    ind = torch.triu_indices(row=3, col=3, offset=0)
    strains[ind[0], ind[1]] = strains_vec
    vects				= torch.matmul(vects, torch.transpose(strains, 0, 1))
    pos 				= torch.matmul(scaled_pos, vects)
    volume 				= torch.det(vects)
    
    # Avoid truncating too many terms
    assert((-np.log(accuracy)/N**(1/6)) >= 1)	

    """  DEFINITIONS  """  	
    # Define Coulomb potential object
    Cpot = Coulomb(
        chemical_symbols=chemical_symbols,
        charge_dict=charge_dict,
        get_shifts=inflated_cell_truncation,
        device=device
    )
    Cpot.set_cutoff_parameters(
        vects=vects, 
        N=N)
    coulomb_energy = Cpot.all_energy(pos, vects, volume)

    # Define Buckingham potential object
    Bpot = Buckingham(
        filename='libraries/buck.lib',
        chemical_symbols=chemical_symbols,
        get_shifts=inflated_cell_truncation,
        device=device
    )
    Bpot.set_cutoff_parameters(
        vects=vects, 
        N=N)
    buckingham_energy = Bpot.all_energy(pos, vects, volume)

    potentials = {}	
    potentials['Coulomb'] = Cpot
    potentials['Buckingham'] = Bpot
    initial_energy = torch.add(coulomb_energy, buckingham_energy)

    if not os.path.isdir(outdir):
        os.mkdir(outdir) 

    prettyprint({
        'Chemical Symbols':chemical_symbols, 
        'Positions':atoms.positions, \
        'Cell':atoms.get_cell(), 
        'Electrostatic energy':coulomb_energy, 
        'Interatomic energy':buckingham_energy, \
        'Total energy':initial_energy
    })

    return potentials, strains_vec, strains, \
        vects, scaled_pos, device, rng


class GracefulKiller:
  kill_now = False
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self, signum, frame):
    self.kill_now = True


def repeat(atoms, outdir, outfile, charge_dict, line_search_fn,
    optimizer, usr_flag=False, out=1, **kwargs):
    """The function that performs the optimisation. It calls repetitively 
    iter_step for each updating step.

    Parameters
    ----------
    init_energy : double
        The energy of the initial configuration.
    atoms : Python ASE's Atoms instance.
        Object with the parameters to optimise.
    potentials : dict[str, Potential]
        Dictionary containing the names and Potential 
        instances of the energy functions to be used 
        as objective functions.
    outdir : str
        Name of the folder to place the output files.
    outfile : str
        Name of the output files.
    step_func : function
        The function to be used for line minimisation.
    direction_func : function
        The function to be used for the calculation of
        the direction vector (optimiser).
    usr_flag : bool
        Flag that is used to stop after each iteration
        and wait for user input, if true.
    max_step : float
        The upper bound of the step size.
    out : int
        Frequency of produced output files -- after 
        how many iterations the ouput should be written
        to a file.

    Returns
    -------
    (int, dict[str, _])

    """
    start_time = time.time()

    res                 = init(charge_dict, atoms, outdir)
    potentials          = res[0]
    strains_vec         = res[1]
    strains             = res[2]
    vects               = res[3]
    scaled_pos          = res[4]
    device              = res[5]
    rng                 = res[6]
    pos 				= torch.matmul(scaled_pos, vects)
    volume 				= torch.det(vects)
    N = len(pos)

    final_iteration = None
    history = []

    if not os.path.isdir(outdir+"imgs"):
        os.mkdir(outdir+"imgs")
    if not os.path.isdir(outdir+"structs"):
        os.mkdir(outdir+"structs")
    if not os.path.isdir(outdir+"imgs/"+outfile+"/"):
        os.mkdir(outdir+"imgs/"+outfile)
    if not os.path.isdir(outdir+"structs/"+outfile+"/"):
        os.mkdir(outdir+"structs/"+outfile)

    # Gradient for current point on PES
    grad = {
        'positions': torch.zeros((N, 3), device=device),
        'strains': torch.zeros((6,), device=device)
    }
    for name in potentials:
        grad_res = potentials[name].get_gradient(
            potentials[name].energy, scaled_pos, N,
            vects, strains_vec, volume)
        grad['positions'] = torch.add(
            grad['positions'], grad_res['positions'])
        grad['strains'] = torch.add(
            grad['strains'], grad_res['strains'])
        
    # Gradient norm
    gnorm = EwaldPotential.get_gnorm(grad)
    optimizer.lnscheduler.gnorm = gnorm
        
    # # Update Lipschitz constant if needed
    # if optimizer.requires_lipschitz:
    #     optimizer.cparams['L'] = max(torch.max(grad['positions']).item(), torch.mean(grad['strains']).item())
    
    # Hessian for current point on PES
    secdrv = {}
    hessian = torch.tensor(np.zeros((3*N+6, 3*N+6)), device=device)
    if optimizer.requires_hessian:
        for name in potentials:
            hess_res = potentials[name].get_hessian(
                potentials[name].grad, scaled_pos, vects, 
                strains_vec, volume)
        hessian = torch.add(hessian, hess_res)
        secdrv = {'Hessian': hessian, 'L': optimizer.cparams['L']}
        
    
    # Sum energy values
    total_energy = 0
    for name in potentials:
        total_energy += potentials[name].energy.item()

    # Keep info of this iteration
    iteration = {
    'Time': time.time()-start_time, 'Gradient': grad, 
    'Positions':atoms.get_positions(), 'Strains':np.ones((6,)), 
    'Cell':np.array(atoms.get_cell()), 'Iter':optimizer.iterno, 
    'Step': 0, 'Gnorm':gnorm, 'Energy':total_energy, **secdrv
    }

    # Iterations
    killer = GracefulKiller()
    while(not killer.kill_now):
        final_iteration = iteration

        # Check for termination
        prettyprint(iteration)
        if optimizer.completion_check(gnorm):
            print("Writing result to file",
            outfile+"_"+str(optimizer.iterno),"...")
            # write(outdir+"imgs/"+outfile+"/"+outfile+"_"+\
                # str(optimizer.iterno)+".png", atoms)
            write(outdir+"structs/"+outfile+"/"+outfile+"_"+\
                str(optimizer.iterno)+".cif", atoms)
            dict_file = open(
                outdir+"structs/"+outfile+"/"+outfile+"_"+\
                str(optimizer.iterno)+".pkl", "wb")
            pickle.dump(
                {**iteration, 'Optimised': True}, 
                dict_file)
            dict_file.close()				
            break
        elif (optimizer.iterno%out)==0:
            print("Writing result to file",
            outfile+"_"+str(optimizer.iterno),"...")
            # write(outdir+"imgs/"+outfile+"/"+outfile+"_"+\
                # str(optimizer.iterno)+".png", atoms)
            write(outdir+"structs/"+outfile+"/"+outfile+"_"+\
                str(optimizer.iterno)+".cif", atoms)
            dict_file = open(
                outdir+"structs/"+outfile+"/"+outfile+"_"+\
                str(optimizer.iterno)+".pkl", "wb")
            pickle.dump(
                {**iteration, 'Optimised': False}, 
                dict_file)
            dict_file.close()
        if (('iterno' in kwargs) & (kwargs['iterno'] <= optimizer.iterno)):
            break

        if usr_flag:
            usr = input()
            if 'n' in usr:
                return iteration
        
        # Tensors to numpy
        params = np.ones((N+2,3))
        params[:N] = pos.cpu().detach().numpy().copy()
        vects_np = vects.cpu().detach().numpy().copy()
        
        # Delete the tensors
        for name in potentials:
            del potentials[name].energy
        del pos
        del vects
        del volume
        del strains
        del strains_vec
        
        # Save grad to numpy
        grad_np = np.zeros((N+2, 3))
        grad_np[:N] = grad['positions'].cpu().detach().numpy()
        grad_np[N:] = np.reshape(grad['strains'].cpu().detach().numpy(),
                                 newshape=(2, 3))
        
        # Normalise gradient    
        grad_norm = np.zeros((N+2, 3))    
        if gnorm>0:
            grad_norm = grad_np/gnorm
        # Normalise hessian    
        hessian_np = hessian.cpu().detach().numpy()
        # hnorm = np.linalg.norm(hessian_np)
        # hess_norm = hessian_np/hnorm

        ''' 1 --- Apply an optimization step --- 1 '''
        params = optimizer.step(
            grad=grad_np, params=params, 
            line_search_fn=line_search_fn, 
            hessian=hessian_np, gnorm=gnorm,
            debug=False, rng=rng, atoms=atoms, potentials=potentials)

        # Make a method history
        history.append(type(optimizer).__name__)

        ''' 2 --- Update parameters --- 2 '''
        # Make sure ions stay in unit cell
        pos_temp = wrap_positions(params[:N], vects_np)
        # Update strains
        strains_np		= np.zeros((3,3))
        ind = np.triu_indices(3)
        strains_np[ind[0], ind[1]] = np.reshape(
            params[N:], newshape=(6,))
        strains_np[1][0]       = strains_np[0][1]
        strains_np[2][1]       = strains_np[1][2]
        strains_np[2][0]       = strains_np[0][2]
        strains_np = (strains_np-1)+np.identity(3)

        # Apply strains to all unit cell vectors as a 3x3 tensor
        pos_np = pos_temp @ strains_np.T
        vects_np = vects_np @ strains_np.T

        # Calculate new point on energy surface
        atoms.positions = pos_np
        atoms.set_cell(vects_np)
        
        # Get new tensors
        strains_vec = torch.tensor(
            strains_np[np.triu_indices(3)], device=device, requires_grad=True)
        strains		= .5*torch.ones((3,3), device=device)+ .5*torch.eye(3, device=device, dtype=torch.float64)
        ind = torch.triu_indices(row=3, col=3, offset=0)
        strains[ind[0], ind[1]] = strains[ind[0], ind[1]]*strains_vec
        strains[1][0]       = strains[0][1]
        strains[2][1]       = strains[1][2]
        strains[2][0]       = strains[0][2]
        
        vects 				= torch.tensor(vects_np, dtype=torch.float64, device=device, requires_grad=False)
        scaled_pos 			= torch.tensor(pos_np @ np.linalg.inv(vects_np), device=device, requires_grad=True)
        vects				= torch.matmul(vects, torch.transpose(strains, 0, 1))
        pos 				= torch.matmul(scaled_pos, vects)
        volume 				= torch.det(vects)

        # Assign parameters calculated with altered volume
        for name in potentials:
            if hasattr(potentials[name], 'set_cutoff_parameters'):
                potentials[name].set_cutoff_parameters(vects, N)

        ''' 3 --- Re-calculate energy --- 3 '''
        # Calculate energy on current PES point
        total_energy = 0
        for name in potentials:
            total_energy += potentials[name].all_energy(pos, vects, volume).item()

        ''' 4 --- Re-calculate derivatives --- 4 '''
        # Gradient for current point on PES
        grad = {
            'positions': torch.zeros((N, 3), device=device),
            'strains': torch.zeros((6,), device=device)
        }
        for name in potentials:
            grad_res = potentials[name].get_gradient(
                potentials[name].energy, scaled_pos, N,
                vects, strains_vec, volume)
            grad['positions'] = torch.add(
                grad['positions'], grad_res['positions'])
            grad['strains'] = torch.add(
                grad['strains'], grad_res['strains'])
            
        # Hessian for current point on PES
        hessian = torch.tensor(np.zeros((3*N+6, 3*N+6)), device=device)
        if optimizer.requires_hessian:
            for name in potentials:
                hess_res = potentials[name].get_hessian(
                    potentials[name].grad, scaled_pos, vects, 
                    strains_vec, volume)
            hessian = torch.add(hessian, hess_res)
            secdrv = {'Hessian': hessian, 'L': optimizer.cparams['L']}
            if type(optimizer).__name__ == 'CubicMin':
                secdrv = {**secdrv, 'Cubic': optimizer.reg_value}
    
        # Gradient norm
        gnorm = EwaldPotential.get_gnorm(grad)
        optimizer.lnscheduler.gnorm = gnorm        

        # Save grad to numpy
        grad_np = np.zeros((N+2, 3))
        grad_np[:N] = grad['positions'].cpu().detach().numpy()
        grad_np[N:] = np.reshape(grad['strains'].cpu().detach().numpy(),
                                 newshape=(2, 3))
            
        iteration = {
        'Time': time.time()-start_time,
        'Gradient':grad, 'Positions':atoms.positions.copy(), 
        'Strains':params[N:], 'Cell':np.array(atoms.get_cell()), 
        'Iter':optimizer.iterno, 'Method': history[-1], 
        'Step':optimizer.lnscheduler.curr_step, 'Gnorm':gnorm, 
        'Energy':total_energy, **secdrv}

    return final_iteration

