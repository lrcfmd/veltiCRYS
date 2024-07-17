import torch
from torch import Tensor
import numpy as np
import math

def get_normals(vects: Tensor) -> Tensor:
	norm1 = torch.cross(vects[1,:],vects[2,:]) 
	norm2 = torch.cross(vects[2,:],vects[0,:]) 
	norm3 = torch.cross(vects[0,:],vects[1,:]) 
	normals = torch.cat((norm1, norm2, norm3))

	return torch.reshape(normals, shape=(3,3))

def inflated_cell_truncation(vects: Tensor, cutoff: float) -> Tensor:
    volume = torch.det(vects)
    normals = get_normals(vects)

	# Find translation distance 
    translate = np.zeros((3,))
    for i in range(3):
        nnorm = torch.linalg.vector_norm(normals[i])
        height = volume.item() / nnorm.item()
        translate[(i+2)%3] = math.ceil(round((cutoff-height/2) / height))

        if translate[(i+2)%3]<0:
            print("\x1B[33mWarning:\033[0m The translate vector component {}  \
                is negative: {} \n" % (i+2)%3,translate[(i+2)%3])
            return None
    
    translate[0] = math.ceil(translate[0])
    translate[1] = math.ceil(translate[1])
    translate[2] = math.ceil(translate[2])

    shifts_no = (2*int(translate[0])+1) * \
				(2*int(translate[1])+1) * \
				(2*int(translate[2])+1)-1
    if shifts_no == 0:
        return None

    count = 0
    shifts_np = np.zeros((shifts_no,3))
    for shift in np.ndindex(
		2*int(translate[0])+1, 2*int(translate[1])+1, 2*int(translate[2])+1 ):
		
        if shift!=(translate[0],translate[1],translate[2]):
            shifts_np[count][0] = shift[0] - translate[0]
            shifts_np[count][1] = shift[1] - translate[1]
            shifts_np[count][2] = shift[2] - translate[2]
            count += 1
    
    shifts = torch.from_numpy(shifts_np)
    shifts= torch.tensor(shifts_np, dtype=torch.float64, device=vects.device)
    
    return torch.matmul(shifts, vects)
