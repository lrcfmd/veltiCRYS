### Python scripts

This package includes Python scripts that tie the Cython implementations together in order to produce an optimization run. More specifically, it consists of the following:

* [direction.py](direction.py)

This module includes methods that implement nonlinear Gradient Descent (GD) and Conjugate Gradient (CG) direction updates. CG is implemented with the Polak–Ribière formula.
GD needs only the function's gradient in order to run properly:
```python
	GD(grad)

```

whereas CG needs the current and previous residual of the function, as well as the last direction vector:
```python
	CG(grad, Residual=last_residual, Direction=last_direction)

```

* [finite_differences.py](finite_differences.py)

This script can be used for debugging purposes. It calculates the first derivative of a given function usibg finite differences and compares the result to the given analytical gradient.

* [linmin.py](linmin.py)

This module should be used to define and use line minimization procedures. Each procedure needs to follow a certain signature type in order to be compatible with the rest of the software:
```python
	linmin_example(atoms, strains, grad, gnorm, direction, potentials,
		init_energy, update, max_step=default_value, min_step=default_value, **kwargs)

```
The parameter updates are executed with the included function *calculate_temp_energy*. This function applies changes to lattice and ion positions using the `position_update` and `lattice_update` methods and returns a dictionary with the new energy value and the parameters' and step size values that were used for the update. Any parameter change should be performed using the given update methods. 


The file currently includes constant step size application (`steady_step` method) and step size scheduling rules:
1. `scheduled_bisection`
	This rule uses the median of min_step and max_step every scheduled number of iterations. The number of iterations to step size reduction can be defined as kwargs['schedule'] when calling the rule.

2. `scheduled_exp`
	This rule takes a max_step and mutliplies it by 0.999 at every iteration as long as max_step\*0.999>min_step.

3. `gnorm_scheduled_bisection`
	This rule uses the median of min_step and max_step whenever the gradient norm of a given function falls by an order of magnitude β. The size of the order of magnitude is defined in the main body of the optimization run. For example, if the gradient norm is initially 0.01 but in the next iteration becomes 0.001 and β=10 then the next step size value used will be (min_step+max_step)/2. 

* [utils.py](utils.py)

This module consists of simple methods for formatting appearance of outputs and input wrappers.