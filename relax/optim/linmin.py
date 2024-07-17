import math

class LnSearch:
	def __init__(self, min_step=1e-5, max_step=1e-3, **kwargs) -> None:
		
		self.max_step=max_step
		self.curr_step=max_step
		self.min_step=min_step

		if 'order' in kwargs:
			self.order=kwargs['order']
		if 'schedule' in kwargs:
			self.schedule=kwargs['schedule']
		if 'exponent' in kwargs:
			self.exp = kwargs['exponent']
		if 'Gnorm' in kwargs:
			self.gnorm = kwargs['gnorm']


	def steady_step(self, **kwargs):
		return self.curr_step


	def scheduled_bisection(self, **kwargs):
		step = self.curr_step

		if (kwargs['iteration']>0) & (kwargs['iteration']%self.schedule==0):
			step = (self.curr_step + self.min_step)/2
	
		# Make sure step size decreases
		self.curr_step = step

		return step


	def scheduled_exp(self, **kwargs):	
		step = self.curr_step
		if (kwargs['iteration']>0):
			if step > self.min_step:
				step = self.curr_step*self.exp
    
		# Make sure step size decreases
		self.curr_step = step
  
		return step	


	def gnorm_scheduled_bisection(self, **kwargs):
		step = self.curr_step
		last_gnorm = self.gnorm
  
		# Gradient norm difference in magnitude with last gnorm
		self.gmag = -math.floor(math.log(kwargs['gnorm'], self.order)) + \
      		math.floor(math.log(last_gnorm, self.order))
		
		if self.gmag>0:
			step = (self.curr_step+self.min_step)/2
   
		# Make sure step size decreases
		self.curr_step = step
		self.gnorm = kwargs['gnorm']

		return step
