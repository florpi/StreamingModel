import numpy as np

class CLPT():
	'''
	Reads predictions from Convolutional Lagrangian PT code

	'''

	def __init__(self,
			data_path: str,
			linear_growth: float,
			):
		'''
		Args:
			data_path: path where outputs from clpt code are stored.
			linear_growth: linear prediction for the growth factor.
		'''

		self.linear_growth = linear_growth
		# Free bias parameters
		self.b1 = None # 1st order lagrangian bias
		self.b2 = None # 2nd order lagrangian bias

		self.xifile = np.loadtxt(data_path + 'xi.txt')
		self.v12file = np.loadtxt(data_path + 'v12.txt')
		self.s12file = np.loadtxt(data_path + 's12.txt')
		self.multipoles_file = np.loadtxt(data_path + 'xi_s.txt')

		self.r = self.xifile[:,0]

	def get_multipoles(self):

		s = self.multipoles_file[:,0]
		monopole = self.multipoles_file[:,1]
		quadrupole = self.multipoles_file[:,2]
		hexadecapole = self.multipoles_file[:,3]

		return s, monopole, quadrupole, hexadecapole

	
	def get_linear_tpcf(self):

		return self.xifile[:,1]


	def get_tpcf(self, r, b1, b2):
		# Note r only needed when using curve_fit

		r_min = np.min(r)
		r_max = np.max(r)

		r_range = (self.r >= r_min) & (self.r <= r_max)
		return (self.xifile[:,2] + b1*self.xifile[:,3] + b2*self.xifile[:,4] + \
				(b1**2)*self.xifile[:,5] + b1*b2*self.xifile[:,6] + b2**2*self.xifile[:,7])[r_range]

	def get_linear_v12(self):
		return self.v12file[:,1]

	def get_v12(self,b1, b2):

		return self.linear_growth*( self.v12file[:,2] + b1*self.v12file[:,3] + b2*self.v12file[:,4] + \
				b1**2*self.v12file[:,5] + b1*b2*self.v12file[:,6] ) /(1.+ self.get_tpcf(self.r, b1, b2))   

	def get_s12(self, b1, b2):

		s12_par  = self.linear_growth**2*( self.s12file[:,1] + b1*self.s12file[:,2] + \
				b2*self.s12file[:,3] + b1**2*self.s12file[:,4] ) / (1.+ self.get_tpcf(self.r, b1,b2))  

		s12_perp = self.linear_growth**2*0.5*( self.s12file[:,5] + b1*self.s12file[:,6] +\
				b2*self.s12file[:,7] + b1**2*self.s12file[:,8] ) / (1.+ self.get_tpcf(self.r, b1, b2))

		return s12_par, s12_perp



