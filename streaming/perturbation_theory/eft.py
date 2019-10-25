import numpy as np
import sys
sys.path.insert(0, '../../../CLEFT_GSM/config2pt/')
from lesm import LSM
from scipy.interpolate import interp1d


class EFT():
	'''
	Reads predictions from Effective Field Theory PT code

	'''

	def __init__(self,
			data_path: str,
			linear_pk_file: str,
			linear_growth: float,
			b1:float=None,
			b2:float=None,
			bs:float=None,
			alpha_eft:float=None,
			alpha_eft_v:float=None,
			s_Fog:float=0.,
			run: bool = False,
			):
		'''
		Args:
			data_path: path where outputs from clpt code are stored.
			linear_growth: linear prediction for the growth factor.
			linear_pk_file: filename containing the linear power spectrum
			and its k range.
			run: 
		'''

		self.data_path = data_path
		self.linear_pk_file = linear_pk_file
		self.linear_growth = linear_growth
		# Free bias parameters
		self.b1 = b1 
		self.b2 = b2 
		self.bs = bs 
		# Free eft parameters
		# Fit to 2pcf
		self.alpha_eft = alpha_eft 
		# Fit mean velocity
		self.alpha_eft_v = alpha_eft_v 
		# Fit std
		self.beta_eft_sigma = 0. 
		self.s_Fog = s_Fog

		self.xifile = np.loadtxt(data_path + linear_pk_file + '.xiStuff')
		self.v12file = np.loadtxt(data_path + linear_pk_file + '.vpStuff')
		self.s12file = np.loadtxt(data_path + linear_pk_file + '.s2Stuff')

		self.r = self.xifile[:,0]

		if run:
			self.runCLEFT()

		if (self.b1 is not None) and (self.b2 is not None) and (self.bs is not None):
			self.tpcf = interp1d(self.r, self.get_tpcf(self.r, self.b1, self.b2, self.bs,self.alpha_eft),
				bounds_error=False, fill_value=(-1., 0.))
			self.m_10 = interp1d(self.r, self.get_v12(self.r, self.b1, self.b2, self.bs, self.alpha_eft, self.alpha_eft_v),
				bounds_error=False, fill_value=(0., 0.))
			c_20, c_02 = self.get_s12(self.b1, self.b2, self.bs, self.alpha_eft, self.alpha_eft_v, self.s_Fog)
			self.c_20 = interp1d(self.r, c_20, 
					bounds_error=False, fill_value=(c_20[0],c_20[-1]))
			self.c_02 = interp1d(self.r, c_02,
					bounds_error=False, fill_value=(c_02[0], c_02[-1]))
	

	def get_linear_tpcf(self):

		return self.xifile[:,1]

	def get_linear_v12(self):
		return self.v12file[:,1]


	def runCLEFT(self):
		'''

		Calls the EFT code to predict the GSM results using CLPT,
		it also produces the necessary files to read the ingredients predictions.

		Returns:
			model: function to compute the resulting multipoles using the Gaussian
			Streaming Model

		'''

		model = LSM()


	def get_multipoles(self,
			b1: float, 
			b2: float,
			bs: float,
			alpha_eft: float,
			alpha_eft_v: float,
			s2FoG: float):	
		'''

		Calls the EFT code to predict the GSM multipoles using CLPT

		Args:
			b1: 1st Lagrangian bias parameter.
			b2: 2nd Lagrangian bias parameter.
			bs: shear dependence of the bias
			alpha_eft: EFT counterterm
			alpha_eft_v: EFT counterterm for velocity only
			s2FoG: constant shift to standard deviation pairwise velocity

		Returns:
			xiell: Array containing the multipoles of the two point
			correlation function

		'''

		Apar = 1				  # AP parameter
		Aperp = 1				   # AP parameter

		model = LSM()

		
		xiell = model(self.data_path + self.linear_pk_file,self.linear_growth,
					b1,b2,bs,alpha_eft,alpha_eft_v,s2FoG,Apar,Aperp)

		return xiell[:,0], xiell[:,1], xiell[:,2]




	def get_tpcf(self,
			r: np.array,
			b1: float, 
			b2: float,
			bs: float,
			alpha_eft: float):
		'''

		Reads EFT results for the correlation function.

		Args:
			b1: 1st Lagrangian bias parameter.
			b2: 2nd Lagrangian bias parameter.
			bs: shear dependence of the bias
			alpha_eft: EFT counterterm

		Returns:

		'''
		r_min = np.min(r)
		r_max = np.max(r)

		r_range = (self.r >= r_min) & (self.r <= r_max)

		d2xiLin = 0
		return (self.xifile[:,2] + b1 * self.xifile[:,3] + b2 * self.xifile[:,4] + (b1**2)*self.xifile[:,5] + \
				b1*b2 * self.xifile[:,6] + (b2**2)*self.xifile[:,7] + alpha_eft *self.xifile[:,8] + \
				d2xiLin * self.xifile[:,9] + bs * self.xifile[:,10] + b1*bs * self.xifile[:,11] + \
				b2*bs* self.xifile[:,12] + bs**2*self.xifile[:,13])[r_range]



	def get_v12(self,
			r: np.array,
			b1: float, 
			b2: float,
			bs: float,
			alpha_eft: float,
			alpha_eft_v: float):
		'''

		Reads EFT results for mean pairwise velocity

		Args:
			b1: 1st Lagrangian bias parameter.
			b2: 2nd Lagrangian bias parameter.
			bs: shear dependence of the bias
			alpha_eft: EFT counterterm
			alpha_eft_v: EFT counterterm for velocity only

		Returns:

		'''
		r_min = np.min(r)
		r_max = np.max(r)

		r_range = (self.r >= r_min) & (self.r <= r_max)


		alpha_eft_prime_v = 0

		return (self.linear_growth*( self.v12file[:,2] + b1*self.v12file[:,3] + b2*self.v12file[:,4] + \
				(b1**2)*self.v12file[:,5] + b1*b2*self.v12file[:,6] + b2**2*self.v12file[:,7] + \
				alpha_eft_v*self.v12file[:,8] + alpha_eft_prime_v*self.v12file[:,9] + bs*self.v12file[:,10] + \
				bs*b1*self.v12file[:,11]) /(1.+ self.get_tpcf(self.r, b1, b2, bs, alpha_eft)))[r_range]


	def get_s12(self,
				b1: float, 
				b2: float, 
				bs: float,
				alpha_eft: float,
				alpha_eft_v: float,
				s_Fog:float= 0.):
		'''

		Reads EFT results for mean pairwise velocity

		Args:
			b1: 1st Lagrangian bias parameter.
			b2: 2nd Lagrangian bias parameter.
			bs: shear dependence of the bias
			alpha_eft: EFT counterterm
			alpha_eft_v: EFT counterterm for velocity only
			beta_eft_sigma: EFT counterterm for variance 

		Returns:
			xiell: Array containing the multipoles of the two point
			correlation function

		'''
		beta_eft_sigma = 0.

		s12_par_EFT = self.linear_growth**2*( self.s12file[:,1] + b1*self.s12file[:,2] + b2*self.s12file[:,3] + \
				(b1**2)*self.s12file[:,4] +  \
				bs*self.s12file[:,7] + beta_eft_sigma*self.s12file[:,8]) /(1.+ self.get_tpcf(self.r, b1, b2, bs, alpha_eft))

		s12_perp_EFT = self.linear_growth**2*( self.s12file[:,9] + b1*self.s12file[:,10] + b2*self.s12file[:,11] + \
				(b1**2)*self.s12file[:,12] + \
				bs*self.s12file[:,15] + beta_eft_sigma*self.s12file[:,16]) /(1.+ self.get_tpcf(self.r, b1, b2, bs, alpha_eft))


		return s12_par_EFT + s_Fog, s12_perp_EFT + s_Fog



