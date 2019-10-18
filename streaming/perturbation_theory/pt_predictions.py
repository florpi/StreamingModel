import numpy as np
from scipy.optimize import curve_fit
from clpt import CLPT
from eft import EFT 




class PerturbationTheory():
	'''
	Class to analyze predictions for Convolutional Lagrangian PT and Effective
	Field Theory

	'''

	def __init__(self, 
				data_path: str,
				linear_pk_file: str,
				linear_growth: float,
				simulation_tpcf = None,
				r_min: float = 30.,
				):
		'''
		Args:
			data_path: path where all results from both codes are stored.
			linear_pk_file: filename containing the linear power spectrum
			and its k range.
			linear_growth: linear prediction for the growth factor.
			simulation_tpcf: function with the interpolated real space tpcf
			measured in a simulation. Used to fit free parameters.
			r_min: minimum r to use when fitting the tpcf to find bias parameters. (In Mpc/h)

		'''

		self.simulation_tpcf = simulation_tpcf
		r_max = 150.
		
		# **************** CLPT ********************************************************#
		self.clpt = CLPT(data_path, linear_growth)
		if simulation_tpcf is not None:
			print('Fitting free parameters to the simulation tpcf...')
			self.fit_pt_parameters(self.clpt, r_min, r_max)

			print(f'Found best fit bias parameters: (b1 = {self.clpt.b1}, b2 = {self.clpt.b2})')
				
			self.clpt.tpcf = self.clpt.get_tpcf(self.clpt.r, self.clpt.b1, self.clpt.b2)
			self.clpt.v12 = self.clpt.get_v12(self.clpt.b1, self.clpt.b2)
			self.clpt.s12_par, self.clpt.s12_perp = self.clpt.get_s12(self.clpt.b1, self.clpt.b2)

		# **************** EFT ********************************************************#
		self.eft = EFT(data_path, linear_pk_file, linear_growth, run = True)
		if simulation_tpcf is not None:
			print('Fitting free parameters to the simulation tpcf...')
			self.fit_pt_parameters(self.eft, r_min, r_max)

			print(f'Found best fit bias parameters: (b1 = {self.eft.b1}, b2 = {self.eft.b2}, bs = {self.eft.bs})')
			print(f'Found best fit alpha_eft parameter: alpha_eft = {self.eft.alpha_eft}')

			self.eft.tpcf = self.eft.get_tpcf(self.eft.r, self.eft.b1, self.eft.b2, self.eft.bs, self.eft.alpha_eft)
			self.eft.v12 = self.eft.get_v12(self.eft.b1, self.eft.b2, self.eft.bs, self.eft.alpha_eft, 0.)
			self.eft.s12_par, self.eft.s12_perp = self.eft.get_s12(self.eft.b1, self.eft.b2, self.eft.bs, self.eft.alpha_eft,
					0.)


			

	def fit_pt_parameters(self, perturbation_theory, 
								r_min, r_max):
		'''
		Fit the free parameters to the tpcf measured in the simulation.
		Args:
			perturbation_theory: object from either clpt or eft class.

		'''
		b1_initial = 0.3
		b2_initial = 0.
		b_s_initial = -2/7 * (b1_initial -1.)
		alpha_eft_initial = -1.
		p0 = (b1_initial, b2_initial)
		if isinstance(perturbation_theory, EFT):
			p0 = (b1_initial, b2_initial, b_s_initial, alpha_eft_initial)

		r_range = (perturbation_theory.r > r_min) & (perturbation_theory.r < r_max)
	
		popt, pcov = curve_fit(perturbation_theory.get_tpcf, 
				perturbation_theory.r[r_range], self.simulation_tpcf(perturbation_theory.r[r_range]), 
				p0 = p0)

		perturbation_theory.b1 = popt[0]
		perturbation_theory.b2 = popt[1]

		if isinstance(perturbation_theory, EFT):
			perturbation_theory.alpha_eft = popt[2]
			perturbation_theory.bs= popt[3]

