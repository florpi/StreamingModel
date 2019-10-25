import numpy as np
from scipy.optimize import curve_fit
import pickle
import copy
from streaming.models.inputs import SimulationMeasurements
from streaming.perturbation_theory.clpt import CLPT 
from streaming.perturbation_theory.eft import EFT 
from streaming.moments.fit_config import *

#********** Class to fit and store parameters **************************#
class FitFunction():

	def __init__(self, function, x, y, p0 = None,
			derivative_function = None,
			second_derivative_function = None):

		self.function = function
		self.x = x
		self.y = y
		self.fit(p0)
		self.derivative_function = derivative_function
		self.second_derivative_function = second_derivative_function

	def eval(self, x):

		return self.function(x, *self.popt)

	def derivative(self, x, n):

		if n == 1:
			return self.derivative_function(x, *self.popt)
		elif n==2:
			return self.second_derivative_function(x, *self.popt)
		else:
			return NotImplementedError


	def fit(self, p0):

		if self.function.__name__ == 'power_law':
			self.popt, self.pcov = curve_fit(log_power_law, 
										self.x, np.log(self.y), p0 = p0)


		else:
			self.popt, self.pcov = curve_fit(self.function, 
										self.x, self.y, p0 = p0)

	
#********** Class to fit all elements of the streaming model **************************#


class AnalyticalIngredients():

	def __init__(self, simulation_measurement, moments_list, 
					perturbation_theory = False, derivative_list = None, ):

		self.data_path = '/cosma/home/dp004/dc-cues1/CLPT_GSRSD/data/'

		self.simulation_measurement = simulation_measurement
		self.moments_list = moments_list
		self.derivative_list = derivative_list
		self.perturbation_theory = perturbation_theory
		self.analytical_moments_dict = {}

		if not perturbation_theory:
			self.fit_moments()

		else:
			self.load_pt_parameters()

	def load_pt_parameters(self):
		# load free parameters
		with open(f'../perturbation_theory/{self.perturbation_theory}_params.pkl', 'rb') as f:
			pt_parameters = pickle.load(f)

		if self.perturbation_theory == 'clpt':
			pt = CLPT(self.data_path,
					**pt_parameters)
		elif self.perturbation_theory == 'eft':
			pt = EFT(self.data_path,
					linear_pk_file = 'linear_pow.txt',
					**pt_parameters,
					run = False)

		else:
			return NotImplementedError

		for moment in self.moments_list:
			self.analytical_moments_dict[moment] = getattr(pt, moment)


	def fit_moments(self):

		with open('fitting_config.pickle', 'rb') as f:
			moments_config = pickle.load(f)

		for moment in self.moments_list:

			if moment == 'tpcf':
				r = self.simulation_measurement.r_tpcf
			else:
				r = self.simulation_measurement.r
	
			fit_range = (r > moments_config[moment]['fit_range'][0]) & \
					(r < moments_config[moment]['fit_range'][1])


			self.analytical_moments_dict[moment] =  FitFunction(moments_config[moment]['function'],
									r[fit_range],
									getattr(self.simulation_measurement, moment).mean(r[fit_range]),
									p0 = moments_config[moment]['p0'],
									derivative_function = moments_config[moment]['derivative'], 
									second_derivative_function = moments_config[moment]['second_derivative'])


if __name__ == "__main__":

	PDF_FILENAME = "../data/pairwise_velocity_pdf.hdf5"
	TPCF_FILENAME = "../data/tpcf.hdf5"

	simulation = SimulationMeasurements(PDF_FILENAME, TPCF_FILENAME)

	fit = AnalyticalIngredients(simulation, ['tpcf','m_10', 'c_20'], derivative_list = ['d_m_10'])
	print(fit.analytical_moments_dict['m_10'].derivative(simulation.r, n=1))

	print('m_10')
	print(fit.analytical_moments_dict['m_10'].eval(simulation.r))



	ai = AnalyticalIngredients(simulation, ['tpcf','m_10', 'c_20'], perturbation_theory = 'clpt')

	print('m_10 - clpt')
	print(ai.analytical_moments_dict['m_10'](simulation.r))


	ai = AnalyticalIngredients(simulation, ['tpcf','m_10', 'c_20'], perturbation_theory = 'eft')

	print('m_10 - eft')
	print(ai.analytical_moments_dict['m_10'](simulation.r))




