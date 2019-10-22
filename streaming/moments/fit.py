import numpy as np
from scipy.optimize import curve_fit
import pickle
import copy
from streaming.models.inputs import SimulationMeasurements

#********** Fitting functions used **************************#
def power_law(r, a, b):
	return (r/a)**(-b)

def log_power_law(r, a, b):
	return np.log(power_law(r, a, b))

def derivative_power_law(r, a, b):
	return -b/r * (r/a)**(-b)


def exponential(r, a, b, c):
	return -a * np.exp(-b * r) + c

def derivative_exponential( r, a, b, c):
	return a *b * np.exp(-b * r) 

def second_derivative_exponential( r, a, b, c):
	return -a*b**2*np.exp(-b*r)

def high_order(r, a, b, c,):
	return a/np.sqrt(r) + b * np.log10(r)  + c


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

	def __init__(self, simulation_measurement, moments_list, derivative_list = None):

		moments_config = {
				'tpcf': {'function' : power_law,
						'derivative': derivative_power_law,
						'second_derivative': None,
						'fit_range': (1, 60),
						'p0': None,
						},
				'm_10': {'function' : exponential,
						'derivative': derivative_exponential,
						'second_derivative': second_derivative_exponential,
						'fit_range': (15, 100),
						'p0': None,
						},
				'c_20': {'function' : exponential,
						'derivative': derivative_exponential,
						'second_derivative': second_derivative_exponential,
						'fit_range': (5, 60),
						'p0': (3, 0.01, 0),
						},
				'c_02': {'function' : exponential,
						'derivative': derivative_exponential,
						'second_derivative': second_derivative_exponential,
						'fit_range': (5, 60),
						'p0': (3, 0.01, 0),
						},
				'c_12': {'function' : high_order,
						'fit_range': (5, 60),
						'derivative': None,
						'second_derivative': None,
						'p0': None,
						},
				'c_30': {'function' : high_order,
						'fit_range': (5, 60),
						'derivative': None,
						'second_derivative': None,
						'p0': None,
						},
				'c_22': {'function' : high_order,
						'fit_range': (5, 60),
						'derivative': None,
						'second_derivative': None,
						'p0': None,
						},
				'c_40': {'function' : high_order,
						'fit_range': (5, 60),
						'derivative': None,
						'second_derivative': None,
						'p0': None,
						},
				'c_04': {'function' : high_order,
						'fit_range': (5, 60),
						'derivative': None,
						'second_derivative': None,
						'p0': None,
						},

					}

		self.moment2best_fit = {}


		for moment in moments_list:

			if moment == 'tpcf':
				fit_range = (simulation_measurement.r_tpcf > moments_config[moment]['fit_range'][0]) & \
						(simulation_measurement.r_tpcf < moments_config[moment]['fit_range'][1])


				self.moment2best_fit[moment] =  FitFunction(moments_config[moment]['function'],
										simulation_measurement.r_tpcf[fit_range],
										getattr(simulation_measurement, moment).mean(simulation_measurement.r_tpcf[fit_range]),
										p0 = moments_config[moment]['p0'],
										derivative_function = moments_config[moment]['derivative'], 
										second_derivative_function = moments_config[moment]['second_derivative'])



			else:
				fit_range = (simulation_measurement.r > moments_config[moment]['fit_range'][0]) & \
						(simulation_measurement.r < moments_config[moment]['fit_range'][1])


				self.moment2best_fit[moment] =  FitFunction(moments_config[moment]['function'],
										simulation_measurement.r[fit_range],
										getattr(simulation_measurement, moment).mean(simulation_measurement.r[fit_range]),
										p0 = moments_config[moment]['p0'],
										derivative_function = moments_config[moment]['derivative'], 
										second_derivative_function = moments_config[moment]['second_derivative'])

if __name__ == "__main__":

	PDF_FILENAME = "../data/pairwise_velocity_pdf.hdf5"
	TPCF_FILENAME = "../data/tpcf.hdf5"

	simulation = SimulationMeasurements(PDF_FILENAME, TPCF_FILENAME)

	ai = AnalyticalIngredients(simulation, ['tpcf','m_10', 'c_20'], derivative_list = ['d_m_10'])
	print('d_m_10')
	print(ai.moment2best_fit['d_m_10'](simulation.r))






