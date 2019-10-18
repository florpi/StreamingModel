import copy
import numpy as np
from streaming.models import stream
from scipy.optimize import fsolve


def increase(simulation, key, percentage):
	
	def moment(r):
		value = getattr(simulation, key).mean(r)
		return value * (1. + percentage)
	return moment

def decrease(simulation, key, percentage):
	
	def moment(r):
		value = getattr(simulation, key).mean(r)		
		return value * (1. - percentage)
	return moment

def compute_alpha_beta(percentage):
	'''
	Find alpha and beta that produce:
		- A variation of value percentage at 5 Mpc/h
		- A variation of 1% at 30 Mpc/h
	'''

	def constraints(values):
		alpha, beta = values

		return (alpha - percentage * 5 **beta, alpha - 0.01 * 30**beta)

	alpha, beta = fsolve(constraints, (1, 10))

	return alpha, beta
	

def gradual_increase(simulation, key, alpha, beta):
	
	def moment(r):
		value = getattr(simulation, key).mean(r)		
		return value * (1. + alpha/r**beta)
	
	return moment


def gradual_decrease(simulation, key, alpha, beta):
	   
	def moment(r):
		value = getattr(simulation, key).mean(r)		
		return value * (1. - alpha/r**beta)
	
	return moment

class ModifyMoment():
	def __init__(self, simulation, measured_moments, moment_key, percentage):

		alpha, beta = compute_alpha_beta(percentage)

		varied_moments = copy.deepcopy(measured_moments)
		
		varied_moments[moment_key]['function'] = increase(simulation, moment_key, percentage)
		self.increase = stream.Stream(simulation, 'skewt',
						best_fit_moments = varied_moments)

		varied_moments[moment_key]['function'] = decrease(simulation, moment_key, percentage)
		self.decrease = stream.Stream(simulation, 'skewt',
						best_fit_moments = varied_moments)


		varied_moments[moment_key]['function'] = gradual_increase(simulation, moment_key, alpha, beta)
		self.gradual_increase = stream.Stream(simulation, 'skewt',
						best_fit_moments = varied_moments)

		varied_moments[moment_key]['function'] = gradual_decrease(simulation, moment_key, alpha, beta)
		self.gradual_decrease = stream.Stream(simulation, 'skewt',
						best_fit_moments = varied_moments)




class ModifyMoments():
	def __init__(self, simulation, percentage, 
						moments_to_vary = ['tpcf', 'm_10', 'c_20', 'c_02']):

		measured_moments = {
					'tpcf': {'function': simulation.tpcf.mean, 'popt': ()},
					'm_10': {'function': simulation.m_10.mean, 'popt': ()},
					'c_20': {'function': simulation.c_20.mean, 'popt': ()},
					'c_02': {'function': simulation.c_02.mean, 'popt': ()},
					'c_30': {'function': simulation.c_30.mean, 'popt': ()},
					'c_12': {'function': simulation.c_12.mean, 'popt': ()},
					'c_22': {'function': simulation.c_22.mean, 'popt': ()},
					'c_40': {'function': simulation.c_40.mean, 'popt': ()},
					'c_04': {'function': simulation.c_04.mean, 'popt': ()},
		}

		for key in moments_to_vary:
			print(f'Varying {key}')
			setattr(self, key, ModifyMoment(simulation, measured_moments, key, percentage))
