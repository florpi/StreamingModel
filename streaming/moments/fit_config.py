import pickle
import numpy as np

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


with open('fitting_config.pickle', 'wb') as f:
	pickle.dump(moments_config, f, protocol = pickle.HIGHEST_PROTOCOL)
