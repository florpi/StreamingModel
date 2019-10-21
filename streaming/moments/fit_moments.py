import numpy as np
from scipy.optimize import curve_fit
from streaming.models.inputs import SimulationMeasurements
import pickle



class AnalyticalMoment():

	def __init__(self, function, r_to_fit, moment_to_fit, p0 = None, tpcf = False):

		self.function = function
		if tpcf:
			self.popt, self.pcov = curve_fit(log_tpcf, r_to_fit, np.log(moment_to_fit), p0)
		else:
			self.popt, self.pcov = curve_fit(function, r_to_fit, moment_to_fit, p0)


def tpcf(r, a, b):
	return (r/a)**(-b)

def log_tpcf(r, a, b):
	return np.log(tpcf(r, a, b,))


def m_10(r, a, b, c):

	return -a * np.exp(-b * r) + c

def d_m_10(r, a, b, c):

	return a *b * np.exp(-b * r) 


def c_20(r, a, b, c):

	return - a *np.exp(-b * r) + c

def c_02(r, a, b, c):

	return a * r**c + b

def c_30(r, a, b,c):
	
	return a/np.sqrt(r) + b * np.log10(r)  + c

def c_12(r, a, b,c):

	return a/np.sqrt(r) + b * np.log10(r)  + c

def c_22(r, a, b,c):
	
	return a/np.sqrt(r) + b * np.log10(r)  + c


def c_40(r, a, b,c):
	
	return a/np.sqrt(r) + b * np.log10(r)  + c

def c_04(r, a, b,c):

	return a/np.sqrt(r) + b * np.log10(r)  + c

if __name__ == "__main__":

	PDF_FILENAME = "../data/pairwise_velocity_pdf.hdf5"
	TPCF_FILENAME = "../data/tpcf.hdf5"

	simulation = SimulationMeasurements(PDF_FILENAME, TPCF_FILENAME)

	r_tpcf_threshold = (simulation.r_tpcf > 1) & (simulation.r_tpcf < 60)
	tpcf_best_fit = AnalyticalMoment(tpcf, simulation.r_tpcf[r_tpcf_threshold], 
			simulation.tpcf.mean(simulation.r_tpcf[r_tpcf_threshold]), tpcf = True)

	m_10_best_fit = AnalyticalMoment(m_10, simulation.r[15:100], simulation.m_10.mean(simulation.r[15:100]))
	c_20_best_fit = AnalyticalMoment(c_20, simulation.r[5:60], simulation.c_20.mean(simulation.r[5:60]),
									p0 = (3., 0.01, 0))
	c_02_best_fit = AnalyticalMoment(c_02, simulation.r[5:60], simulation.c_02.mean(simulation.r[5:60]))
	c_30_best_fit = AnalyticalMoment(c_30, simulation.r[5:60], simulation.c_30.mean(simulation.r[5:60]))
	c_12_best_fit = AnalyticalMoment(c_12, simulation.r[5:60], simulation.c_12.mean(simulation.r[5:60]))
	c_22_best_fit = AnalyticalMoment(c_22, simulation.r[5:60], simulation.c_22.mean(simulation.r[5:60]))
	c_40_best_fit = AnalyticalMoment(c_40, simulation.r[5:60], simulation.c_40.mean(simulation.r[5:60]))
	c_04_best_fit = AnalyticalMoment(c_04, simulation.r[5:60], simulation.c_04.mean(simulation.r[5:60]))


	best_fit_moments = {
			'tpcf': {'function': tpcf, 'popt': tpcf_best_fit.popt},
			'm_10': {'function': m_10, 'popt': m_10_best_fit.popt},
			'c_20': {'function': c_20, 'popt': c_20_best_fit.popt},
			'c_02': {'function': c_02, 'popt': c_02_best_fit.popt},
			'c_30': {'function': c_30, 'popt': c_30_best_fit.popt},
			'c_12': {'function': c_12, 'popt': c_12_best_fit.popt},
			'c_22': {'function': c_22, 'popt': c_22_best_fit.popt},
			'c_40': {'function': c_40, 'popt': c_40_best_fit.popt},
			'c_04': {'function': c_04, 'popt': c_04_best_fit.popt},
			}

	with open('best_fit_moments.pkl', 'wb') as f:
		pickle.dump(best_fit_moments, f)



