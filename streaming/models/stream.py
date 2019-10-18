import numpy as np
from halotools.mock_observables import tpcf_multipole
from streaming.models.skewt.moments2pdf import moments2skewt
from streaming.models.gaussian.moments2pdf import moments2gaussian
from streaming.models.gaussian.bestfit_gaussian import bestfit_gaussian 
from streaming.models.skewt.bestfit_skewt import bestfit_skewt 
from streaming.models.measured.measured_pdf import measured_los_pdf 
from streaming.integral import real2redshift
import copy
from operator import attrgetter


class Stream():

	def __init__(self, sim_measurement: None, model: str, best_fit_moments: None =None, method:  str = 'moments'):
		'''Initializes class for different streaming models.

		Args: 
			sim_measurement: instance of class Measured
			model: string
			either simulation, gaussian or skewt.
			best_fit_moments: dict 
			dictionary containing the functional form and optimal parameters for the fitted
			moments
			method: either moments method or maximum_likelihood to estimate the model parameters
					given the simulation measurment

		'''

		self.sim_measurement = sim_measurement

		self.s = np.arange(0., 50., 1.)
		self.s_c = 0.5 * (self.s[1:] + self.s[:-1])
		self.mu =  np.sort(1 - np.geomspace(0.0001, 1., 120))
		self.mu_c = 0.5 * (self.mu[1:] + self.mu[:-1])

		self.best_fit_moments = best_fit_moments

		if model == 'measured':
			if self.best_fit_moments is not None:
					self.sim_measurement_best_fit = self.use_fitted_moments(self.best_fit_moments)

			self.jointpdf_los = measured_los_pdf(self.sim_measurement)
			self.label = 'N-body'

		elif model	== 'gaussian':

			if method == 'moments':
				if self.best_fit_moments is not None:
					self.sim_measurement_best_fit = self.use_fitted_moments(self.best_fit_moments)
					self.jointpdf_los = moments2gaussian(self.sim_measurement_best_fit)
				else:
					self.jointpdf_los = moments2gaussian(self.sim_measurement)

				self.label = 'Gaussian-Moments'

			elif method == 'maximum_likelihood':
				self.jointpdf_los = bestfit_gaussian(self.sim_measurement)
				self.label = 'Gaussian-ML'


		elif model == 'skewt':

			if method == 'moments':

				if self.best_fit_moments is not None:
					self.sim_measurement_best_fit = self.use_fitted_moments(self.best_fit_moments)
					self.jointpdf_los = moments2skewt(self.sim_measurement_best_fit)
				else:
					self.jointpdf_los = moments2skewt(self.sim_measurement)

				self.label = 'ST-Moments'
			elif method == 'maximum_likelihood':
				self.jointpdf_los = bestfit_skewt(self.sim_measurement)
				self.label = 'ST-ML'
			
		else:

			raise NotImplementedError("That Streaming Model is not implement, try one of: \
					measured, gaussian or skewt.")

		self.monopole, self.quadrupole, self.hexadecapole = self.multipoles(self.s, self.mu)

		if model == 'measured':
			r_parallel_measured = np.arange(-70.5,70.5,1.)
			r_parallel_measured = np.sort(np.concatenate((r_parallel_measured, np.array([-0.0001, 0.0001]))))

			self.integrand, self.pi_sigma = self.pi_sigma(self.s_c, self.s_c, n = 0, r_parallel = r_parallel_measured)

		else:
			self.integrand, self.pi_sigma = self.pi_sigma(self.s_c, self.s_c, n = 300, r_parallel = None)



	def multipoles(self, s: np.array, mu: np.array): 

		if self.best_fit_moments:
			self.s_mu = real2redshift.simps_integrate(s, mu, self.sim_measurement_best_fit.tpcf.mean, self.jointpdf_los)
		else:
			self.s_mu = real2redshift.simps_integrate(s, mu, self.sim_measurement.tpcf.mean, self.jointpdf_los)

		monopole = tpcf_multipole(self.s_mu, mu, order = 0)
		quadrupole = tpcf_multipole(self.s_mu, mu, order = 2)
		hexadecapole = tpcf_multipole(self.s_mu, mu, order = 4)
		#self.wedges = tpcf_wedges(self.s_mu, mu)

		return monopole, quadrupole, hexadecapole

	def pi_sigma(self, s_perp: np.array, s_parallel: np.array, n: int, r_parallel: np.array):

		integrand, tpcf_pi_sigma = real2redshift.simps_integrate_pi_sigma(s_perp, s_parallel,
				self.sim_measurement.tpcf.mean, self.jointpdf_los, n = n, r_parallel = r_parallel)

		return integrand, tpcf_pi_sigma


	def use_fitted_moments(self,  best_fit_moments):

		sim_measurement_best_fit = copy.deepcopy(self.sim_measurement)

		'''
		for key in best_fit_moments:
			get_atr = attrgetter(f'{key}')
			print('before')
			print(sim_measurement_best_fit.m_10.mean(sim_measurement_best_fit.r)[0])
			print(sim_measurement_best_fit.c_20.mean(sim_measurement_best_fit.r)[0])
			print(sim_measurement_best_fit.c_02.mean(sim_measurement_best_fit.r)[0])
	
			mean =	lambda r: best_fit_moments[key]['function'](r, 
					*best_fit_moments[key]['popt']) 

			print('after')
			print(sim_measurement_best_fit.m_10.mean(sim_measurement_best_fit.r)[0])
			print(sim_measurement_best_fit.c_20.mean(sim_measurement_best_fit.r)[0])
			print(sim_measurement_best_fit.c_02.mean(sim_measurement_best_fit.r)[0])
		'''
		if 'tpcf' in best_fit_moments.keys():
			sim_measurement_best_fit.tpcf.mean = lambda r: best_fit_moments['tpcf']['function'](r,
					*best_fit_moments['tpcf']['popt'])

		if 'm_10' in best_fit_moments.keys():
			sim_measurement_best_fit.m_10.mean = lambda r: best_fit_moments['m_10']['function'](r,
					*best_fit_moments['m_10']['popt'])
			sim_measurement_best_fit.c_20.mean = lambda r: best_fit_moments['c_20']['function'](r,
					*best_fit_moments['c_20']['popt'])
			sim_measurement_best_fit.c_02.mean = lambda r: best_fit_moments['c_02']['function'](r,
					*best_fit_moments['c_02']['popt'])

		if 'c_30' in best_fit_moments.keys():
			sim_measurement_best_fit.c_30.mean = lambda r: best_fit_moments['c_30']['function'](r,
				*best_fit_moments['c_30']['popt'])
			sim_measurement_best_fit.c_12.mean = lambda r: best_fit_moments['c_12']['function'](r,
				*best_fit_moments['c_12']['popt'])
			sim_measurement_best_fit.c_22.mean = lambda r: best_fit_moments['c_22']['function'](r,
				*best_fit_moments['c_22']['popt'])
			sim_measurement_best_fit.c_40.mean = lambda r: best_fit_moments['c_40']['function'](r,
				*best_fit_moments['c_40']['popt'])
			sim_measurement_best_fit.c_04.mean = lambda r: best_fit_moments['c_04']['function'](r,
				*best_fit_moments['c_04']['popt'])

		return sim_measurement_best_fit


		


