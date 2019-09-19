import numpy as np
from scipy.special import binom
from sympy import Symbol, integrate

#TODO: Write all terms using class, prepare output summary 
# (order by order of magnitude in s?, mu dependence integrated)
class Term:

	def __init__(self, s, function_term):

		mu = Symbol('mu')
		s_dependence, mu_dependence = function_term(s, mu)

		self.s = s_dependence
		self.mu = mu_dependence

	def mu_multipoles(self):

		mu_monopole = integrate(self.mu, (mu, 0, 1))
		mu_quadrupole = 5./2. * integrate(self.mu * (3 * mu**2 - 1), (mu, 0, 1))
		mu_hexadecapole = 9./8. * integrate(self.mu * (35 * mu**4 - 30. * mu**2 + 3), (mu, 0, 1))

		return mu_monopole, mu_quadrupole, mu_hexadecapole




class AnalyticalTaylorExpansion:
	'''

	Perform a Taylor Expansion of the streaming model integral up to a given order

	Args:
		moments_df: Pandas data frame containing dictionaries for the different moments.
		order: integer defining the order of the expansion.
		cutoff_tpcf: integer defining the maximum order of the two point correlation function 
		derivatives to include in the expansion.
		color: color used for plotting.


	'''
	def __init__(self, s, moments_df, order): 

		self.moments_df = moments_df

		self.s = s
		
		self.s_c = 0.5 * (self.s[1:] + self.s[:-1])

		# 0-th order term

		self.monopole, self.quadrupole, self.hexadecapole = self.multipoles(self.s_c, 0)

		counter_order = 1

		while counter_order <= order:

			mono_order, quad_order, hexa_order = self.multipoles(self.s_c, order = counter_order)

			self.monopole += mono_order.astype(np.float32)
			self.quadrupole += quad_order.astype(np.float32)
			self.hexadecapole += hexa_order.astype(np.float32)

			counter_order += 1

	def d1tpcf_m1(self, s, mu):

		return -self.moments_df['d1_tpcf']['function'](s, *self.moments_df['tpcf']['popt']) * \
					self.moments_df['m_10']['function'](s, *self.moments_df['m_10']['popt']),  mu**2


	def tpcf_m1(self, s, mu):
		return  -(1 + self.moments_df['tpcf']['function'](s, *self.moments_df['tpcf']['popt'])) * \
					self.moments_df['m_10']['function'](s, *self.moments_df['m_10']['popt'])/s,  (1 - mu**2)


	def tpcf_d1m1(self, s, mu):
		return  -(1 + self.moments_df['tpcf']['function'](s, *self.moments_df['tpcf']['popt'])) * \
					self.moments_df['d1_m_10']['function'](s, *self.moments_df['m_10']['popt']),  mu**2

	def first_order(self, s):

		mu = Symbol('mu')

		s_d1tpcf_m1, mu_d1tpcf_m1 = self.d1tpcf_m1(s, mu)
		s_tpcf_m1, mu_tpcf_m1 = self.tpcf_m1(s, mu)
		s_tpcf_d1m1, mu_tpcf_d1m1 = self.tpcf_d1m1(s, mu)

		monopole, quadrupole, hexadecapole = s_d1tpcf_m1.reshape(1,-1) * self.mu_multipoles(mu, mu_d1tpcf_m1).reshape(-1,1) + \
				s_tpcf_m1.reshape(1,-1) * self.mu_multipoles(mu, mu_tpcf_m1).reshape(-1,1) + \
				s_tpcf_d1m1.reshape(1,-1) * self.mu_multipoles(mu, mu_tpcf_d1m1).reshape(-1,1)

		return monopole, quadrupole, hexadecapole

	def mu_multipoles(self, mu, mu_dependence):

		mu_monopole = integrate(mu_dependence, (mu, 0, 1))
		mu_quadrupole = 5./2. * integrate(mu_dependence * (3 * mu**2 - 1), (mu, 0, 1))
		mu_hexadecapole = 9./8. * integrate(mu_dependence * (35 * mu**4 - 30. * mu**2 + 3), (mu, 0, 1))

		return np.array([mu_monopole, mu_quadrupole, mu_hexadecapole])


	def multipoles(self, s, order):

		if order == 0:

			monopole = self.moments_df['tpcf']['function'](s, *self.moments_df['tpcf']['popt'])
			quadrupole = np.zeros_like(s)
			hexadecapole = np.zeros_like(s)

			return np.asarray(monopole), np.asarray(quadrupole), np.asarray(hexadecapole)

		elif order == 1:

			monopole, quadrupole, hexadecapole = self.first_order(s)

			return np.asarray(monopole), np.asarray(quadrupole), np.asarray(hexadecapole)





