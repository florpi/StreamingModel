import numpy as np
import matplotlib.pyplot as plt
from halotools.mock_observables import tpcf_multipole
import math
from scipy.special import binom


def multipoles(mu, s_mu):
	'''

	Computes monopole, quadrupole and hexadecapole of the redshift space correlation function

	Args:
		s : array of s distances in Mpc/h
		mu : array of mu angular bins (between 0 and 1)

	Returns:

		monopole: monopole evaluated at the given s bins
		quadrupole: quadrupole evaluated at the given s bins
		hexadecapole: hexadecapole evaluated at the given s bins

	'''

	monopole = tpcf_multipole(s_mu, mu, order = 0)
	quadrupole = tpcf_multipole(s_mu, mu, order = 2)
	hexadecapole = tpcf_multipole(s_mu, mu, order = 4)

	return monopole, quadrupole, hexadecapole



def derivative(f, a, n = 1, args = (), h = 1.e-2, cutoff_n = None):
	'''

	Computes numerical derivative at a given order of the function f evaluated at a

	Args:
		f: function to take derivative of.
		a: value where the derivative is evaluated.
		n: order of the derivative.
		args: extra arguments to function f.
		h: step length in derivative.
		cutoff_n: n-th order for which to set derivatives to zero. 

	Returns:

		value of the derivative at the evalution points a.

	'''
	
	if cutoff_n and n >= cutoff_n:
		return np.zeros_like(a)

	else:
		if n == 0:

			return f(a, *args)

		elif n == 1:
			return (f(a + h, *args) - f(a - h, *args))/(2*h)

		elif n == 2:
					return (f(a + h, *args) - 2 * f(a, *args) + f(a - h, *args))/h**2

		elif n == 3:
			return (- 0.5 * f(a - 2* h , *args) + f(a - h, *args) - f(a + h, *args) + 0.5 *f(a + 2*h, *args)  )/h**3

		elif n == 4:
			return ( f(a + 2*h, *args) - 4. * f(a + h, *args) + \
					6. * f(a, *args) - 4. * f(a - h, *args) + f(a - 2* h, *args) ) / h**4
		else:
			raise ValueError("Too high order, not implemented.")


def derivative_product_rule( f, g, a,  n, args_f = (), args_g = (), 
		cutoff_f = None, cutoff_g = None):
	'''

	Computes the n-th derivative of a product of two functions: f.g, evaluated at a.

	Args:
		f: function 1.
		g: function 2.
		a: value where the derivative is evaluated.
		n: order of the derivative.
		args_f: tuple of arguments for function f.
		args_g: tuple of arguments for function g.
		cutoff_f: maximum order of the f derivative.
		cutoff_g: maximum order of the g derivative.

	Returns:
		Sum of all terms in the product rule

	'''

	return np.sum(
			[ binom(n, k)*derivative(f, a, n = (n - k), args = args_f, cutoff_n = cutoff_f) * \
			derivative(g, a, n = k, args = args_g, cutoff_n = cutoff_g) for k in range(n + 1)],
	axis = 0 
	 )




class TaylorExpansion:
	'''

	Perform a Taylor Expansion of the streaming model integral up to a given order

	Args:
		moments_df: Pandas data frame containing dictionaries for the different moments.
		order: integer defining the order of the expansion.
		cutoff_tpcf: integer defining the maximum order of the two point correlation function 
		derivatives to include in the expansion.
		color: color used for plotting.


	'''
	def __init__(self, s , mu, moments_df, order, cutoff_tpcf = None, color = None):

		self.color = color

		self.moments_df = moments_df

		self.s = s
		self.mu = mu
		
		self.s_c = 0.5 * (self.s[1:] + self.s[:-1])
		self.mu_c = 0.5 * (self.mu[1:] + self.mu[:-1])

		MU, S = np.meshgrid(self.mu_c, self.s_c)

		self.s_parallel = S * MU
		self.s_perpendicular = S * np.sqrt( 1 - MU**2 )

		# 0-th order term
		self.tpcf_s = self.tpcf_real(self.s_parallel, self.s_perpendicular)

		counter_order = 1

		while counter_order <= order:

			moment_counter_order = self.select_moment(counter_order)

			self.tpcf_s += (-1)**counter_order * 1./math.factorial(counter_order) * \
					derivative_product_rule(self.one_plus_tpcf_real, moment_counter_order, self.s_parallel, counter_order, 
					args_f = (self.s_perpendicular, ), args_g = (self.s_perpendicular, ), 
					cutoff_f = cutoff_tpcf)
			counter_order += 1

		self.mono, self.quad, self.hexa = multipoles(self.mu, self.tpcf_s)


	def tpcf_real(self, s_parallel, s_perpendicular):
		'''

		Computes the two point correlation function on all combinations of s_perp and s_parallel

		Args:
			s_perpendicular: array of distances perpendicular to line of sight.
			s_parallel: array of distances parallel to line of sight.

		Returns:
			2D array with the two point correlation function evaluated at the grid defined by s_perpendicular and s_parallel

		'''

		s_perpedicular = np.atleast_2d(s_perpendicular)
		s_parallel = np.atleast_2d(s_parallel)
		s = np.sqrt(s_perpendicular**2 + s_parallel**2)

		return self.moments_df['tpcf']['function'](s, *self.moments_df['tpcf']['popt'])

	def one_plus_tpcf_real(self, s_parallel, s_perpendicular):
		'''
		Computes one plus the two point correlation function on all combinations of s_perp and s_parallel

		Args:
			s_perpendicular: array of distances perpendicular to line of sight.
			s_parallel: array of distances parallel to line of sight.

		Returns:
			2D array with one plus the two point correlation function evaluated at the grid defined by s_perpendicular and s_parallel

		'''

		return 1. + self.tpcf_real(s_parallel, s_perpendicular)


	def moments_with_symmetries(self, s, r_order, t_order, mode):
		'''
		Apply symmetries to the moments of the radial and tangential velocity field.

		Args:
			s: array of pair distances.
			r_order: order of the moment for the radial velocity.
			t_order: order of the moment for the tangential velocity.
			mode: if central moments use c, if moments about the origin use m
		Returns:
			array of moments evaluated at s with the given order.
		'''

		# TODO: Shouldnt give zeroes but skip that element in the loop !
		if (t_order%2 !=0):
			# Due to isotropy all momens with t_order odd vanish
			return np.zeros_like(s)

		elif (r_order == 0) and (t_order == 0):
			# The PDF is normalised
			return np.ones_like(s)

		elif (mode == 'c') and (r_order + t_order == 1):
			# The first order central moments are zero
			return np.zeros_like(s)

		else:
			return self.moments_df[f'{mode}_{r_order}{t_order}']['function'](s, 
					*self.moments_df[f'{mode}_{r_order}{t_order}']['popt'])


			
	def project_moment_to_los(self, s_parallel, s_perpendicular, n, mode = 'c'):
		'''
		Project the moments of the radial and tangential velocity field onto the line of sight moments.

		Args:
			s_perpendicular: array of distances perpendicular to line of sight.
			s_parallel: array of distances parallel to line of sight.
			n: order of the moment.
			mode: Type of moment. If central moments use c, if moments about the origin use m.
		Returns:
			2D n-th moment of the line of sight velocity PDF evaluated at s_parallel and s_perpendicular.
			
		'''

		s_perpedicular = np.atleast_2d(s_perpendicular)
		s_parallel = np.atleast_2d(s_parallel)

		s = np.sqrt(s_parallel**2 + s_perpendicular**2)
		mu = s_parallel/s


		return np.sum(
				[binom(n,k) * mu**k * np.sqrt(1 - mu**2)**(n-k) * \
					self.moments_with_symmetries(s, k, n-k, mode) for k in range(n + 1)],
				axis = 0)



	def central2origin(self, s_parallel, s_perpendicular, n):
		'''
		Translates central moments into moments about the origin

		Args:
			s_perpendicular: array of distances perpendicular to line of sight.
			s_parallel: array of distances parallel to line of sight.
			n: order of the moment.
		Returns:
			2D n-th moment about the origin of the line of sight velocity Pdf evaluated at s_parallel and
			s_perpendicular.

		'''
			

		return np.sum(
				[ binom(n,k) * self.project_moment_to_los(s_parallel, s_perpendicular, k) * \
						self.project_moment_to_los(s_parallel, s_perpendicular, 1, mode = 'm')**(n - k) \
						for k in range(0, n + 1)],
				axis = 0)

	
	def select_moment(self, n):
		'''
		Wrapper around central2origin function to return the moment about the origin with only s_parallel
		and s_perpendicular as arguments.

		Args:
			n: order of the moment.
		Returns:
			Function to compute the n-th moment about the origin of the line of sight velocity PDF.
		'''

		def moment_about_origin(s_parallel, s_perpendicular):

			return self.central2origin(s_parallel, s_perpendicular, n)

		return moment_about_origin 

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

		self.mono, self.quad, self.hexa = self.multipoles(self.s_c, 0)

		counter_order = 1

		while counter_order <= order:

			mono_order, quad_order, hexa_order = self.multipoles(self.s_c, order = counter_order)

			self.mono += mono_order
			self.quad += quad_order
			self.hexa += hexa_order

			counter_order += 1



	def multipoles(self, s, order):

		if order == 0:

			monopole = self.moments_df['tpcf']['function'](s, *self.moments_df['tpcf']['popt'])
			quadrupole = np.zeros_like(s)
			hexadecapole = np.zeros_like(s)

			return monopole, quadrupole, hexadecapole

		elif order == 1:

			b_term = - self.moments_df['d1_tpcf']['function'](s, *self.moments_df['tpcf']['popt']) * \
					self.moments_df['m_10']['function'](s, *self.moments_df['m_10']['popt'])

			
			self.mono_b_term = 1./3. * b_term
			self.quad_b_term = 2./3. * b_term
			self.hexa_b_term = np.zeros_like(s)


			c_term = -(1 + self.moments_df['tpcf']['function'](s, *self.moments_df['tpcf']['popt'])) * \
					self.moments_df['m_10']['function'](s, *self.moments_df['m_10']['popt'])/s


			self.mono_c_term = 2./3. * c_term
			self.quad_c_term = -2./3. * c_term
			self.hexa_c_term = np.zeros_like(s)

			d_term =  -(1 + self.moments_df['tpcf']['function'](s, *self.moments_df['tpcf']['popt'])) * \
					self.moments_df['d1_m_10']['function'](s, *self.moments_df['m_10']['popt'])


			self.mono_d_term = 1./3. * d_term
			self.quad_d_term = 2./3. * d_term
			self.hexa_d_term = np.zeros_like(s)


			monopole = self.mono_b_term + self.mono_c_term + self.mono_d_term
			quadrupole = self.quad_b_term + self.quad_c_term + self.quad_d_term
			hexadecapole = self.hexa_b_term + self.hexa_c_term + self.hexa_d_term

			return monopole, quadrupole, hexadecapole





