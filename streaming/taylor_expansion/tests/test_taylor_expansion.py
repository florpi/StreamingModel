from taylor_expansion import multipoles, derivative, derivative_product_rule, TaylorExpansion
import numpy as np
import pandas as pd


def tpcf(r, a, b):
	return a*r**2 + b


def m1(r, a, b):

	return a*r**2 + b


def test_multipoles():

	mu =  np.sort(1 - np.geomspace(0.0001, 1., 400))
	s = np.arange(0.,50.,1)

	s_c = (0.5 * (s[1:] + s[:-1]) )
	mu_c = (0.5 * (mu[1:] + mu[:-1]))

	expected_mono = 1./3. * s_c


	s_mu = s_c.reshape(-1, 1) * (mu_c**2).reshape(1, -1)

	mono, quad, hexa = multipoles(mu, s_mu)

	np.testing.assert_allclose(mono, expected_mono, rtol= 1.e-3)

def test_derivative():
	a = 2.
	b = 10.


	def toy_tpcf(r):
		return  tpcf(r, a, b)


	r = np.linspace(0., 10., 10)

	np.testing.assert_allclose(derivative(toy_tpcf, r), 2 *a * r, rtol = 1.e-5)
	np.testing.assert_allclose(derivative(toy_tpcf, r, n = 2, cutoff_n = 2), np.zeros_like(r), rtol = 1.e-5)
	np.testing.assert_allclose(derivative(toy_tpcf, r, n = 2), 2 * a *np.ones_like(r), rtol = 1.e-4)

def test_derivative_product_rule():

	a = 5.
	b = 2.


	def toy_tpcf(r):
		return  tpcf(r, a, b)


	r = np.linspace(0., 10., 10)


	np.testing.assert_allclose(derivative_product_rule(toy_tpcf, toy_tpcf, r, 1),
		 								4 * a**2 * r**3 + 4 * a * r *b, rtol = 1.e-5)


def test_moment_projection():

	a = 1.
	b = 2.

	c = 10.
	d = 1.

	moments = {
				'tpcf': {'function': tpcf , 'popt': [a,b]},
				'm1r0t': {'function': m1, 'popt':[a,b] }, 

				'c2r0t': {'function': tpcf, 'popt':[a,b] },
				'c0r2t': {'function': tpcf, 'popt':[c,d] },

				'c3r0t': {'function': tpcf, 'popt':[a,b] },
				'c1r2t': {'function': tpcf, 'popt':[c,d] },

				'c2r2t': {'function': tpcf, 'popt':[a,b] },
				'c4r0t': {'function': tpcf, 'popt':[c,d] },
				'c0r4t': {'function': tpcf, 'popt':[a,b] },
				}

	moments_df = pd.DataFrame(moments)


	mu = np.linspace(0.,1.,10)
	s = np.arange(0.,50.,1)


	te = TaylorExpansion(s, mu, moments_df, 3)

	s_perp = np.arange(0.5, 50.5, 1.)

	s_par = s_perp.reshape(-1, 1)
	s_perp = s_perp.reshape(1, -1)

	assert te.tpcf_s.shape[0] == 49 
	assert te.tpcf_s.shape[1] == 9 

	mean = te.project_moment_to_los(s_par, s_perp, n = 1, mode = 'm')

	s = np.sqrt(s_perp.reshape(-1,1)**2 + s_perp.reshape(1,-1)**2)
	mu = s_perp/s

	expected_mean = tpcf(s, a, b) * mu

	np.testing.assert_almost_equal(mean, expected_mean.T)


	expected_m2 = tpcf(s, a, b) * mu **2 + tpcf(s,c,d) * (1 - mu**2)

	m2 = te.project_moment_to_los(s_par, s_perp, n = 2, mode = 'c')

	np.testing.assert_almost_equal(m2, expected_m2.T)

	expected_m3 = tpcf(s, a, b) * mu **3 + 3 * tpcf(s,c,d) * mu * (1 - mu**2)

	m3 = te.project_moment_to_los(s_par, s_perp, n = 3, mode = 'c')

	np.testing.assert_almost_equal(m3, expected_m3.T)

	expected_m4 = 6 * tpcf(s, a, b) * mu **2 * (1 - mu**2) + \
			+ tpcf(s,c,d) * mu**4 + tpcf(s, a, b) * (1 - mu**2)**2

	m4 = te.project_moment_to_los(s_par, s_perp, n = 4, mode = 'c')

	np.testing.assert_almost_equal(m4, expected_m4.T)

	
def test_central2origin():

	a = 1.
	b = 2.

	c = 10.
	d = 1.

	moments = {
				'tpcf': {'function': tpcf , 'popt': [a,b]},
				'm1r0t': {'function': m1, 'popt':[a,b] }, 
				'm0r1t': {'function': None, 'popt':None }, 

				'c2r0t': {'function': tpcf, 'popt':[a,b] },
				'c0r2t': {'function': tpcf, 'popt':[c,d] },

				'c3r0t': {'function': tpcf, 'popt':[a,b] },
				'c1r2t': {'function': tpcf, 'popt':[c,d] },

				'c2r2t': {'function': tpcf, 'popt':[a,b] },
				'c4r0t': {'function': tpcf, 'popt':[c,d] },
				'c0r4t': {'function': tpcf, 'popt':[a,b] },
				}

	moments_df = pd.DataFrame(moments)


	mu = np.linspace(0.,1.,10)
	s = np.arange(0.,50.,1)


	te = TaylorExpansion(s, mu, moments_df, 3)

	s_perp = np.arange(0.5, 50.5, 1.)

	s_par = s_perp.reshape(-1, 1)
	s_perp = s_perp.reshape(1, -1)

	m1_ = te.central2origin(s_par, s_perp, 1)

	expected_m1 = te.project_moment_to_los(s_par, s_perp, 1, mode = 'm')

	np.testing.assert_allclose( m1_, expected_m1, rtol = 1.e-5)

	m2 = te.central2origin(s_par, s_perp, 2)

	expected_m2 = m1_**2 + \
				te.project_moment_to_los(s_par, s_perp, 2, mode = 'c')

	np.testing.assert_allclose( m2, expected_m2, rtol = 1.e-5)

	m3 = te.central2origin(s_par, s_perp, 3)

	expected_m3 = -2 * m1_**3 + 3. * m2 * m1_ + \
			te.project_moment_to_los(s_par, s_perp, 3, mode = 'c')
	
	np.testing.assert_allclose( m3, expected_m3, rtol = 1.e-5)

	m4 = te.central2origin(s_par, s_perp, 4)

	expected_m4 =  3. * m1_**4 - 6. * m1_**2 * m2 + 4. * m1_ * m3 + te.project_moment_to_los(s_par, s_perp, 4, mode = 'c')

	np.testing.assert_allclose( m4, expected_m4, rtol = 1.e-5)


	
def test_zero_order():
	a = 1.
	b = 2.

	c = 10.
	d = 1.

	moments = {
				'tpcf': {'function': tpcf , 'popt': [a,b]},
				'm1r0t': {'function': m1, 'popt':[a,b] }, 
				'm0r1t': {'function': None, 'popt':None }, 

				}

	moments_df = pd.DataFrame(moments)


	mu = np.linspace(0.,1.,60)
	s = np.arange(0.,50.,1)

	te = TaylorExpansion(s, mu, moments_df, 0)

	s_c = 0.5 * (s[1:] + s[:-1])
	mu_c = 0.5 * (mu[1:] + mu[:-1])

	np.testing.assert_almost_equal(te.tpcf_s, np.tile(tpcf(s_c, a, b).reshape(-1,1), len(mu_c)))
	np.testing.assert_almost_equal(te.mono, tpcf(s_c, a, b))



def test_first_order_no_tpcf_derivatives():
	a = 1.
	b = 2.

	moments = {
				'tpcf': {'function': tpcf , 'popt': [a,b]},
				'm1r0t': {'function': m1, 'popt':[a,b] }, 
				'm0r1t': {'function': None, 'popt':None }, 

				}

	moments_df = pd.DataFrame(moments)


	mu =  np.sort(1 - np.geomspace(0.0001, 1., 120))
	s = np.arange(0.,50.,1)

	te = TaylorExpansion(s, mu, moments_df, 1, cutoff_tpcf = 1)

	s_c = (0.5 * (s[1:] + s[:-1]) ).reshape(-1, 1)
	mu_c = (0.5 * (mu[1:] + mu[:-1])).reshape(1,-1)

	expected_tpcf = tpcf(s_c, a,b) + (1 + tpcf(s_c, a,b)) *\
			(- (1 - mu_c**2)/s_c* m1(s_c, a, b)\
			- mu_c**2 * 2 * a* s_c) 


	np.testing.assert_allclose(te.tpcf_s, expected_tpcf, rtol= 1.e-3)

	expected_mono = tpcf(te.s_c, a, b) - (1 + tpcf(te.s_c, a, b)) * (2./3. * m1(te.s_c, a, b )/te.s_c +  2./3. * a *te.s_c)

	np.testing.assert_allclose(te.mono, expected_mono, rtol= 1.e-3)


def test_first_order():
	a = 1.
	b = 2.


	moments = {
				'tpcf': {'function': tpcf , 'popt': [a,b]},
				'm1r0t': {'function': m1, 'popt':[a,b] }, 
				'm0r1t': {'function': None, 'popt':None }, 

				}

	moments_df = pd.DataFrame(moments)


	mu =  np.sort(1 - np.geomspace(0.0001, 1., 120))
	s = np.arange(0.,50.,1)

	te = TaylorExpansion(s, mu, moments_df, 1, cutoff_tpcf = 2)

	s_c = (0.5 * (s[1:] + s[:-1]) ).reshape(-1, 1)
	mu_c = (0.5 * (mu[1:] + mu[:-1])).reshape(1,-1)

	expected_tpcf = tpcf(s_c, a,b) + (1 + tpcf(s_c, a,b)) *\
			(- (1 - mu_c**2)/s_c* m1(s_c, a, b)\
			- mu_c**2 * 2 * a* s_c)  - mu_c**2 * m1(s_c, a, b) * 2 * a * s_c


	np.testing.assert_allclose(te.tpcf_s, expected_tpcf, rtol= 1.e-3)

	expected_mono = tpcf(te.s_c, a, b) - (1 + tpcf(te.s_c, a, b)) * (2./3. * m1(te.s_c, a, b )/te.s_c +  2./3. * a *te.s_c) \
			- 2./3. *a * te.s_c * m1(te.s_c, a, b)

	np.testing.assert_allclose(te.mono, expected_mono, rtol= 1.e-3)

	

def line(x, a, b):
	return a*x + b

def test_second_order():

	def m(x):
		return line(x, 2.,1.)

	def m_dot(x):
		return 2. * np.ones_like(x)

	def c(x):
		return line(x, 1.,2.)

	def c_dot(x):
		return 1. * np.ones_like(x)

	def tpcf_dot(x):
		return 2. * a *x
	
	def tpcf_second_dot(x):
		return 2. * a

	a = 1.
	b = 2.

	moments = {
				'tpcf': {'function': tpcf , 'popt': [a,b]},
				'm1r0t': {'function': m, 'popt':() }, 
				'c2r0t': {'function': c, 'popt':() },
				'c0r2t': {'function': c, 'popt':() },
				}

	moments_df = pd.DataFrame(moments)


	mu =  np.sort(1 - np.geomspace(0.0001, 1., 120))
	s = np.arange(0.,50.,1)

	te = TaylorExpansion(s, mu, moments_df, 1, cutoff_tpcf = 2)


	expected_mono = tpcf(te.s_c, a, b) - (1 + tpcf(te.s_c, a, b)) * (2./3. * m(te.s_c)/te.s_c +  1./3. * m_dot(te.s_c) )\
			- 1./3. *tpcf_dot(te.s_c) * m(te.s_c)

	np.testing.assert_allclose(te.mono, expected_mono, rtol= 1.e-3)

	te = TaylorExpansion(s, mu, moments_df, 2, cutoff_tpcf = 2)

	expected_mono += 1./3. * tpcf_dot(te.s_c) * c_dot(te.s_c) + 4./15.*m(te.s_c)**2/te.s_c*tpcf_dot(te.s_c) +\
			2./5. * m_dot(te.s_c) * m(te.s_c) * tpcf_dot(te.s_c) \
			+ (1 + tpcf(te.s_c, a, b)) * (
			1./3. * c_dot(te.s_c)/te.s_c + 2./15. * m(te.s_c)**2/te.s_c**2 + 1./5. * m_dot(te.s_c)**2 \
					+ 2./3.*m_dot(te.s_c) * m(te.s_c)/te.s_c
			)
	np.testing.assert_allclose(te.mono, expected_mono, rtol= 1.e-3)

	te = TaylorExpansion(s, mu, moments_df, 2, cutoff_tpcf = 3)

	expected_mono += 1./6. * tpcf_second_dot(te.s_c)*c(te.s_c) + 1./10. * tpcf_second_dot(te.s_c) * m(te.s_c)**2 + \
			1./3. * tpcf_dot(te.s_c) * c(te.s_c)/te.s_c + 1./15. * tpcf_dot(te.s_c) * (m(te.s_c))**2 / te.s_c 

	np.testing.assert_allclose(te.mono, expected_mono, rtol= 1.e-3)
