import numpy as np
from scipy.stats import norm


def moments2gaussian(mean ,std):
	'''
	Args:
		mean: np.array
		std: np.array

	Returns:
		pdf_los: function


	'''

	def pdf_los(vlos, rperp, rparallel):

		r = np.sqrt(rperp** 2 + rparallel** 2)
		mu = rparallel/r


		return norm.pdf(vlos, loc = mean, scale = std)

	return pdf_los


