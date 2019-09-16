import numpy as np
from streaming.moments.project_moments import project_moment_to_los
import streaming.moments.compute_moments as cm 
from scipy.stats import norm


def moments2gaussian(sim_measurement):
	'''
	Args:
		mean: np.array
		std: np.array

	Returns:
		pdf_los: function


	'''

	def pdf_los(vlos, rperp, rparallel):


		mean = project_moment_to_los(sim_measurement, rparallel, 
				rperp, 1, mode = 'm')
		scale = cm.std(project_moment_to_los(sim_measurement, rparallel, 
			rperp, 2, mode = 'c'))

		return norm.pdf(vlos, loc = mean, scale = scale)

	return pdf_los


