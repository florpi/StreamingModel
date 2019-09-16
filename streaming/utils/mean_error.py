import numpy as np
from scipy.interpolate import interp1d 

class MeanError:
	'''Class producing the mean and standard deviation of a given statistic for all simulation boxes.

	Attributes:
		mean: mean over boxes.
		std: standard deviation over boxes.

	'''

	def __init__(self, box_list: list, interpolate_x: None = False):

		
		mean, std = self.compute_mean_error(box_list)

		self.mean = mean
		self.std = std 

		if interpolate_x is not False:
			self.interpolate(interpolate_x)


	def compute_mean_error(self, box_list: list):

		mean = np.mean(box_list, axis = 0)
		std = np.std(box_list, axis = 0)

		return mean, std

	def interpolate(self, interpolate_x: None):

		self.mean = interp1d(interpolate_x, self.mean, 
				fill_value = (self.mean[0], self.mean[-1]), bounds_error = False)


