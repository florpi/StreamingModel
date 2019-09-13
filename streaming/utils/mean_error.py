import numpy as np

class MeanError:

	def __init__(self, box_list):

		
		mean, std = self.compute_mean_error(box_list)

		self.mean = mean
		self.std = std 



	def compute_mean_error(self, box_list):

		mean = np.mean(box_list, axis = 0)
		std = np.mean(box_list, axis = 0)

		return mean, std


