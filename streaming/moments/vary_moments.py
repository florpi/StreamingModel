import copy
import numpy as np
from streaming.models import stream


def increase(simulation, key, percentage):
    
    def moment(r):
        value = getattr(simulation, key).mean(r)
        return value * (1. + percentage)
    return moment

def decrease(simulation, key, percentage):
    
    def moment(r):
        value = getattr(simulation, key).mean(r)        
        return value * (1. - percentage)
    return moment

def slope_increase(simulation, key, percentage):
       
    def moment(r):
        value = getattr(simulation, key).mean(r)        
        return value * (1. + percentage * r)
    
    return moment

def slope_decrease(simulation, key, percentage):
       
    def moment(r):
        value = getattr(simulation, key).mean(r)        
        return value * (1. - percentage * r)
    
    return moment

def gradual_increase(simulation, key):
    
    def moment(r):
        value = getattr(simulation, key).mean(r)        
        return value * (1. + 0.8/r)
    
    return moment


def gradual_decrease(simulation, key):
       
    def moment(r):
        value = getattr(simulation, key).mean(r)        
        return value * (1. - 0.8/r)
    
    return moment

class ModifyMoment():
	def __init__(self, simulation, measured_moments, moment_key):

		varied_moments = copy.deepcopy(measured_moments)
		
		varied_moments[moment_key]['function'] = increase(simulation, moment_key, 0.05)
		self.increase = stream.Stream(simulation, 'skewt',
                        best_fit_moments = varied_moments)

		varied_moments[moment_key]['function'] = decrease(simulation, moment_key, 0.05)
		self.decrease = stream.Stream(simulation, 'skewt',
                        best_fit_moments = varied_moments)


		varied_moments[moment_key]['function'] = gradual_increase(simulation, moment_key)
		self.gradual_increase = stream.Stream(simulation, 'skewt',
                        best_fit_moments = varied_moments)

		varied_moments[moment_key]['function'] = gradual_decrease(simulation, moment_key)
		self.gradual_decrease = stream.Stream(simulation, 'skewt',
                        best_fit_moments = varied_moments)




class ModifyMoments():
	def __init__(self, simulation):

		measured_moments = {
					'm_10': {'function': simulation.m_10.mean, 'popt': ()},
					'c_20': {'function': simulation.c_20.mean, 'popt': ()},
					'c_02': {'function': simulation.c_02.mean, 'popt': ()},

		}

		for key in measured_moments:
			setattr(self, key, ModifyMoment(simulation, measured_moments, key))
