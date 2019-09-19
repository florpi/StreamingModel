import matplotlib.pyplot as plt
import matplotlib.font_manager
import matplotlib as mpl
mpl.style.use('~/StreamingModel/streaming/utils/mplstyle')


def plot_residuals_std(simulation_direct_measurement, list_of_models, attribute, colors):


	fig, (ax1, ax2) = plt.subplots(nrows=2,sharex = True, squeeze = True,
		                          gridspec_kw = {'height_ratios':[4,1]})


	measured_attribute  = getattr(simulation_direct_measurement, attribute)
	ax1.errorbar(simulation_direct_measurement.s_c, 
				simulation_direct_measurement.s_c **2* measured_attribute .mean,
				yerr = simulation_direct_measurement.s_c**2* measured_attribute .std,
				label = 'N-body ', color='black')


	for i, model in enumerate(list_of_models):

		ax1.plot(model.s_c, model.s_c**2 * getattr(model, attribute),
				label = model.label, color = colors[i])

		ax2.plot(model.s_c, (getattr(model, attribute) - measured_attribute.mean)/measured_attribute.std,
				color = colors[i]) 

	ax2.set_ylim(-5,5)

	ax2.fill_between(simulation_direct_measurement.s_c,-1., 1., facecolor = 'yellow', alpha = 0.5)
	ax2.axhline(y = 0., linestyle='-', color='gray', alpha = 0.5)

	ax1.legend()
