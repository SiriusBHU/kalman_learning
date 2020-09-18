from __future__ import division, print_function
import book_format
book_format.set_style()

import kf_book.book_plots as book_plots
# from kf_book.book_plots import plot_errorbars
# plot_errorbars([(160, 8, 'A'), (170, 8, 'B')], xlims=(150, 180))
# plot_errorbars([(160, 3, 'A'), (170, 9, 'B')], xlims=(150, 180))
# plot_errorbars([(160, 1, 'A'), (170, 9, 'B')], xlims=(150, 180))


import numpy as np
measurements = np.random.uniform(160, 170, size=10000)
mean = measurements.mean()
print('Average of measurements is {:.4f}'.format(mean))

mean = np.random.normal(165, 5, size=10000).mean()
print('Average of measurements is {:.4f}'.format(mean))

import kf_book.gh_internal as gh
import matplotlib.pyplot as plt
# gh.plot_hypothesis1()
# gh.plot_hypothesis2()
# gh.plot_hypothesis3()
# gh.plot_hypothesis4()
# gh.plot_hypothesis5()
#
# gh.plot_estimate_chart_1()
# gh.plot_estimate_chart_2()
# gh.plot_estimate_chart_3()
# plt.show()


from kf_book.book_plots import figsize
weights = [158.0, 164.2, 160.3, 159.9, 162.1, 164.6,
           169.6, 167.4, 166.4, 171.0, 171.2, 172.6]

time_step = 1.0  # day
scale_factor = 4.0 / 10.0


def predict_using_gain_guess(estimated_weight, gain_rate, do_print=False):
    # storage for the filtered results
    estimates, predictions = [estimated_weight], []

    # most filter literature uses 'z' for measurements
    for z in weights:
        # predict new position
        predicted_weight = estimated_weight + gain_rate * time_step

        # update filter
        estimated_weight = predicted_weight + scale_factor * (z - predicted_weight)

        # save and log
        estimates.append(estimated_weight)
        predictions.append(predicted_weight)
        if do_print:
            gh.print_results(estimates, predicted_weight, estimated_weight)

    return estimates, predictions


initial_estimate = 160.
estimates, predictions = predict_using_gain_guess(
    estimated_weight=initial_estimate, gain_rate=1, do_print=True)
book_plots.set_figsize(10)
# weights.insert(0, 160)
plt.figure()
gh.plot_gh_results(weights, estimates, predictions, [160, 172])
# plt.show()
print(weights)


e, p = predict_using_gain_guess(initial_estimate, -1.)
plt.figure()
gh.plot_gh_results(weights, e, p, [160, 172])
# plt.show()
print(weights)


# change gain rate
weight = 160.  # initial guess
gain_rate = -1.0  # initial guess

time_step = 1.
weight_scale = 4. / 10
gain_scale = 1. / 3
estimates = [weight]
predictions = []

for z in weights:
    # prediction step
    weight = weight + gain_rate * time_step
    gain_rate = gain_rate
    predictions.append(weight)

    # update step
    residual = z - weight

    gain_rate = gain_rate + gain_scale * (residual / time_step)
    weight = weight + weight_scale * residual

    estimates.append(weight)

plt.figure()
gh.plot_gh_results(weights, estimates, predictions, [160, 172])
# plt.show()
print(weights)


# book_plots.predict_update_chart()
# plt.show()

from kf_book.gh_internal import plot_g_h_results
def g_h_filter(data, x0, dx, g, h, dt):

    prediction, estimation = [], [x0]
    for _d in data:

        # predict use dx and dt
        _pred = estimation[-1] + dx * dt
        _res = _d - _pred

        # update estimation using g filter
        _est = _pred + g * _res

        # update dx using h filter
        dx = dx + h * (_res / dt)

        prediction.append(_pred)
        estimation.append(_est)

    return estimation, prediction


estimates, predictions = g_h_filter(
    weights, initial_estimate, dx=1., g=0.6, h=2 / 3, dt=1.)
book_plots.set_figsize(10)
# weights.insert(0, 160)
plt.figure()
gh.plot_gh_results(weights, estimates, predictions, [160, 172])
plt.show()





