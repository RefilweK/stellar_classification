#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# lspestimate.py
# Refilwe Kgoadi
# (rgkgoadi@gmail.com)
# (refilwe.kgoadi1@my.jcu.edu.au)
"""
This python module estimates the period of a light curve using the Lomb-Scargle periodogram method (Lomb 1976 and
Scargle 1982). The module import astropy to implement Lomb-Scargle
The second part of the modules applies Fourier decomposition to calculate fourier parameters from the fitted sinusoids.
"""

import numpy as np
from astropy.timeseries import LombScargle
from scipy.optimize import curve_fit


def argmaxes(arr, n):
    arr = -np.asarray(arr)
    sort = np.argsort(arr)
    return sort[:n]


def lomb(time, flux, flux_err):
    """

    :param time:
    :param flux:
    :param flux_err:
    :return:
    """
    freq_model = LombScargle(time, flux, flux_err)
    freq, power = freq_model.autopower(normalization="standard", nyquist_factor=100)
    return freq, power


def model(x, a, b, c, freq):
    fourier_model = (a * np.sin(2 * np.pi * freq * x) + b * np.cos(2 * np.pi * freq * x) + c)
    return fourier_model


def yfunc_maker(freq):
    def func(x, a, b, c):
        return a * np.sin(2 * np.pi * freq * x) + b * np.cos(2 * np.pi * freq * x) + c

    return func


class PeriodSearching:
    def __init__(self, time, flux, flux_err):
        self.time = time
        self.flux = flux
        self.flux_err = flux_err
        self.peaks = 1

    def compute_lomb(self):
        frequency, power = lomb(self.time, self.flux, self.flux_err)
        return frequency, power

    def compute_lsp(self):
        frequency, power = self.compute_lomb()
        fmaxs = argmaxes(power, self.peaks)
        best_periods = 1 / frequency[fmaxs]
        return best_periods

    def log_period(self):
        frequency, power = self.compute_lomb()
        fmaxs = argmaxes(power, self.peaks)
        best_periods = 1 / frequency[fmaxs]
        best_period = float(np.log10(best_periods))
        return best_period

    def fourier_comps(self):
        time = self.time - np.min(self.time)
        a, ph = [], []
        for i in range(3):
            freq, power = self.compute_lomb()
            fmax = np.argmax(power)
            fund_freq = freq[fmax]
            atemp, phtemp = [], []
            oflux = self.flux
            for j in range(4):
                function_to_fit = yfunc_maker((j + 1) * fund_freq)
                popt0, popt1, popt2 = curve_fit(function_to_fit, time, oflux)[0][:3]
                atemp.append(np.sqrt(popt0 ** 2 + popt1 ** 2))
                phtemp.append(np.arctan(popt1 / popt0))
                # fit_model = model(time, popt0, popt1, popt2, (j + 1) * fund_freq)
                # flux = np.array(self.flux) - fit_model
            a.append(atemp)
            ph.append(phtemp)
        ph = np.asarray(ph)
        scaledph = ph - ph[:, 0].reshape((len(ph), 1))
        return a, scaledph

    def fourier_fit(self):

        a, sph = self.fourier_comps()
        results = {"freq1_harmonics": (a[0], sph[0]), "freq2_harmonics": (a[1], sph[1]),
                   "freq3_harmonics": (a[2], sph[2]), }
        return results

    def fourier_results(self):
        period = self.log_period()
        results = self.fourier_fit()
        # freq1_harmonics
        freq1_harmonics_amplitude = results["freq1_harmonics"][0]
        freq1_harmonics_amplitude_0 = freq1_harmonics_amplitude[0]
        freq1_harmonics_amplitude_1 = freq1_harmonics_amplitude[1]
        freq1_harmonics_amplitude_2 = freq1_harmonics_amplitude[2]
        freq1_harmonics_amplitude_3 = freq1_harmonics_amplitude[3]
        freq1_harmonics_rel_phase = results["freq1_harmonics"][1]
        freq1_harmonics_rel_phase_1 = freq1_harmonics_rel_phase[1]
        freq1_harmonics_rel_phase_2 = freq1_harmonics_rel_phase[2]
        freq1_harmonics_rel_phase_3 = freq1_harmonics_rel_phase[3]
        # freq2_harmonics
        freq2_harmonics_amplitude = results["freq2_harmonics"][0]
        freq2_harmonics_amplitude_0 = freq2_harmonics_amplitude[0]
        freq2_harmonics_amplitude_1 = freq2_harmonics_amplitude[1]
        freq2_harmonics_amplitude_2 = freq2_harmonics_amplitude[2]
        freq2_harmonics_amplitude_3 = freq2_harmonics_amplitude[3]
        freq2_harmonics_rel_phase = results["freq2_harmonics"][1]
        freq2_harmonics_rel_phase_1 = freq2_harmonics_rel_phase[1]
        freq2_harmonics_rel_phase_2 = freq2_harmonics_rel_phase[2]
        freq2_harmonics_rel_phase_3 = freq2_harmonics_rel_phase[3]
        # freq3_harmonics
        freq3_harmonics_amplitude = results["freq3_harmonics"][0]
        freq3_harmonics_amplitude_0 = freq3_harmonics_amplitude[0]
        freq3_harmonics_amplitude_1 = freq3_harmonics_amplitude[1]
        freq3_harmonics_amplitude_2 = freq3_harmonics_amplitude[2]
        freq3_harmonics_amplitude_3 = freq3_harmonics_amplitude[3]
        freq3_harmonics_rel_phase = results["freq3_harmonics"][1]
        freq3_harmonics_rel_phase_1 = freq3_harmonics_rel_phase[1]
        freq3_harmonics_rel_phase_2 = freq3_harmonics_rel_phase[2]
        freq3_harmonics_rel_phase_3 = freq3_harmonics_rel_phase[3]
        return {"log periodLS": period,
                "freq1_harmonics_amplitude_0": freq1_harmonics_amplitude_0,
                "freq1_harmonics_amplitude_1": freq1_harmonics_amplitude_1,
                "freq1_harmonics_amplitude_2": freq1_harmonics_amplitude_2,
                "freq1_harmonics_amplitude_3": freq1_harmonics_amplitude_3,
                "freq1_harmonics_rel_phase_1": freq1_harmonics_rel_phase_1,
                "freq1_harmonics_rel_phase_2": freq1_harmonics_rel_phase_2,
                "freq1_harmonics_rel_phase_3": freq1_harmonics_rel_phase_3,
                "freq2_harmonics_amplitude_0": freq2_harmonics_amplitude_0,
                "freq2_harmonics_amplitude_1": freq2_harmonics_amplitude_1,
                "freq2_harmonics_amplitude_2": freq2_harmonics_amplitude_2,
                "freq2_harmonics_amplitude_3": freq2_harmonics_amplitude_3,
                "freq2_harmonics_rel_phase_1": freq2_harmonics_rel_phase_1,
                "freq2_harmonics_rel_phase_2": freq2_harmonics_rel_phase_2,
                "freq2_harmonics_rel_phase_3": freq2_harmonics_rel_phase_3,
                "freq3_harmonics_amplitude_0": freq3_harmonics_amplitude_0,
                "freq3_harmonics_amplitude_1": freq3_harmonics_amplitude_1,
                "freq3_harmonics_amplitude_2": freq3_harmonics_amplitude_2,
                "freq3_harmonics_amplitude_3": freq3_harmonics_amplitude_3,
                "freq3_harmonics_rel_phase_1": freq3_harmonics_rel_phase_1,
                "freq3_harmonics_rel_phase_2": freq3_harmonics_rel_phase_2,
                "freq3_harmonics_rel_phase_3": freq3_harmonics_rel_phase_3}
