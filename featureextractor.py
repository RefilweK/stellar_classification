#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# featureextractor.py
# Refilwe Kgoadi
# (rgkgoadi@gmail.com)
# (refilwe.kgoadi1@my.jcu.edu.au)
"""
This python module engineers/extracts features from stellar light curve (and other forms time series).
These are a set of features used to classify variable stars in Machine Learning techniques.
Features are based from on python packages by Nun et al. 2015 and Kim and Bailer-Jones 2016.
The current version of this module can be applied to an observable as a function of time and its associated error.
"""
import warnings
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import features
import lspestimate


class FeatureExtractor(object):
    def __init__(self, target, time, flux, flux_err):
        self.target = target
        self.time = time
        self.flux = flux
        self.flux_err = flux_err
        # Ensure that variables in the time series are equal.
        if (len(self.time) != len(self.flux)) or \
                (len(self.time) != len(self.flux_err)) or \
                (len(self.flux) != len(self.flux_err)):
            raise RuntimeError('The length of time, flux, and err must be same.')
        # if the number of data points is too small.
        min_len = 150
        if len(self.time) < min_len:
            warnings.warn("The number of data points are less than {}.".format(min_len))

    def periodic_features(self):
        lc = np.array((self.time, self.flux, self.flux_err))
        lsp = lspestimate.PeriodSearching(*lc)
        features_ = lsp.fourier_results()
        features_["EPIC/KIC/TIC"] = self.target
        return features_

    def describe_features(self):
        features_1 = features.dist_features(self.flux, self.flux_err)
        return features_1

    def structurefunctions(self):
        flux = self.flux
        time = self.time
        nsf, np_ = 100, 100
        sf1, sf2, sf3 = np.zeros(nsf), np.zeros(nsf), np.zeros(nsf)
        f = interp1d(time, flux)
        time_int = np.linspace(np.min(time), np.max(time), np_)
        flux_int = f(time_int)
        for tau in np.arange(1, nsf):
            sf1[tau - 1] = np.mean(np.power(np.abs(flux_int[0:np_ - tau] - flux_int[tau:np_]), 1.0))
            sf2[tau - 1] = np.mean(np.abs(np.power(np.abs(flux_int[0:np_ - tau] - flux_int[tau:np_]), 2.0)))
            sf3[tau - 1] = np.mean(np.abs(np.power(np.abs(flux_int[0:np_ - tau] - flux_int[tau:np_]), 3.0)))
            sf1_log = np.log10(np.trim_zeros(sf1))
            sf2_log = np.log10(np.trim_zeros(sf2))
            sf3_log = np.log10(np.trim_zeros(sf3))
        if len(sf1_log) and len(sf2_log):
            m_21, b_21 = np.polyfit(sf1_log, sf2_log, 1)
        if len(sf1_log) and len(sf3_log):
            m_31, b_31 = np.polyfit(sf1_log, sf3_log, 1)
        if len(sf2_log) and len(sf3_log):
            m_32, b_32 = np.polyfit(sf2_log, sf3_log, 1)
        return {"structure_function_index_21": m_21, "structure_function_index_31": m_31,
                "structure_function_index_32": m_32}

    def calcfeatures(self):
        """"""
        ts_features = {}
        features_dicts = [self.describe_features(), self.structurefunctions(), self.periodic_features()]
        for features_dict in features_dicts:
            for key, value in features_dict.items():
                ts_features.setdefault(key, value)
        features_ = pd.DataFrame(ts_features.values()).T
        features_.columns = ts_features.keys()
        lc_features = features_.round(6)
        lc_features.set_index("EPIC/KIC/TIC", inplace=True)
        return lc_features
