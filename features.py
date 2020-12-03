#!/usr/bin/env pthon3
# -*- coding: utf-8 -*-
# features.py
# Refilwe Kgoadi
# (rgkgoadi@gmail.com)
# (refilwe.kgoadi1@m.jcu.edu.au)
"""
Functions to extract features in the feature extractor module
"""
import math
import pandas as pd
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d


def descriptives(y, y_err):
    """Calculates descriptive parameters of a y"""
    median = np.median(y)
    std = np.std(y)
    weighted_err = 1. / y_err
    weighted_sum = np.sum(weighted_err)
    weighted_mean = np.sum(y * weighted_err / weighted_sum)
    weighted_std = np.sqrt(np.sum((y - weighted_mean) ** 2 * weighted_err) / weighted_sum)
    describe = {'median': median, 'std': std, 'weighted_err': weighted_err, 'weighted_mean': weighted_mean,
                'weighted_std': weighted_std, 'weighted_err': weighted_err}
    return describe


def dist_features(y, y_err):
    y_distribution = descriptives(y, y_err)
    w_mean = y_distribution['weighted_mean']
    w_err = y_distribution['weighted_err']
    w_std = y_distribution["weighted_std"]
    median = y_distribution['median']
    y = y
    n = len(y)
    sorted_y = np.sort(y)
    # Amplitude
    amplitude = (np.median(sorted_y[-int(math.ceil(0.05 * n)):]) - np.median(
        sorted_y[0:int(math.ceil(0.05 * n))])) / 2.0
    count = np.sum(np.logical_or(y > w_mean + w_std, y < w_mean - w_std))
    # Beyond 1 standard deviation
    beyond1std = float(count) / n
    # Eta
    n = len(y)
    y_diff = np.diff(y)
    sqd_y_diff = y_diff * y_diff
    var_y = np.median(np.abs(y - w_std)) * 1.483
    var_y = var_y * var_y
    eta_ = np.median(sqd_y_diff) / var_y
    eta = eta_ / (n - 1)
    # GSkew
    f_3_value = np.percentile(y, 3)
    f_97_value = np.percentile(y, 97)
    gskew = (np.median(y[y <= f_3_value]) + np.median(y[y >= f_97_value]) - 2 * median)
    # RCS
    s = np.cumsum(y - w_mean) / (n * w_std)
    rcs = np.max(s) - np.min(s)
    # Mean variance
    meanvariance = w_std / w_mean
    # Kurtosis 
    kurtosis = stats.kurtosis(y)
    # MAD 
    mad = np.median(np.abs(y - median))
    percentiles = np.percentile(y, [5.0, 10, 17.5, 25.0, 32.5, 40.0, 60.0, 67.5, 75.0, 82.5, 90.0, 95.0])
    # y IQR
    y_iqr = percentiles[8] - percentiles[3]
    # Percentile differences
    f_5_95 = (percentiles[-1] - percentiles[0])
    f_40_60 = (percentiles[-6] - percentiles[5])
    f_325_675 = (percentiles[-5] - percentiles[4])
    f_75_25 = (percentiles[-4] - percentiles[3])
    f_175_825 = (percentiles[-3] - percentiles[2])
    f_10_90 = (percentiles[-2] - percentiles[1])
    # Ratios divided b f_5_95
    y_percentile_ratio_mid_20 = f_40_60 / f_5_95
    y_percentile_ratio_mid_35 = f_325_675 / f_5_95
    y_percentile_ratio_mid_50 = f_75_25 / f_5_95
    y_percentile_ratio_mid_65 = f_175_825 / f_5_95
    y_percentile_ratio_mid_80 = f_10_90 / f_5_95
    # Percent difference y percentile
    percent_difference_y_percentile = f_5_95 / median
    # Percent amplitude
    percent_amplitude = np.max(np.abs(y)) / median
    count_ = np.sum(np.logical_and(y < median + amplitude, y > median - amplitude))
    medianbrp = float(count_) / n
    data_last = y[-30:]
    pairslopetrend = (float(len(np.where(np.diff(data_last) > 0)[0]) - len(np.where(np.diff(data_last) <= 0)[0])) / 30)
    # stetsonk
    prefactor = (n / (n - 1))
    sigma_i = prefactor * (y - median) / y_err
    stetsonk = np.sum(np.abs(sigma_i)) / (np.sqrt(np.sum(sigma_i * sigma_i)))
    stetsonk = stetsonk * (n ** (-0.5))
    # half_flux_amplitude_ratio(self):
    ## For lower (fainter) magnitude than average.
    index = np.where(y > w_mean)[0]
    lower_weight = w_err.iloc[index]
    lower_weight_sum = np.sum(lower_weight)
    lower_flux = y.iloc[index]
    lower_weighted_std = np.sum((lower_flux - w_mean) ** 2 * lower_weight) / lower_weight_sum
    # For higher (brighter) magnitude than average.
    index = np.where(y <= w_mean)
    higher_weight = w_err.iloc[index]
    higher_weight_sum = np.sum(higher_weight)
    higher_flux = y.iloc[index]
    higher_weighted_std = np.sum((higher_flux - w_mean) ** 2 * higher_weight) / higher_weight_sum
    amp_ratio = np.sqrt(lower_weighted_std / higher_weighted_std)
    # Skewness
    skewness = stats.skew(y)
    # Shapiro Normality test
    shapiro_wilk = stats.shapiro(y)[0]
    # Kurtosis
    kurtosis = stats.kurtosis(y)
    features = {'amplitude': amplitude, 'amp_ratio': amp_ratio, 'beyond1std': beyond1std, 'eta': eta, "gskew": gskew,
                "iqr": y_iqr, "kurtosis": kurtosis, 'mad': mad, 'medianbrp': medianbrp, 'meanvariance': meanvariance,
                'pairslopetrend': pairslopetrend, "rcs": rcs, 'skewness': skewness, 'shapiro_wilk': shapiro_wilk,
               "y_percentile_ratio_mid_20": y_percentile_ratio_mid_20,
                "y_percentile_ratio_mid_35": y_percentile_ratio_mid_35,
                "y_percentile_ratio_mid_50": y_percentile_ratio_mid_50,
                "y_percentile_ratio_mid_65": y_percentile_ratio_mid_65,
                "y_percentile_ratio_mid_80": y_percentile_ratio_mid_80, "percent_amplitude": percent_amplitude,
                "percent_difference_y_percentile": percent_difference_y_percentile}
    return features


def structurefunctions(x, y):
    nsf, np_ = 100, 100
    sf1, sf2, sf3 = np.zeros(nsf), np.zeros(nsf), np.zeros(nsf)
    f = interp1d(x, y)
    x_int = np.linspace(np.min(x), np.max(x), np_)
    y_int = f(x_int)
    for tau in np.arange(1, nsf):
        sf1[tau - 1] = np.mean(np.power(np.abs(y_int[0:np_ - tau] - y_int[tau:np_]), 1.0))
        sf2[tau - 1] = np.mean(np.abs(np.power(np.abs(y_int[0:np_ - tau] - y_int[tau:np_]), 2.0)))
        sf3[tau - 1] = np.mean(np.abs(np.power(np.abs(y_int[0:np_ - tau] - y_int[tau:np_]), 3.0)))
        sf1_log = np.log10(np.trim_zeros(sf1))
        sf2_log = np.log10(np.trim_zeros(sf2))
        sf3_log = np.log10(np.trim_zeros(sf3))
    if len(sf1_log) and len(sf2_log):
        m_21, b_21 = np.polfit(sf1_log, sf2_log, 1)
    if len(sf1_log) and len(sf3_log):
        m_31, b_31 = np.polfit(sf1_log, sf3_log, 1)
    if len(sf2_log) and len(sf3_log):
        m_32, b_32 = np.polfit(sf2_log, sf3_log, 1)
    return {"structure_function_index_21": m_21, "structure_function_index_31": m_31,
            "structure_function_index_32": m_32}
