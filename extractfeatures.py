from os import makedirs, getcwd
from os.path import join, isdir, isfile
import math
import pandas as pd
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
import lcperiod
import warnings

work_dir = "/Volumes/Kepler_lcs/Kepler/data/training_set/"


def folders():
    featsdata = work_dir + "features/"
    folddata = work_dir + "fold_data/"
    dmdtdata: dict = work_dir + "dmdt_data/"
    if not isdir(featsdata):
        makedirs(featsdata)
    if not isdir(folddata):
        makedirs(folddata)
    return featsdata, folddata


class FeatureExtractor(object):
    def __init__(self, target, time, flux, flux_err):
        self.target = target
        self.time = time
        self.flux = flux
        self.flux_err = flux_err
        # Path directories
        self.feat_dir, self.fold_dir= folders()
        # Descriptive and weights
        self.median = np.median(self.flux)
        self.std = np.std(self.flux)
        self.weight = 1. / self.flux_err
        self.weighted_sum = np.sum(self.weight)
        self.weighted_mean = np.sum(self.flux * self.weight / self.weighted_sum)
        self.weighted_std = np.sqrt(np.sum((self.flux - self.weighted_mean) ** 2 * self.weight) / self.weighted_sum)
        # Ensure that variables in the time series are equal.
        if (len(self.time) != len(self.flux)) or \
                (len(self.time) != len(self.flux_err)) or \
                (len(self.flux) != len(self.flux_err)):
            raise RuntimeError('The length of time, flux, and err must be same.')
        # if the number of data points is too small.
        min_len = 150
        if len(self.time) < min_len:
            warnings.warn("The number of data points are less than {}.".format(min_len))

    def amplitude(self, flux):
        n = len(self.flux)
        sorted_flux = np.sort(flux)
        amplitude = (np.median(sorted_flux[-int(math.ceil(0.05 * n)):]) -
                     np.median(sorted_flux[0:int(math.ceil(0.05 * n))])) / 2.0
        return amplitude

    def beyond1std(self):
        flux = self.flux
        n = len(flux)
        count = np.sum(np.logical_or(flux > self.weighted_mean + self.weighted_std,
                                     flux < self.weighted_mean - self.weighted_std))
        beyond1std = float(count) / n
        return beyond1std

    def eta(self, flux):
        n = len(flux)
        flux_diff = np.diff(flux)
        sqd_flux_diff = flux_diff * flux_diff
        var_flux = np.median(np.abs(flux - np.std(flux))) * 1.483
        var_flux = var_flux * var_flux
        eta_ = np.median(sqd_flux_diff) / var_flux
        eta = eta_ / (n - 1)
        return eta

    def gskew(self):
        median = self.median
        f_3_value = np.percentile(self.flux, 3)
        f_97_value = np.percentile(self.flux, 97)
        gskew = (np.median(self.flux[self.flux <= f_3_value]) +
                 np.median(self.flux[self.flux >= f_97_value]) - 2 * median)
        return gskew

    def rcs(self, flux):
        n = len(flux)
        mean = np.mean(flux)
        s = np.cumsum(flux - mean) / (n * np.std(flux))
        rcs = np.max(s) - np.min(s)
        return rcs

    def meanvariance(self):
        """The variability index calculated by the std of the flux divided by its mean"""
        meanvariance = self.std / np.mean(self.flux)
        return meanvariance

    def smallkurtosis(self):
        n = len(self.flux)
        mean = np.mean(self.flux)
        std = self.std
        s = sum(((self.flux - mean) / std) ** 4)
        c1 = float(n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
        c2 = float(3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
        small_k = (c1 * s) - c2
        return small_k

    def medianbrp(self):
        n = len(self.flux)
        amp = self.amplitude(self.flux)
        count = np.sum(np.logical_and(self.flux < self.median + amp, self.flux > self.median - amp))
        medianbrp = float(count) / n
        return medianbrp

    def pairslopetrend(self):
        data_last = self.flux[-30:]
        pairslopetrend = (float(len(np.where(np.diff(data_last) > 0)[0]) -
                                len(np.where(np.diff(data_last) <= 0)[0])) / 30)
        return pairslopetrend

    def meanvariance(self):
        """The variability index calculated by the std of the flux divided by its mean"""
        meanvariance = self.std / np.mean(self.flux)
        return meanvariance

    def stetsonk(self, flux, flux_err, median):
        n = len(flux)
        # Stetson index elements
        prefactor = (n / (n - 1))
        sigma_i = prefactor * (flux - median) / flux_err
        stetsonk_ = np.sum(np.abs(sigma_i)) / (np.sqrt(np.sum(sigma_i * sigma_i)))
        stetsonk = stetsonk_ * (n ** (-0.5))
        return stetsonk

    def mad(self):
        mad = np.median(np.abs(self.flux - self.median))
        return mad

    def half_flux_amplitude_ratio(self):
        """
        Return ratio of amplitude of higher and lower magnitudes.


        A ratio of amplitude of higher and lower magnitudes than average,
        considering weights. This ratio, by definition, should be higher
        for EB than for others.
        """

        # For lower (fainter) magnitude than average.
        weight = self.weight
        index = np.where(self.flux > self.weighted_mean)[0]
        lower_weight = weight.iloc[index]
        lower_weight_sum = np.sum(lower_weight)
        lower_flux = self.flux.iloc[index]
        lower_weighted_std = np.sum((lower_flux - self.weighted_mean) ** 2 * lower_weight) / lower_weight_sum

        # For higher (brighter) magnitude than average.
        index = np.where(self.flux <= self.weighted_mean)
        higher_weight = weight.iloc[index]
        higher_weight_sum = np.sum(higher_weight)
        higher_flux = self.flux.iloc[index]
        higher_weighted_std = np.sum((higher_flux - self.weighted_mean) ** 2 * higher_weight) / higher_weight_sum
        amp_ratio = np.sqrt(lower_weighted_std / higher_weighted_std)
        # Return ratio.
        return amp_ratio

    def flux_dist_features(self):
        amplitude = self.amplitude(self.flux)
        amp_ratio = self.half_flux_amplitude_ratio()
        beyond1std = self.beyond1std()
        eta_inverse = 1 / self.eta(self.flux)
        gskew = self.gskew()
        kurtosis = stats.kurtosis(self.flux)
        mad = self.mad()
        meanvariance = self.meanvariance()
        medianbrp = self.medianbrp()
        pairslopetrend = self.pairslopetrend()
        rcs = self.rcs(self.flux)
        skewness = stats.skew(self.flux)
        shapiro_wilk = stats.shapiro(self.flux)[0]
        stetsonk = self.stetsonk(self.flux, self.flux_err, self.median)
        smallkurtosis = self.smallkurtosis()

        return {"amplitude": amplitude, "amp_ratio": amp_ratio, "beyond1std": beyond1std, "eta_inverse": eta_inverse,
                "gskew": gskew, "kurtosis": kurtosis, "mad": mad, "meanvariance": meanvariance, "medianbrp": medianbrp,
                "pairslopetrend": pairslopetrend, "rcs": rcs, "shapiro_wilk": shapiro_wilk, "skewness": skewness,
                "smallkurtosis": smallkurtosis, "stetsonk": stetsonk}

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

    def flux_percentiles(self):
        percentiles = np.percentile(self.flux,
                                    [5.0, 10, 17.5, 25.0, 32.5, 40.0, 60.0, 67.5, 75.0, 82.5, 90.0, 95.0])
        # Flux IQR
        flux_iqr = percentiles[8] - percentiles[3]
        # Percentile differences
        f_5_95 = (percentiles[-1] - percentiles[0])
        f_40_60 = (percentiles[-6] - percentiles[5])
        f_325_675 = (percentiles[-5] - percentiles[4])
        f_75_25 = (percentiles[-4] - percentiles[3])
        f_175_825 = (percentiles[-3] - percentiles[2])
        f_10_90 = (percentiles[-2] - percentiles[1])
        # Ratios divided by f_5_95
        flux_percentile_ratio_mid_20 = f_40_60 / f_5_95
        flux_percentile_ratio_mid_35 = f_325_675 / f_5_95
        flux_percentile_ratio_mid_50 = f_75_25 / f_5_95
        flux_percentile_ratio_mid_65 = f_175_825 / f_5_95
        flux_percentile_ratio_mid_80 = f_10_90 / f_5_95
        # Percent difference flux percentile
        percent_difference_flux_percentile = f_5_95 / self.median
        # Percent amplitude
        percent_amplitude = np.max(np.abs(self.flux)) / self.median
        return {"iqr": flux_iqr, "flux_percentile_ratio_mid_20": flux_percentile_ratio_mid_20,
                "flux_percentile_ratio_mid_35": flux_percentile_ratio_mid_35,
                "flux_percentile_ratio_mid_50": flux_percentile_ratio_mid_50,
                "flux_percentile_ratio_mid_65": flux_percentile_ratio_mid_65,
                "flux_percentile_ratio_mid_80": flux_percentile_ratio_mid_80,
                "percent_amplitude": percent_amplitude,
                "percent_difference_flux_percentile": percent_difference_flux_percentile}

    def periodic_feats(self):
        lc = np.array((self.time, self.flux, self.flux_err))
        star_period = lcperiod.PeriodSearching(*lc)
        fourier_components = star_period.fourier_results()
        return fourier_components

    def phasedlc_features(self):
        lc = np.array((self.time, self.flux, self.flux_err))
        star_per = lcperiod.PeriodSearching(*lc)
        lc_folded = star_per.folded_lc()
        file_nm = "{}_folded.csv".format(self.target)
        lc_folded.to_csv(join(self.fold_dir + file_nm), index=False)
        # Dictionary for additional features
        features = {}
        # Range of a cumulative sum of phase folded light curve
        rcs = float(self.rcs(lc_folded))
        features["psi_cs"] = rcs
        features["EPIC/KIC/TIC"] = self.target
        return features

    def calcfeatures(self):
        """"""
        features = {}
        features_dicts = [self.flux_dist_features(), self.structurefunctions(),
                          self.flux_percentiles(), self.phasedlc_features(),
                          self.periodic_feats()]
        for features_dict in features_dicts:
            for key, value in features_dict.items():
                features.setdefault(key, value)
        features_ = pd.DataFrame(features.values()).T
        features_.columns = features.keys()
        features = features_.round(6)
        features.set_index("EPIC/KIC/TIC", inplace=True)
        return features
