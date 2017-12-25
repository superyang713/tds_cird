import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress

from scipy.integrate import quad
from scipy.optimize import curve_fit
import math


def biexp_decay(x, a1, a2, b1, b2, c1, c2, d):
    """Expression for bi-exponential decay"""
    return a1 * np.exp(b1 * (x - c1)) + a2 * np.exp(b2 * (x - c2)) + d


def write_file(data, tds_after_cird):
    """Create a txt file for igor process."""
    message = input(
        "Save the data to p{}? ".format(tds_after_cird.filename)
        + "(Y/N): "
    )
    if message == 'y':
        file = open(
            tds_after_cird.filepath[:-12] + 'p' + tds_after_cird.filename, 'w+'
        )
        file.write(
            'IGOR\nWAVES, Counts_before_CIRD_{}, temps_before_CIRD{}'
            'Counts_after_CIRD_{}, temps_after_CIRD{}, Difference_{},'
            'temp_average_{}, Time_elapsed_{}, \nBegin\n'
            .format(tds_after_cird.filename[0:8], tds_after_cird.filename[0:8],
                    tds_after_cird.filename[0:8], tds_after_cird.filename[0:8],
                    tds_after_cird.filename[0:8], tds_after_cird.filename[0:8],
                    tds_after_cird.filename[0:8])
        )
        for a, b, c, d, e, f, g in data[
            ["counts_background_subtracted_before",
             "temperature_before",
             "counts_background_subtracted_after",
             "temperature_after",
             "counts_difference",
             "temperature_average",
             "time_elapsed"]
        ].as_matrix():
            file.write('{} {} {} {} {} {} {}\n'.format(a, b, c, d, e, f, g))
        file.write('End\n')
        file.close()
        print('Success!')


def step_average(data, percent_npts, show_plot=True):
    """
    Find the average counts of each pressure range during sensitivity
    mesurements.
    """
    x = np.abs(np.diff(data["counts_raw"]))
    if show_plot:
        indeces = detect_peaks(x, mph=10000, mpd=50, show=True)
    else:
        indeces = detect_peaks(x, mph=10000, mpd=50, show=False)

    indeces = np.array(indeces)
    offset = np.diff(indeces) * (1-percent_npts) / 2

    index_one = math.ceil(indeces[0] + offset[0])
    index_two = math.ceil(indeces[1] - offset[0])
    index_three = math.ceil(indeces[1] + offset[1])
    index_four = math.ceil(indeces[2] - offset[1])
    index_five = math.ceil(indeces[2] + offset[2])
    index_six = math.ceil(indeces[3] - offset[2])

    high_counts = np.average(data["counts_raw"].iloc[index_one:index_two])
    mid_counts = np.average(data["counts_raw"].iloc[index_three:index_four])
    low_counts = np.average(data["counts_raw"].iloc[index_five:index_six])

    return high_counts, mid_counts, low_counts


def exp_decay(x, a, b, c):
    """Expression for exponential decay"""
    return a * np.exp(b * x) + c


def generateDict():
    monthKeys = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug',
                 'sep', 'oct', 'nov', 'dec']
    monthValues = ['01', '02', '03', '04', '05', '06', '07', '08', '09',
                   '10', '11', '12']
    return dict(zip(monthKeys, monthValues))


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) <
                    math.fabs(value - array[idx])):
        return int(idx - 1)
    else:
        return int(idx)


def tds_sensitivity(averages, high_p, middle_p, low_p, show_plot=False):
    """
    pressure exponent ^-10 has been included.
    averages: the output tuple from the function step_averages()
    """
    xs = np.array([0, high_p, middle_p, low_p])
    ys = np.array([0, averages[0], averages[1], averages[2]])
    sensitivity_factor = linregress(xs, ys)[0]
    intercept = linregress(xs, ys)[1]
    if show_plot:
        plt.plot(xs, ys, 'o', label='original data')
        plt.plot(xs, intercept + sensitivity_factor*xs,
                 'r', label='fitted line')
        plt.show()
    return sensitivity_factor


def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):
    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indexes of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) &
                           (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) &
                           (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[
            np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))),
                    invert=True)
        ]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indexes by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind


def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02*x.size, x.size*1.02-1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge))
        # plt.grid()
        plt.show()


class CIRD:
    """
    It takes a file name to generate the CIRD object.
    """
    def __init__(self, filename):
        self.filename = filename

    def load_data(self, dwell_time=0.2, show_plot=True):
        monthDict = generateDict()
        self.filepath = os.path.join(
            os.path.expanduser('~'), 'Dropbox (MIT)', 'littlemachine', '20' +
            self.filename[6:8], monthDict[self.filename[:3]] + '_' +
            self.filename[:3], self.filename)
        self.data = pd.read_csv(
            self.filepath, delim_whitespace=True, skipfooter=3, skiprows=3,
            names=["counts", "temp"], engine="python"
        )
        self.data["time"] = [
            t for t in np.linspace(0, dwell_time * len(self.data.index),
                                   num=len(self.data.index), endpoint=False)
        ]
        if show_plot:
            self.data.plot(x="time", y="counts")
            plt.show()

    def level_background(self, npts=50, sensitivity_corrected=True):
        """
        Find the beam start time if the feature is obvious.
        Level the background using the beam star time.
        Ask for user's input for beamstop time.
        """
        self.index_beamstart = np.argmax(np.diff(self.data["counts"]))+1
        self.time_beamstart = self.data["time"].iloc[self.index_beamstart]
        self.time_beamstop = float(input("Estimated beam stop time: "))
        self.index_beamstop = int(self.time_beamstop * 5)
        X = pd.concat(
            [self.data["time"].iloc[
                (self.index_beamstart-npts):self.index_beamstart],
                self.data["time"].iloc[-npts:]
             ]
        )
        y = pd.concat(
            [self.data["counts"].iloc[
                (self.index_beamstart-npts):self.index_beamstart],
                self.data["counts"].iloc[-npts:]
             ]
        )
        slope = linregress(X, y)[0]
        counts_leveled = self.data["counts"].sub(self.data["time"] * slope)
        sensitivity = float(input("Please enter the sensitivity factor:"))
        background_sensitivity = 1.4754
        self.data["counts_leveled_background"] = counts_leveled /\
            (sensitivity / background_sensitivity)
        self.data["counts_leveled_background"] =\
            self.data["counts_leveled_background"] -\
            np.mean(self.data["counts_leveled_background"][-npts:])

    def peak_fit(self, area_high=False):
        """docstring for exponential_fit"""
        self.X = self.data["time"].iloc[
            self.index_beamstart:self.index_beamstop]
        self.y = self.data["counts_leveled_background"].iloc[
            self.index_beamstart:self.index_beamstop]
        self.param = curve_fit(
            biexp_decay, self.X, self.y,
            p0=[300, 300, -0.02, -0.02, self.time_beamstart,
                self.time_beamstart, 200])[0]
        print(self.param)
        print(self.time_beamstart)
        if area_high:
            baseline = self.param[6]
        else:
            baseline = 0
        self.area = quad(
            biexp_decay, self.time_beamstart, self.time_beamstop,
            args=(self.param[0], self.param[1], self.param[2], self.param[3],
                  self.param[4], self.param[5], baseline)
        )[0]

    def create_plot(self, area_high=False):
        """
        Add fitted curve to the dataframe.
        plot the fitted curve as well as leveled CIRD data
        """
        exponential_fit = biexp_decay(self.data["time"], *self.param)
        df_exp = pd.DataFrame({"exponential_fit": exponential_fit.values})
        self.data = pd.concat([self.data, df_exp], axis=1)
        self.data["exponential_fit"].mask(
            self.data["exponential_fit"].index < self.index_beamstart,
            inplace=True)
        self.data["exponential_fit"].mask(
            self.data["exponential_fit"].index > self.index_beamstop,
            inplace=True)
        self.data.fillna(0, inplace=True)
        self.data.set_index("time")[
            ["exponential_fit",
             "counts_leveled_background"]].plot(figsize=(16, 8))
        if area_high:
            plt.fill_between(self.X, biexp_decay(self.X, *self.param, 0))
        else:
            plt.fill_between(
                self.X, biexp_decay(self.X, *self.param), self.param[6])
        plt.text(0, 500, "AREA = {}".format(round(self.area)), fontsize=12)
        plt.show()

    def write_file(self):
        """Create a txt file for igor process."""
        message = input(
            "Do you want to save the data to p{}? ".format(self.filename)
            + "(Y/N): "
        )
        if message == 'y':
            file = open(
                self.filepath[:-12] + 'p' + self.filename, 'w+'
            )
            file.write(
                'IGOR\nWAVES, time_elapsed_{}, counts_{},'
                'curve_fit_{}, \nBegin\n'
                .format(self.filename[0:8], self.filename[0:8],
                        self.filename[0:8])
            )
            for col1, col2, col3 in self.data.as_matrix(
                ["time",
                 "counts_leveled_background",
                 "exponential_fit"]):
                file.write('{} {} {}\n'.format(col1, col2, col3))
            file.write('End\n')
            file.close()
            print('Success!')


def generate_dataframe(tds_before_cird, tds_after_cird, background):
    """
    combine all the processed data into a single dataframe for the
    convinient data output to a txt file.
    """
    temp_average = 1/2 * (tds_before_cird.temp + tds_after_cird.temp)
    counts_difference = tds_before_cird.counts_background_sub -\
        tds_after_cird.counts_background_sub
    data_dict = {
        'time_elapsed': tds_before_cird.time,
        'temperature_average': temp_average,
        'temperature_before': tds_before_cird.temp,
        'counts_raw_before': tds_before_cird.counts_raw,
        'counts_leveled_before': tds_before_cird.counts_leveled,
        'counts_sensitivity_corrected_before':
        tds_before_cird.counts_sensitivity_corrected,
        'counts_background_subtracted_before':
        tds_before_cird.counts_background_sub,
        'temperature_after': tds_after_cird.temp,
        'counts_raw_after': tds_after_cird.counts_raw,
        'counts_leveled_after': tds_after_cird.counts_leveled,
        'counts_sensitivity_corrected_after':
        tds_after_cird.counts_sensitivity_corrected,
        'counts_background_subtracted_after':
        tds_after_cird.counts_background_sub,
        'counts_difference': counts_difference,
        'temperature_background': background.temp,
        'counts_raw_background': background.counts_raw,
        'counts_leveled_background': background.counts_leveled,
    }
    data_processed = pd.DataFrame(data_dict)
    return data_processed


class TDS:
    """For TDS data processing"""
    def __init__(self, filename):
        self.filename = filename

    def load_data(self, dwell_time=0.2, show_plot=True):
        monthDict = generateDict()
        self.filepath = os.path.join(
            os.path.expanduser('~'), 'Dropbox (MIT)', 'littlemachine', '20' +
            self.filename[6:8], monthDict[self.filename[:3]] + '_' +
            self.filename[:3], self.filename)
        self.data = pd.read_csv(
            self.filepath, delim_whitespace=True, skipfooter=3, skiprows=3,
            names=["counts_raw", "temp"], engine="python"
        )
        if show_plot:
            plt.figure(figsize=(8, 8))
            plt.plot(self.data["temp"], self.data["counts_raw"])

    def level_background(self, dwell_time=0.2, npts=50, start_temp=95,
                         total_npts=870, temp_fluc_plot=False,
                         leveled_plot=True):
        startpoint = find_nearest(self.data["temp"], start_temp)
        if startpoint + total_npts > len(self.data["temp"]):
            raise ValueError('Please modify the start_temp or total_npts!')
        self.data = self.data.iloc[
            startpoint:total_npts+startpoint].reset_index(drop=True)
        self.data["time"] = [
            t for t in np.linspace(
                0, dwell_time * len(self.data.index), num=len(self.data.index),
                endpoint=False)
        ]
        self.counts_raw = self.data["counts_raw"]
        self.temp = self.data["temp"]
        self.time = self.data["time"]
        if temp_fluc_plot:
            plt.figure(figsize=(8, 8))
            plt.plot(np.diff(self.temp))
        # Level the plot.
        X = pd.concat([self.temp.iloc[0:npts], self.temp.iloc[-npts:]])
        y = pd.concat([
            self.counts_raw.iloc[0:npts], self.counts_raw.iloc[-npts:]])
        slope = linregress(X, y)[0]
        self.counts_leveled = self.counts_raw.sub(self.temp * slope)
        if leveled_plot:
            plt.figure(figsize=(8, 8))
            plt.plot(self.temp, self.counts_leveled)

    def shift_x_axis(self, npts):
        """Shift the spectrum down to the x-axis"""
        self.counts_leveled = self.counts_leveled -\
            np.mean(self.counts_leveled[-npts:])

    def background_subtraction(self, background, sensitivity=1.0,
                               show_plot=True):
        self.counts_background_sub = self.counts_leveled / sensitivity -\
            background.counts_leveled
        self.counts_sensitivity_corrected = self.counts_leveled / sensitivity
        if show_plot:
            plt.figure(figsize=(8, 8))
            plt.plot(self.temp, self.counts_leveled/sensitivity, linewidth=1,
                     label="Sensitivity adjusted TDS")
            plt.plot(background.temp, background.counts_leveled, linewidth=1,
                     label="Background TDS")
            plt.plot(self.temp, self.counts_background_sub, linewidth=1,
                     label="Final TDS")
            plt.legend()
            plt.ylabel('Counts')
            plt.xlabel('Temperature (K)')
            plt.grid()
            plt.title(self.filename + ', ramp rate = 2 K/sec')
            plt.show()
