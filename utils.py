__name__ = "Point process example utilities"

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def jitter():
    return (np.random.rand()-0.5)*0.2

def slim_aspect():
    w, h = plt.figaspect(.1)
    return (w,1)

def find_percent_limits(counts, fraction=0.01):
    counter = Counter(counts)
    cutoff = len(counts) * fraction
    start = min(counter)
    while counter[start] < cutoff:
        start += 1
    end = max(counter)
    while counter[end] < cutoff:
        end -= 1
    return start, end

def _add_point(points, scale=2.0):
    wait_time = np.random.exponential(scale)
    if len(points) == 0:
        last = 0
    else:
        last = points[-1]
    points.append(last + wait_time)


def sample_poisson_process(window_size=100, scale=2.0):
    points = []
    _add_point(points, scale)
    while points[-1] < window_size:
        _add_point(points, scale)
    return points

def plot_series(series, title, window=[0,100]):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=slim_aspect())

    ax.set(xlim=window, ylim=[-1,1], title=title, xlabel="Time", yticks=[])
    ax.xaxis.set_label_coords(0.95,-0.3)
    ax.scatter(x=series, y=[jitter() for _ in series], marker="o", color="black")

def plot_spatial(data, title, xwindow, ywindow):
    xsize = xwindow[1] - xwindow[0]
    ysize = ywindow[1] - ywindow[0]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=plt.figaspect(ysize / xsize))

    ax.set(xlim=xwindow, ylim=ywindow, title=title)
    ax.scatter(x=data[:,0], y=data[:,1])

def histogram_with_expected(counts, expected, title):
    """Plot a histogram of the counts with an overlayed expected distribution.
    The counts should be samples from a discrete probability distribution.

    counts - an array or list of data to be binned.  Should be small integers.
    expected - a dict from integer count to probability
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=plt.figaspect(.6))

    start, end = find_percent_limits(counts, 0.001)
    ax.set(xlim=[start, end+1], title=title, xlabel="Count", ylabel="Frequency")
    xdata = np.arange(start, end+1)
    ax.xaxis.set_ticklabels(xdata)
    ax.xaxis.set_ticks(xdata + 0.5)
    ax.hist(counts, bins=range(0, max(counts)+1))
    data = []
    for x in xdata:
        if x in expected:
            data.append(expected[x] * len(counts))
        else:
            data.append(0)
    ax.plot(xdata + 0.5, data, marker="o", color="black")

def histogram_continuous(data, bin_size, expected, title):
    """Plot a histogram of samples drawn from a continuous probability
    distribution.  Will bin the data automatically into bins of equal size.

    data - list of samples
    bin_size - width of each bin
    expected - dict from start point of each bin to probability.  The start point
    of each bin should be an integer multiple of bin_size
    """
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=plt.figaspect(.6))

    def round_bin(x):
        return int(np.round(x / bin_size))
    def assign_bin(x):
        """Return n iff n * bin_size <= x < (n+1) * bin_size"""
        return int(np.round( x / bin_size - 0.5 ))
    
    ex = { round_bin(x) : expected[x] for x in expected }
    counts = [ assign_bin(x) for x in data ]
    start, end = find_percent_limits(counts, 0.001)
    bins = [ x * bin_size for x in range(start, end+1) ]
    ax.set(xlim=[start * bin_size, (end+1) * bin_size], title=title,
        ylabel="Frequency")
    ax.hist(data, bins=bins)
    x, y = [], []
    for t in range(start, end+1):
        x.append( (t+0.5) * bin_size )
        if t in ex:
            y.append(ex[t] * len(data))
        else:
            y.append(0)
    ax.plot(x,y, marker='o', color="black")
