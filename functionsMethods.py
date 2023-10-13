"""
DESCRIPTION:
Module containing all simple custom functions, which take simulation parameters and a single array as inputs, and
return one or two outputs. Methods functions are those directly called by analysis functions. The contents of this file 
are imported for use in other scripts.

NOTES:
The type hints are not enforced and only serve debugging purposes.

UPDATE:
Last edited by Neil Wesch on 12 October 2023.
"""

import numpy as np

"""
f_positiveFourierTranform()
### For an input array x the fourier transform over nWindow points is taken and normalized. Only the positive AC
components are kept, from df up to the Nyquist frequency. The corrective factor of 2
applied to account for omitting the negative frequencies from the symmetric spectrum is not necessary, as all end
quantities have the factor in the numerator and denominator. The sampling frequencies y for an array of length x with
timestep dt is calculated.
"""


def f_positiveFourierTransform(dt: float, nWindow:int, x):  # Fourier transform time-domain signal and keep only positive frequency compoents

    x_trans = np.fft.fft(x, nWindow, axis=0) / nWindow  # normalized Fourier transform of x with nWindow sampling points
    z = 2. * x_trans[1:x.size//2]  # keep first positive AC frequency up to Nyquist frequency (factor of 2 accounts for symmetrical negative power spectrum)
    y = np.fft.fftfreq(x.size, dt)
    y = y[1:x.size//2]  # keep first positive AC frequency up to Nyquist frequency

    return (z, y)  # returns the positive frequency Fourier amplitudes and the positive frequency values


"""
f_rolledPositiveFourierTransform()
### Performs the exact same operations as f_positiveFourierTransform with one major exception. The input x is first
rolled such that the maximum value is found at index zero. The roll has the effect of neutralizing the leading delay
between input and spike output. This is done to get the f_gaussianFilter not to filter more just because phase is
deterministicly drifting (with the exp(i omega 2 pi delay)) term however, this time delay is real and should be added
back to the phase before it is reported.
"""


def f_rolledPositiveFourierTransform(dt: float, nWindow: int, x):  # rolls function so maximum index is first, performs Fourier transform, only keeps positive frequencies

    x_roll = np.roll(x, -np.argmax(x)+1)  # takes the maximum and cyclically shifts it to the beginning of the array
    x_roll = np.fft.fft(x_roll, nWindow, axis=0)/nWindow  # performs a normalized discrete fourier transform
    z = 2. * x_roll[1:nWindow//2]  # keep all positive Fourier components up to Nyquist frequency, exclude DC component
    y = np.fft.fftfreq(nWindow, dt)  # calculate the frequency points associated with fft of period nWindow and step size 1/df
    y = y[1:nWindow//2]  # keep first AC frequency up to Nyquist frequency

    return (z, y)  # returns the positive frequency Fourier amplitudes and the positive frequency values


"""
f_gaussianFilter()
### A bank of gaussian filters is applied to the noisy input x. The noisy input has values at frequencies f. This is equal
to the variable freqPos, which is all positive frequencies up to Nyquist. Each Gaussian filter is centered at fPrime and
has a standard deviation of f/2pi. The filtered output z is evaluated at each point specified in the user defined freqFilt.
"""

def f_gaussianFilter(x, f, freqFilt):  # applies Gaussian filters at freqFilt to signal x evaluated at points f

    z = np.zeros(freqFilt.size, dtype=complex)  # define vector to hold complex output
    for i, fPrime in enumerate(freqFilt):  # iterate over every frequency specified by the user
        filter = np.exp(-2 * np.pi ** 2 * (f - fPrime) ** 2 / f ** 2)  # formula for Gaussian filter
        filter_norm = filter / np.sum(filter)  # normalization value
        filter_product = np.multiply(x, np.transpose(filter_norm))  # apply filter and normalize
        z[int(i)] = np.sum(filter_product)  # denoised value of x at each value of freqFilt

    return (z)  # returns input array x smoothed across frequencies (denoised)


"""
f_confidenceInterval()
### Returns the ci percentile sorted value of input matrix x for each column. y is the ci percentile index, and z is an
output of the same length as x. At each time/frequency point of x, that is each column, the values are sorted in
ascending order and the yth value is found. Returns a complex quantity.
"""


def f_confidenceInterval(ci: float, x):  # takes the matrix x and determines the ci highest value at every point

    y = int(np.ceil(ci * x.shape[0]))  # find the index where ci highest value will be located after sorting in ascending order
    z = [np.sort(x[:, i], axis=None)[y] for i in range(x.shape[1])] # iterate over all frequency points, sort the values in ascending order, take the ci highest value

    return (z)  # output vector with same length (number of columns) as input matrix


"""
f_confidenceInterval()
### Returns the ci percentile sorted value of input matrix x for each column. y is the ci percentile index, and z is an
output of the same length as x. At each time/frequency point of x, that is each column, the values are projected onto
the vector defined by g in the complex plane. For example, converts the gain and all bootstrapped gains into phasors,
projects the bootstrapped gain onto the complex axis defined by angle of the gain, and finds the upper ci percentile
value. Not this returns a complex quantity.
"""


def f_complexConfidenceInterval(ci: float, g, x):  # desired confidence interval ci, gain g, and bootstrapped gain x

    y = int(np.ceil(ci * x.shape[0]))  # find the index where ci highest value will be located after sorting in ascending order
    z = np.zeros(x.shape[1], dtype=complex)  # define vector to store the ci highest value in at each frequency point

    for i in range(x.shape[1]):  # iterate over all frequency points (those used to Gaussian filter at)

        gAng = np.angle(g[i])  # angle of the gain phasor with real axis
        xMag = np.abs(x[:, i])  # magnitude of the bootstrapped gain phasor
        xAng = np.angle(x[:, i])  # angle of the bootstrapped gain phasor with real axis

        A = xMag  # reassign for notational clairty
        phi = xAng-gAng  # difference in phasor angle between gain and boostrapped gain
        xProj = A * np.exp(1j*2*np.pi*phi)  # the bootstrapped gain is projected along the direction of the gain and the new magnitude calculated

        z[i] = np.sort(np.abs(xProj))[y]  # set the confidence curve at each point equal to the ci percentile value

    return (z)  # single confidence interval gain curve


"""
f_phasor()
### Converts a complex number of the form 1+1j to polar form of A*exp(1j*phi).
"""


def f_phasor(x: complex):  # takes complex vector quantity expressed as 1+1j

    A = np.abs(x)  # the vector magniutde of x
    phi = np.angle(x)  # the angle relative to the real axis

    return (A, phi)  # returns components of phasor A*np.exp(-1j*phi)


"""
f_localVariationCoefficient()
### Calculates the spike variability coefficient as described in Shinomoto (2005) BioSystems. It is a measure of spike
train variability on a spectrum from completely regular (deterministic) to completely random (Poissonian). The value for
each spike is based on the preceeding and procedding spike, and summed over all spikes.
"""

def f_localVariationCoefficient(x):  # calculates local variation from spike time array x

    y = len(x)-1  # number of interspike intervals is one fewer than number of spikes
    z = np.mean([3*(x[n]-x[n+1])**2/(x[n]+x[n+1])**2 for n in range(y)])  # formula for Lv from Shinomoto (2005)

    return (z)  # returns single float between zero and one

