###--- IMPORT MODULES ---###

import numpy as np
import json

###--- IMPORT FUNCTIONS FROM DATA ANALYSIS MODULE ---###

from functionsAnalysis import f_thesaurusData
from functionsAnalysis import f_inputAutoCorrelation
from functionsAnalysis import f_spikeTriggeredAverageCurrent
from functionsAnalysis import f_dynamicGain
from functionsAnalysis import f_gainConfidenceIntervals
from functionsAnalysis import f_gainNoiseFloor
from functionsAnalysis import f_baseballCard

###--- ANALYSIS PARAMETERS ---###

with open('abfDataInfo.json', 'r') as file:  # open JSON file containing the data to be analyzed
    params = json.load(file)  # these two params must be specified correctly for code to operate as desired

dataNames = params['dataNames']  # explicity provide names of .abf files, without extension and ending with a forwardslash

pathName = params['pathName']  # explicitly set working path, recommend using same directory where .abf files are saved

with open('abfAnalysisParams.json', 'r') as file:  # open JSON file containing the analysis parameter values
    params = json.load(file)  # sensible default values have already been specified, see comments beside parameters below

detect = params['detect']  # [mV] user specified spike detection value, postive crossings of this value are registered as spikes
onset = params['onset']  # [mV] user specified voltage crossover from sub- to peri-threshold regions, spike onset voltage
rapid = params['rapid']  # [mV/s] user specified onset rapidness; membrane potential when dV/dt equals this value is the threshold voltage

fup = params['fup']  # [Hz] upper frequency limit to denoise and plot dynamic gain
tSpike = params['tSpike']  # [s] time window around each spike to use for phase plot
tTrace = params['tTrace']  # [s] number of seconds of first sweep current and voltage to be plotted
tWindow = params['tWindow']  # [s] window length for spike triggered average; frequency resolution is proportional

nFreq = int(params['nFreq'])  # [int] number of frequency points filtered quantities are evaluated at between lowest freqeunecy and fup in Hertz
nBtsp = int(params['nBtsp'])  # [int] number of bootstrapped gain curves to calculate
nNofl = int(params['nNofl'])  # [int] number of noise floor gain curves to calculate

###--- ABF TO NUMPY ---###

tauNames, groupNames, cellNames, nPoints, dt, tRecord = f_thesaurusData(pathName, dataNames)  # extracts all neccessary data from .abf files
print('>>> f_thesaurusData')

###--- DERIVED QUANTITIES ---###

nWindow = int(np.ceil(tWindow/dt))  # [int] number of points in the peak centered window
nTrace = int(np.ceil(tTrace/dt))  # [int] number of points to display for voltage and current traces
df = np.round(1./dt/nWindow, 5)  # [Hz] frequency step
freqPos = df*np.arange(1, nWindow//2)  # vector of all positive frequencies (excluding DC and Nyquist)
freqFilt = np.logspace(np.log10(df), np.log10(fup), nFreq)  # logarithmically spaced frequency points at which to apply Gaussian filters

###--- RUN DATA ANALYSIS ---###

f_inputAutoCorrelation(pathName, tauNames, groupNames, nPoints, nWindow)
print('>>> f_inputAutoCorrelation')

f_spikeTriggeredAverageCurrent(pathName, tauNames, groupNames, nPoints, nWindow, dt, detect, rapid, tSpike)
print('>>> f_spikeTriggeredAverageCurrent')

f_dynamicGain(pathName, tauNames, groupNames, nWindow, dt, freqPos, freqFilt)
print('>>> f_dyanmicGain')

f_gainConfidenceIntervals(pathName, tauNames, groupNames, nWindow, nBtsp, dt, freqPos, freqFilt)
print('>>> f_gainConfidenceIntervals')

f_gainNoiseFloor(pathName, tauNames, groupNames, nWindow, nNofl, dt, freqPos, freqFilt)
print('>>> f_gainNoiseFloor')

f_baseballCard(pathName, dataNames, tauNames, groupNames, nTrace, dt, fup, freqFilt)
print('>>> f_baseballCard')
